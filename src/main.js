import './style.css';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Delaunay } from 'd3-delaunay';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
const SIZE = window.matchMedia('(max-width: 600px)').matches ? 480 : 720;
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const FEATURE_POINTS = new Set([
  ...FACE_OVAL, 1, 2, 4, 5, 6, 9, 13, 14, 17, 33, 37, 39, 40, 46, 52, 53, 55, 61, 63, 65, 66, 70, 78, 80, 81, 82, 84, 87, 88, 91, 95,
  105, 107, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 178, 181, 185, 191, 246, 249, 263, 267, 269, 270, 276,
  282, 283, 285, 291, 293, 295, 296, 300, 308, 310, 311, 312, 314, 317, 318, 321, 324, 334, 336, 362, 373, 374, 380, 381, 382,
  384, 385, 386, 387, 388, 390, 398, 402, 405, 409, 415, 466
]);
const FULL_POINTS = Array.from({ length: 468 }, (_, i) => i).filter((i) => i % 4 === 0 || FEATURE_POINTS.has(i));
const FAST_POINTS = Array.from(new Set([
  ...FACE_OVAL.filter((_, index) => index % 2 === 0),
  1, 4, 6, 9, 13, 14, 17, 33, 39, 52, 61, 66, 70, 78, 82, 87, 91, 105, 133, 145, 152, 159, 173, 178, 185,
  263, 269, 282, 291, 296, 300, 308, 312, 317, 321, 334, 362, 374, 386, 398, 402, 409, 415
]));

const canvas = document.querySelector('#output');
canvas.width = canvas.height = SIZE;
const ctx = canvas.getContext('2d');
const video = document.querySelector('#webcam');
const image = document.querySelector('#jacob');
const hairImage = document.querySelector('#hair');
const startButton = document.querySelector('#start');
const trackingLabel = document.querySelector('#tracking-label');
const fracture = document.querySelector('#fracture');
const smoothing = document.querySelector('#smoothing');
const gridToggle = document.querySelector('#grid-toggle');
const fastToggle = document.querySelector('#fast-toggle');
const warpCanvas = document.createElement('canvas');
warpCanvas.width = warpCanvas.height = SIZE;
const warpCtx = warpCanvas.getContext('2d');
const hairCanvas = document.createElement('canvas');
hairCanvas.width = hairCanvas.height = SIZE;
const hairCtx = hairCanvas.getContext('2d', { willReadFrequently: true });

let landmarker;
let initialization;
let sourcePoints;
let sourceHeadPose;
let triangles;
let selectedPoints = FULL_POINTS;
let smoothPoints;
let lastVideoTime = -1;
let gridVisible = false;
let active = false;
let benchmarkSamples = [];
let benchmarkFrames = 0;
let benchmarkComplete = false;

function coverTransform(width, height) {
  const scale = Math.max(SIZE / width, SIZE / height);
  return { scale, dx: (SIZE - width * scale) / 2, dy: (SIZE - height * scale) / 2 };
}

function mapLandmarks(landmarks, width, height, mirror = false) {
  const { scale, dx, dy } = coverTransform(width, height);
  return landmarks.map((point) => {
    const x = point.x * width * scale + dx;
    return [mirror ? SIZE - x : x, point.y * height * scale + dy];
  });
}

function pathFrom(points, indices = null) {
  const path = new Path2D();
  const sequence = indices || points.map((_, index) => index);
  sequence.forEach((index, position) => {
    const point = points[index];
    if (position === 0) path.moveTo(point[0], point[1]);
    else path.lineTo(point[0], point[1]);
  });
  path.closePath();
  return path;
}

function affine(source, target) {
  const [s0, s1, s2] = source;
  const [t0, t1, t2] = target;
  const det = s0[0] * (s1[1] - s2[1]) + s1[0] * (s2[1] - s0[1]) + s2[0] * (s0[1] - s1[1]);
  if (Math.abs(det) < 0.01) return null;
  const a = (t0[0] * (s1[1] - s2[1]) + t1[0] * (s2[1] - s0[1]) + t2[0] * (s0[1] - s1[1])) / det;
  const c = (t0[0] * (s2[0] - s1[0]) + t1[0] * (s0[0] - s2[0]) + t2[0] * (s1[0] - s0[0])) / det;
  const e = (t0[0] * (s1[0] * s2[1] - s2[0] * s1[1]) + t1[0] * (s2[0] * s0[1] - s0[0] * s2[1]) + t2[0] * (s0[0] * s1[1] - s1[0] * s0[1])) / det;
  const b = (t0[1] * (s1[1] - s2[1]) + t1[1] * (s2[1] - s0[1]) + t2[1] * (s0[1] - s1[1])) / det;
  const d = (t0[1] * (s2[0] - s1[0]) + t1[1] * (s0[0] - s2[0]) + t2[1] * (s1[0] - s0[0])) / det;
  const f = (t0[1] * (s1[0] * s2[1] - s2[0] * s1[1]) + t1[1] * (s2[0] * s0[1] - s0[0] * s2[1]) + t2[1] * (s0[0] * s1[1] - s1[0] * s0[1])) / det;
  return [a, b, c, d, e, f];
}

function triangleArea([a, b, c]) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function headPose(points) {
  const left = points[234];
  const right = points[454];
  const top = points[10];
  const dx = right[0] - left[0];
  const dy = right[1] - left[1];
  return {
    x: (left[0] + right[0]) / 2,
    y: top[1],
    width: Math.hypot(dx, dy),
    angle: Math.atan2(dy, dx)
  };
}

function prepareHair() {
  const fit = coverTransform(hairImage.naturalWidth, hairImage.naturalHeight);
  hairCtx.clearRect(0, 0, SIZE, SIZE);
  hairCtx.drawImage(hairImage, fit.dx, fit.dy, hairImage.naturalWidth * fit.scale, hairImage.naturalHeight * fit.scale);
  const pixels = hairCtx.getImageData(0, 0, SIZE, SIZE);
  for (let i = 0; i < pixels.data.length; i += 4) {
    const red = pixels.data[i];
    const green = pixels.data[i + 1];
    const blue = pixels.data[i + 2];
    const cyanDistance = Math.hypot(red - 7, green - 158, blue - 187);
    pixels.data[i + 3] = Math.max(0, Math.min(255, (cyanDistance - 18) * 12));
  }
  hairCtx.clearRect(0, 0, SIZE, SIZE);
  hairCtx.putImageData(pixels, 0, 0);
}

function drawHair(targetPoints) {
  if (!sourceHeadPose) return;
  const target = headPose(targetPoints);
  const scale = target.width / sourceHeadPose.width;
  const mirrored = Math.cos(target.angle) * Math.cos(sourceHeadPose.angle) < 0;
  ctx.save();
  ctx.translate(target.x, target.y);
  ctx.rotate(target.angle - sourceHeadPose.angle);
  ctx.scale(scale, mirrored ? -scale : scale);
  ctx.translate(-sourceHeadPose.x, -sourceHeadPose.y);
  ctx.drawImage(hairCanvas, 0, 0);
  ctx.restore();
}

function drawWarp(targetPoints) {
  warpCtx.clearRect(0, 0, SIZE, SIZE);
  warpCtx.imageSmoothingEnabled = true;
  const sourceFit = coverTransform(image.naturalWidth, image.naturalHeight);
  for (const triangle of triangles) {
    const source = triangle.map((index) => sourcePoints[index]);
    const target = triangle.map((index) => targetPoints[index]);
    const sourceArea = triangleArea(source);
    const targetArea = triangleArea(target);
    if (Math.abs(targetArea) < 0.4 || Math.abs(targetArea / sourceArea) > 12) continue;
    const transform = affine(source, target);
    if (!transform || transform.some((value) => !Number.isFinite(value))) continue;
    warpCtx.save();
    warpCtx.beginPath();
    warpCtx.moveTo(target[0][0], target[0][1]);
    warpCtx.lineTo(target[1][0], target[1][1]);
    warpCtx.lineTo(target[2][0], target[2][1]);
    warpCtx.closePath();
    warpCtx.clip();
    warpCtx.setTransform(...transform);
    warpCtx.drawImage(image, sourceFit.dx, sourceFit.dy, image.naturalWidth * sourceFit.scale, image.naturalHeight * sourceFit.scale);
    warpCtx.restore();
  }
}

function hash(index) {
  const value = Math.sin(index * 91.733) * 43758.5453;
  return value - Math.floor(value);
}

function render(targetPoints, time) {
  drawWarp(targetPoints);
  ctx.fillStyle = '#079db8';
  ctx.fillRect(0, 0, SIZE, SIZE);
  drawHair(targetPoints);

  const sites = selectedPoints.map((index) => targetPoints[index]);
  const voronoi = Delaunay.from(sites).voronoi([0, 0, SIZE, SIZE]);
  const facePath = pathFrom(targetPoints, FACE_OVAL);
  const strength = Number(fracture.value);

  ctx.save();
  ctx.clip(facePath);
  sites.forEach((site, index) => {
    const polygon = voronoi.cellPolygon(index);
    if (!polygon) return;
    const cellPath = pathFrom(polygon.slice(0, -1));
    const phase = hash(index) * Math.PI * 2;
    const pulse = Math.sin(time * 0.0014 + phase) * strength * 0.18;
    const angle = (hash(index + 700) - 0.5) * strength * 0.0015;
    ctx.save();
    ctx.clip(cellPath);
    ctx.translate(site[0], site[1]);
    ctx.rotate(angle);
    ctx.translate(-site[0] + Math.cos(phase) * pulse, -site[1] + Math.sin(phase) * pulse);
    ctx.drawImage(warpCanvas, 0, 0);
    ctx.restore();

    if (gridVisible) {
      ctx.save();
      ctx.strokeStyle = 'rgba(217,255,67,.72)';
      ctx.lineWidth = 0.75;
      ctx.stroke(cellPath);
      ctx.restore();
    }
  });
  ctx.restore();
}

function smoothLandmarks(points) {
  if (!smoothPoints) {
    smoothPoints = points.map((point) => [...point]);
    return smoothPoints;
  }
  const alpha = Number(smoothing.value);
  points.forEach((point, index) => {
    smoothPoints[index][0] += (point[0] - smoothPoints[index][0]) * alpha;
    smoothPoints[index][1] += (point[1] - smoothPoints[index][1]) * alpha;
  });
  return smoothPoints;
}

async function initialize() {
  if (initialization) return initialization;
  initialization = initializeModel();
  return initialization;
}

async function initializeModel() {
  trackingLabel.textContent = 'LOADING FACE MODEL';
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const options = {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'IMAGE',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  };
  try {
    landmarker = await FaceLandmarker.createFromOptions(vision, options);
  } catch {
    options.baseOptions.delegate = 'CPU';
    landmarker = await FaceLandmarker.createFromOptions(vision, options);
  }
  const result = landmarker.detect(image);
  if (!result.faceLandmarks.length) throw new Error('Jacob’s face could not be detected.');
  sourcePoints = mapLandmarks(result.faceLandmarks[0], image.naturalWidth, image.naturalHeight);
  sourceHeadPose = headPose(sourcePoints);
  prepareHair();
  rebuildMesh();
  render(sourcePoints, 0);
  trackingLabel.textContent = 'HIS BODY IS READY';
}

async function startCamera() {
  startButton.disabled = true;
  startButton.querySelector('span').textContent = 'CONNECTING…';
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is not supported in this browser.');
    await initialize();
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: SIZE }, height: { ideal: SIZE } }, audio: false });
    video.srcObject = stream;
    await video.play();
    await landmarker.setOptions({ runningMode: 'VIDEO' });
    setFastMode(false);
    benchmarkSamples = [];
    benchmarkFrames = 0;
    benchmarkComplete = false;
    active = true;
    startButton.querySelector('span').textContent = 'CAMERA ACTIVE';
    trackingLabel.textContent = 'PERFORMANCE TEST • FIND FACE';
    requestAnimationFrame(track);
  } catch (error) {
    console.error(error);
    trackingLabel.textContent = error.name === 'NotAllowedError' ? 'CAMERA ACCESS DENIED' : 'CAMERA UNAVAILABLE';
    startButton.disabled = false;
    startButton.querySelector('span').textContent = 'TRY AGAIN';
  }
}

function track(time) {
  if (!active) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const startedAt = performance.now();
    const result = landmarker.detectForVideo(video, startedAt);
    if (result.faceLandmarks.length) {
      const points = mapLandmarks(result.faceLandmarks[0], video.videoWidth, video.videoHeight, true);
      render(smoothLandmarks(points), time);
      if (!benchmarkComplete) updateBenchmark(performance.now() - startedAt);
      else trackingLabel.textContent = 'TRACKING • LIVE';
    } else {
      trackingLabel.textContent = 'FACE NOT FOUND';
      smoothPoints = null;
    }
  }
  requestAnimationFrame(track);
}

function updateBenchmark(duration) {
  benchmarkFrames += 1;
  if (benchmarkFrames > 4) benchmarkSamples.push(duration);
  trackingLabel.textContent = `PERFORMANCE TEST • ${Math.min(100, Math.round(benchmarkSamples.length / 18 * 100))}%`;
  if (benchmarkSamples.length < 18) return;

  const sorted = [...benchmarkSamples].sort((a, b) => a - b);
  const average = benchmarkSamples.reduce((total, sample) => total + sample, 0) / benchmarkSamples.length;
  const percentile90 = sorted[Math.floor(sorted.length * 0.9)];
  benchmarkComplete = true;
  setFastMode(average > 28 || percentile90 > 38);
  trackingLabel.textContent = 'TRACKING • LIVE';
}

function setFastMode(fast) {
  selectedPoints = fast ? FAST_POINTS : FULL_POINTS;
  fastToggle.textContent = `FAST MODE: ${fast ? 'ON' : 'OFF'}`;
  fastToggle.setAttribute('aria-pressed', String(fast));
  if (sourcePoints) rebuildMesh();
}

function rebuildMesh() {
  const sites = selectedPoints.map((index) => sourcePoints[index]);
  const mesh = Delaunay.from(sites);
  triangles = [];
  for (let i = 0; i < mesh.triangles.length; i += 3) {
    triangles.push([
      selectedPoints[mesh.triangles[i]],
      selectedPoints[mesh.triangles[i + 1]],
      selectedPoints[mesh.triangles[i + 2]]
    ]);
  }
}

startButton.addEventListener('click', startCamera);
gridToggle.addEventListener('click', () => {
  gridVisible = !gridVisible;
  gridToggle.textContent = `GRID: ${gridVisible ? 'ON' : 'OFF'}`;
  gridToggle.setAttribute('aria-pressed', String(gridVisible));
  if (!active && sourcePoints) render(sourcePoints, 0);
});
fastToggle.addEventListener('click', () => {
  benchmarkComplete = true;
  setFastMode(fastToggle.getAttribute('aria-pressed') !== 'true');
  if (sourcePoints) render(smoothPoints || sourcePoints, 0);
});

function initializeWhenAssetsReady() {
  if (!image.complete || !hairImage.complete) return;
  initialize().catch((error) => {
    console.error(error);
    trackingLabel.textContent = 'MODEL LOAD FAILED';
  });
}

image.addEventListener('load', initializeWhenAssetsReady, { once: true });
hairImage.addEventListener('load', initializeWhenAssetsReady, { once: true });
initializeWhenAssetsReady();
