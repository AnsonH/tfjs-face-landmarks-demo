/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import Stats from "stats.js";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

import { TRIANGULATION } from "./triangulation";
import {
  LEFT_NOSE,
  RIGHT_NOSE,
  getAdjustedFaceBox,
  getFaceDirection,
} from "./helpers";

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = "#32EEDB";
const RED = "#FF2C35";
const BLUE = "#157AB3";
const MAGENTA = "#ff00f7";

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function distance(a, b) {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model, ctx, videoWidth, videoHeight, video, canvas, rafID;
const faceBoxDiv = document.getElementById("face-box");
const adjustedFaceBox = document.getElementById("adjusted-face-box");
const faceDirection = document.getElementById("face-direction");

const VIDEO_SIZE = 500;
const mobile = isMobile();
const stats = new Stats();
const state = {
  backend: "webgl",
  maxFaces: 1,
  triangulateMesh: false,
  predictIrises: false,
};

function setupDatGui() {
  const gui = new dat.GUI();
  gui.add(state, "backend", ["webgl"]).onChange(async (backend) => {
    window.cancelAnimationFrame(rafID);
    await tf.setBackend(backend);
    requestAnimationFrame(renderPrediction);
  });

  gui.add(state, "triangulateMesh");
  gui.add(state, "predictIrises");
}

async function setupCamera() {
  video = document.getElementById("video");

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_SIZE,
      height: mobile ? undefined : VIDEO_SIZE,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const RENDER_PREDICTION_INTERVAL_FRAME = 20;
let renderPredictionFrameCount = 0;

function renderFacePredictionStats(faces) {
  ++renderPredictionFrameCount;

  const face = faces?.[0];

  if (renderPredictionFrameCount % RENDER_PREDICTION_INTERVAL_FRAME === 0) {
    console.log(face);

    faceBoxDiv.innerText = JSON.stringify(face?.boundingBox, (_, value) => {
      return typeof value === "number" ? value.toFixed(1) : value;
    });
    adjustedFaceBox.innerText = JSON.stringify(
      getAdjustedFaceBox(face),
      (_, value) => {
        return typeof value === "number" ? value.toFixed(1) : value;
      }
    );
    faceDirection.innerText = JSON.stringify(
      getFaceDirection(face),
      (_, value) => {
        return typeof value === "number" ? value.toFixed(1) : value;
      }
    );

    renderPredictionFrameCount = 0;
  }
}

async function drawPredictionBoundingBox(prediction) {
  ctx.strokeStyle = RED;
  ctx.lineWidth = 1;
  const {
    topLeft: [xMin, yMin],
    bottomRight: [xMax, yMax],
  } = prediction.boundingBox;
  drawPath(
    ctx,
    [
      [xMin, yMin],
      [xMax, yMin],
      [xMax, yMax],
      [xMin, yMax],
    ],
    true
  );
}

async function drawAdjustedFaceBox(prediction) {
  const { right, bottom, left, top } = getAdjustedFaceBox(prediction);

  ctx.strokeStyle = BLUE;
  ctx.lineWidth = 1;

  drawPath(
    ctx,
    [
      [left, top],
      [right, top],
      [right, bottom],
      [left, bottom],
    ],
    true
  );
}

async function renderPrediction() {
  stats.begin();

  const predictions = await model.estimateFaces({
    input: video,
    returnTensors: false,
    flipHorizontal: false,
    predictIrises: state.predictIrises,
  });
  ctx.drawImage(
    video,
    0,
    0,
    videoWidth,
    videoHeight,
    0,
    0,
    canvas.width,
    canvas.height
  );

  if (predictions.length > 0) {
    renderFacePredictionStats(predictions);

    predictions.forEach((prediction) => {
      const keypoints = prediction.scaledMesh;

      drawPredictionBoundingBox(prediction);
      drawAdjustedFaceBox(prediction);

      if (state.triangulateMesh) {
        ctx.strokeStyle = GREEN;
        ctx.lineWidth = 0.5;

        for (let i = 0; i < TRIANGULATION.length / 3; i++) {
          const points = [
            TRIANGULATION[i * 3],
            TRIANGULATION[i * 3 + 1],
            TRIANGULATION[i * 3 + 2],
          ].map((index) => keypoints[index]);

          drawPath(ctx, points, true);
        }
      } else {
        for (let i = 0; i < NUM_KEYPOINTS; i++) {
          let radius = 1;
          ctx.fillStyle = GREEN;

          if (LEFT_NOSE.includes(i) || RIGHT_NOSE.includes(i)) {
            ctx.fillStyle = MAGENTA;
            radius = 2;
          }

          const x = keypoints[i][0];
          const y = keypoints[i][1];

          ctx.beginPath();
          ctx.arc(x, y, radius, 0, 2 * Math.PI);
          ctx.fill();
        }
      }

      renderIrisKeypoints(keypoints);
    });
  }

  stats.end();
  rafID = requestAnimationFrame(renderPrediction);
}

function renderIrisKeypoints(keypoints) {
  if (keypoints.length > NUM_KEYPOINTS) {
    ctx.strokeStyle = RED;
    ctx.lineWidth = 1;

    const leftCenter = keypoints[NUM_KEYPOINTS];
    const leftDiameterY = distance(
      keypoints[NUM_KEYPOINTS + 4],
      keypoints[NUM_KEYPOINTS + 2]
    );
    const leftDiameterX = distance(
      keypoints[NUM_KEYPOINTS + 3],
      keypoints[NUM_KEYPOINTS + 1]
    );

    ctx.beginPath();
    ctx.ellipse(
      leftCenter[0],
      leftCenter[1],
      leftDiameterX / 2,
      leftDiameterY / 2,
      0,
      0,
      2 * Math.PI
    );
    ctx.stroke();

    if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
      const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
      const rightDiameterY = distance(
        keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
        keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]
      );
      const rightDiameterX = distance(
        keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
        keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]
      );

      ctx.beginPath();
      ctx.ellipse(
        rightCenter[0],
        rightCenter[1],
        rightDiameterX / 2,
        rightDiameterY / 2,
        0,
        0,
        2 * Math.PI
      );
      ctx.stroke();
    }
  }
}

async function main() {
  await tf.setBackend(state.backend);
  setupDatGui();

  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById("main").appendChild(stats.dom);

  await setupCamera();
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById("output");
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector(".canvas-wrapper");
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

  ctx = canvas.getContext("2d");
  // ctx.translate(canvas.width, 0);
  // ctx.scale(-1, 1);
  ctx.fillStyle = GREEN;
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 0.5;

  model = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
    {
      maxFaces: state.maxFaces,
      // modelUrl: "http://localhost:3000/facemesh-model.json",
      // detectorModelUrl: "http://localhost:3000/blazeface-model.json",
      // modelUrl: "https://www.bowtie.com.hk/assets/tfjs/facemesh/model.json",
      // detectorModelUrl:
      //   "https://www.bowtie.com.hk/assets/tfjs/blazeface/model.json",
    }
  );
  renderPrediction();
}

main();
