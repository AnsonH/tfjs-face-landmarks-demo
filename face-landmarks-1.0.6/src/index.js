/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import "@tensorflow/tfjs-backend-webgl";

import { Camera } from "./camera";
import { setupDatGui } from "./option_panel";
import { STATE, createDetector } from "./shared/params";
import { setupStats } from "./shared/stats_panel";
import { setBackendAndEnvFlags } from "./shared/util";
import { getFaceDirection } from "./shared/helpers";

let detector, camera, stats;
let startInferenceTime,
  numInferences = 0;
let inferenceTimeSum = 0,
  lastPanelUpdate = 0;
let rafId;

const faceBoxDiv = document.getElementById("face-box");
const faceDirection = document.getElementById("face-direction");

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimateFaceStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimateFaceStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
      1000.0 / averageInferenceTime,
      120 /* maxValue */
    );
    lastPanelUpdate = endInferenceTime;
  }
}

const RENDER_PREDICTION_INTERVAL_FRAME = 10;
let renderPredictionFrameCount = 0;

function renderFacePredictionStats(faces) {
  ++renderPredictionFrameCount;

  const face = faces?.[0];

  if (renderPredictionFrameCount % RENDER_PREDICTION_INTERVAL_FRAME === 0) {
    faceBoxDiv.innerText = JSON.stringify(face?.box, (_, value) => {
      return typeof value === "number" ? value.toFixed(1) : value;
    });
    faceDirection.innerText = JSON.stringify(
      getFaceDirection(face),
      (_, value) => {
        return typeof value === "number" ? value.toFixed(1) : value;
      }
    );

    renderPredictionFrameCount = 0;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let faces = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimateFaces.
    beginEstimateFaceStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      faces = await detector.estimateFaces(camera.video, {
        flipHorizontal: false,
      });
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimateFaceStats();
  }

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old model,
  // which shouldn't be rendered.
  if (faces && faces.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(
      faces,
      STATE.modelConfig.triangulateMesh,
      STATE.modelConfig.boundingBox
    );
  }

  renderFacePredictionStats(faces);
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
}

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);

  await setupDatGui(urlParams);

  stats = setupStats();

  camera = await Camera.setupCamera(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();

  renderPrediction();
}

app();
