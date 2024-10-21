export const RIGHT_NOSE = [
  6, 197, 195, 5, 4, 1, 274, 457, 438, 344, 360, 420, 437, 343, 412, 351,
];

export const LEFT_NOSE = [
  6, 197, 195, 5, 4, 1, 44, 237, 218, 115, 131, 198, 217, 114, 188, 122,
];

export const SILHOUETTE = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 140, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109,
];

export const MAX_MESH_POINT = 468;

export const getSelectedScaledMesh = (prediction, meshPointIdx) => {
  if (!prediction) {
    return [];
  }
  if (prediction.keypoints && prediction.keypoints.length > 0) {
    const filteredIdx = meshPointIdx.filter((n) => n < MAX_MESH_POINT);

    return filteredIdx.map((idx) => prediction.keypoints[idx]);
  }
  throw new EmptyDetectionResultError();
};

export const getAdjustedFaceBox = (prediction, videoOnCanvasSizing) => {
  const coords = getSelectedScaledMesh(prediction, SILHOUETTE).reduce(
    (obj, coord) => {
      return { x: [...obj.x, coord[0]], y: [...obj.y, coord[1]] };
    },
    { x: [], y: [] }
  );

  const right = Math.max(...coords.x);
  const left = Math.min(...coords.x);
  const bottom = Math.max(...coords.y);
  const top = Math.min(...coords.y);
  const width = right - left;
  const height = bottom - top;

  return { left, top, right, bottom, width, height };
};

export const getPolygonArea = (polygon) => {
  const area = polygon.reduce((sum, currentPoint, id) => {
    const { x: currX, y: currY } = currentPoint;
    const nextPoint = id === polygon.length - 1 ? polygon[0] : polygon[id + 1];
    const { x: nextX, y: nextY } = nextPoint;
    const determinant = (currX * nextY - currY * nextX) / 2;
    return sum + determinant;
  }, 0);
  return Math.abs(area);
};

const FACE_DIRECTION = {
  Center: "center",
  Left: "left",
  Right: "right",
};

export const getFaceDirection = (prediction, factor = 3) => {
  const leftNoseArea = getPolygonArea(
    getSelectedScaledMesh(prediction, LEFT_NOSE)
  );
  const rightNoseArea = getPolygonArea(
    getSelectedScaledMesh(prediction, RIGHT_NOSE)
  );

  const leftToRightRatio = leftNoseArea / rightNoseArea;

  const direction =
    leftToRightRatio < factor
      ? leftToRightRatio > 1 / factor
        ? FACE_DIRECTION.Center
        : FACE_DIRECTION.Right
      : FACE_DIRECTION.Left;

  return { leftNoseArea, rightNoseArea, leftToRightRatio, direction };
};
