

import * as LayerUtils$8_cnn from "../LayerUtils.bs.js";

function create(inputWidth, inputHeight, depthNumber, filterWidth, filterHeight, stride) {
  var outputWidth = LayerUtils$8_cnn.computeOutputSize(inputWidth, filterWidth, 0, stride);
  var outputHeight = LayerUtils$8_cnn.computeOutputSize(inputHeight, filterHeight, 0, stride);
  return {
          inputWidth: inputWidth,
          inputHeight: inputHeight,
          depthNumber: depthNumber,
          filterWidth: filterWidth,
          filterHeight: filterHeight,
          stride: stride,
          outputWidth: outputWidth,
          outputHeight: outputHeight
        };
}

function forward(state, inputs) {
  return 1;
}

export {
  create ,
  forward ,
  
}
/* No side effect */
