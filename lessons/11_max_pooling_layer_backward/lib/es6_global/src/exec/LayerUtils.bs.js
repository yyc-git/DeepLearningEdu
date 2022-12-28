

import * as NP$8_cnn from "./NP.bs.js";
import * as Caml_int32 from "../../../../../../node_modules/rescript/lib/es6/caml_int32.js";
import * as ImmutableSparseMap$8_cnn from "./sparse_map/ImmutableSparseMap.bs.js";

function computeOutputSize(inputSize, filterSize, zeroPadding, stride) {
  return Caml_int32.div((inputSize - filterSize | 0) + (zeroPadding << 1) | 0, stride) + 1 | 0;
}

function getConvolutionRegion2D(input, rowIndex, colIndex, filterWidth, filterHeight, stride) {
  return NP$8_cnn.getMatrixRegion(Math.imul(colIndex, stride), filterWidth, Math.imul(rowIndex, stride), filterHeight, input);
}

function getConvolutionRegion3D(inputs, rowIndex, colIndex, filterWidth, filterHeight, stride) {
  return ImmutableSparseMap$8_cnn.map(inputs, (function (input) {
                return getConvolutionRegion2D(input, rowIndex, colIndex, filterWidth, filterHeight, stride);
              }));
}

function createCurrentLayerDeltaMap(param) {
  return NP$8_cnn.zeroMatrixMap(param[0], param[2], param[1]);
}

export {
  computeOutputSize ,
  getConvolutionRegion2D ,
  getConvolutionRegion3D ,
  createCurrentLayerDeltaMap ,
  
}
/* No side effect */
