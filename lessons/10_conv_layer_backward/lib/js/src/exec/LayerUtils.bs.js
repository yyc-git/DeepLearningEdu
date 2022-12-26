'use strict';

var NP$8_cnn = require("./NP.bs.js");
var Caml_int32 = require("rescript/lib/js/caml_int32.js");
var ImmutableSparseMap$8_cnn = require("./sparse_map/ImmutableSparseMap.bs.js");

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

function createLastLayerDeltaMap(param) {
  return NP$8_cnn.zeroMatrixMap(param[0], param[2], param[1]);
}

exports.computeOutputSize = computeOutputSize;
exports.getConvolutionRegion2D = getConvolutionRegion2D;
exports.getConvolutionRegion3D = getConvolutionRegion3D;
exports.createLastLayerDeltaMap = createLastLayerDeltaMap;
/* No side effect */
