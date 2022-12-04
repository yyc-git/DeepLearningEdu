'use strict';

var NP$Cnn = require("../NP.bs.js");
var Caml_int32 = require("rescript/lib/js/caml_int32.js");
var ImmutableSparseMap$Cnn = require("../sparse_map/ImmutableSparseMap.bs.js");

function computeOutputSize(inputSize, filterSize, zeroPadding, stride) {
  return Caml_int32.div((inputSize - filterSize | 0) + (zeroPadding << 1) | 0, stride) + 1 | 0;
}

function getConvolutionRegion2D(input, rowIndex, colIndex, filterWidth, filterHeight, stride) {
  return NP$Cnn.getMatrixRegion(Math.imul(colIndex, stride), filterWidth, Math.imul(rowIndex, stride), filterHeight, input);
}

function getConvolutionRegion3D(inputs, rowIndex, colIndex, filterWidth, filterHeight, stride) {
  return ImmutableSparseMap$Cnn.map(inputs, (function (input) {
                return getConvolutionRegion2D(input, rowIndex, colIndex, filterWidth, filterHeight, stride);
              }));
}

function createPreviousLayerDeltaMap(param) {
  return NP$Cnn.zeroMatrixMap(param[0], param[2], param[1]);
}

exports.computeOutputSize = computeOutputSize;
exports.getConvolutionRegion2D = getConvolutionRegion2D;
exports.getConvolutionRegion3D = getConvolutionRegion3D;
exports.createPreviousLayerDeltaMap = createPreviousLayerDeltaMap;
/* No side effect */
