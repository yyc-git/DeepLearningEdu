'use strict';

var NP$Cnn = require("../NP.bs.js");
var Matrix$Cnn = require("../Matrix.bs.js");
var ArraySt$Cnn = require("../ArraySt.bs.js");
var OptionSt$Cnn = require("../OptionSt.bs.js");
var LayerUtils$Cnn = require("./LayerUtils.bs.js");
var MatrixUtils$Cnn = require("../MatrixUtils.bs.js");
var ImmutableSparseMap$Cnn = require("../sparse_map/ImmutableSparseMap.bs.js");

function create(inputWidth, inputHeight, filterWidth, filterHeight, strideOpt, depthNumberOpt, param) {
  var stride = strideOpt !== undefined ? strideOpt : 1;
  var depthNumber = depthNumberOpt !== undefined ? depthNumberOpt : 1;
  var outputWidth = LayerUtils$Cnn.computeOutputSize(inputWidth, filterWidth, 0, stride);
  var outputHeight = LayerUtils$Cnn.computeOutputSize(inputHeight, filterHeight, 0, stride);
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

function forward(state, param, inputs, param$1) {
  var outputRow = state.outputHeight;
  var outputCol = state.outputWidth;
  var output = ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, state.depthNumber - 1 | 0), (function (outputMap, depthIndex) {
          var input = ImmutableSparseMap$Cnn.getExn(inputs, depthIndex);
          var output = ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, outputRow - 1 | 0), (function (output, rowIndex) {
                  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, outputCol - 1 | 0), (function (output, colIndex) {
                                return MatrixUtils$Cnn.setValue(output, NP$Cnn.max(LayerUtils$Cnn.getConvolutionRegion2D(input, rowIndex, colIndex, state.filterWidth, state.filterHeight, state.stride)), rowIndex, colIndex);
                              }), output);
                }), Matrix$Cnn.create(outputRow, outputCol, []));
          return ImmutableSparseMap$Cnn.set(outputMap, depthIndex, output);
        }), ImmutableSparseMap$Cnn.createEmpty(undefined, undefined));
  return [
          state,
          output,
          output
        ];
}

function bpDelta(param, param$1, nextLayerDelta, state) {
  var previousLayerOutput = OptionSt$Cnn.getExn(param$1[0]);
  var outputRow = state.outputHeight;
  var outputCol = state.outputWidth;
  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, state.depthNumber - 1 | 0), (function (currentLayerDeltaMap, depthIndex) {
                var input = ImmutableSparseMap$Cnn.getExn(previousLayerOutput, depthIndex);
                var currentLayerDelta = ImmutableSparseMap$Cnn.getExn(currentLayerDeltaMap, depthIndex);
                var currentLayerDelta$1 = ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, outputRow - 1 | 0), (function (currentLayerDelta, rowIndex) {
                        return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, outputCol - 1 | 0), (function (currentLayerDelta, colIndex) {
                                      var match = NP$Cnn.getMaxIndex(LayerUtils$Cnn.getConvolutionRegion2D(input, rowIndex, colIndex, state.filterWidth, state.filterHeight, state.stride));
                                      return MatrixUtils$Cnn.setValue(currentLayerDelta, NP$Cnn.getMatrixMapValue(nextLayerDelta, depthIndex, rowIndex, colIndex), Math.imul(rowIndex, state.stride) + match[1] | 0, Math.imul(colIndex, state.stride) + match[2] | 0);
                                    }), currentLayerDelta);
                      }), currentLayerDelta);
                return ImmutableSparseMap$Cnn.set(currentLayerDeltaMap, depthIndex, currentLayerDelta$1);
              }), LayerUtils$Cnn.createPreviousLayerDeltaMap([
                  state.depthNumber,
                  state.inputWidth,
                  state.inputHeight
                ]));
}

function backward(param, previousLayerData, nextLayerDelta, state) {
  return [
          bpDelta(undefined, previousLayerData, nextLayerDelta, state),
          undefined
        ];
}

function createGradientDataSum(state) {
  
}

function addToGradientDataSum(param, param$1) {
  
}

function update(state, param, param$1) {
  return state;
}

function createLayerData(state, activatorData) {
  return {
          layerName: "maxPooling",
          state: state,
          forward: forward,
          backward: backward,
          update: update,
          createGradientDataSum: createGradientDataSum,
          addToGradientDataSum: addToGradientDataSum,
          activatorData: activatorData,
          getWeight: (function (state) {
              
            }),
          getBias: (function (state) {
              
            })
        };
}

exports.create = create;
exports.forward = forward;
exports.bpDelta = bpDelta;
exports.backward = backward;
exports.createGradientDataSum = createGradientDataSum;
exports.addToGradientDataSum = addToGradientDataSum;
exports.update = update;
exports.createLayerData = createLayerData;
/* No side effect */
