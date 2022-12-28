'use strict';

var NP$8_cnn = require("../NP.bs.js");
var Log$8_cnn = require("../Log.bs.js");
var Matrix$8_cnn = require("../Matrix.bs.js");
var ArraySt$8_cnn = require("../ArraySt.bs.js");
var LayerUtils$8_cnn = require("../LayerUtils.bs.js");
var MatrixUtils$8_cnn = require("../MatrixUtils.bs.js");
var ImmutableSparseMap$8_cnn = require("../sparse_map/ImmutableSparseMap.bs.js");

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
  var outputRow = state.outputHeight;
  var outputCol = state.outputWidth;
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, state.depthNumber - 1 | 0), (function (outputMap, depthIndex) {
                var input = ImmutableSparseMap$8_cnn.getExn(inputs, depthIndex);
                var output = ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, outputRow - 1 | 0), (function (output, rowIndex) {
                        return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, outputCol - 1 | 0), (function (output, colIndex) {
                                      return MatrixUtils$8_cnn.setValue(output, NP$8_cnn.max(LayerUtils$8_cnn.getConvolutionRegion2D(input, rowIndex, colIndex, state.filterWidth, state.filterHeight, state.stride)), rowIndex, colIndex);
                                    }), output);
                      }), Matrix$8_cnn.create(outputRow, outputCol, []));
                return ImmutableSparseMap$8_cnn.set(outputMap, depthIndex, output);
              }), ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined));
}

function bpDeltaMap(state, inputs, nextLayerDeltaMap) {
  var outputRow = state.outputHeight;
  var outputCol = state.outputWidth;
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, state.depthNumber - 1 | 0), (function (currentLayerDeltaMap, depthIndex) {
                var input = ImmutableSparseMap$8_cnn.getExn(inputs, depthIndex);
                var currentLayerDelta = ImmutableSparseMap$8_cnn.getExn(currentLayerDeltaMap, depthIndex);
                var currentLayerDelta$1 = ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, outputRow - 1 | 0), (function (currentLayerDelta, rowIndex) {
                        return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, outputCol - 1 | 0), (function (currentLayerDelta, colIndex) {
                                      var match = NP$8_cnn.getMaxIndex(LayerUtils$8_cnn.getConvolutionRegion2D(input, rowIndex, colIndex, state.filterWidth, state.filterHeight, state.stride));
                                      return MatrixUtils$8_cnn.setValue(currentLayerDelta, NP$8_cnn.getMatrixMapValue(nextLayerDeltaMap, depthIndex, rowIndex, colIndex), Math.imul(rowIndex, state.stride) + match[1] | 0, Math.imul(colIndex, state.stride) + match[2] | 0);
                                    }), currentLayerDelta);
                      }), currentLayerDelta);
                return ImmutableSparseMap$8_cnn.set(currentLayerDeltaMap, depthIndex, currentLayerDelta$1);
              }), LayerUtils$8_cnn.createCurrentLayerDeltaMap([
                  state.depthNumber,
                  state.inputWidth,
                  state.inputHeight
                ]));
}

function init(param) {
  var inputs = NP$8_cnn.createMatrixMapByDataArr([
        [
          [
            0,
            1,
            1,
            0
          ],
          [
            2,
            3,
            2,
            2
          ],
          [
            1,
            0,
            0,
            2
          ],
          [
            0,
            1,
            1,
            0
          ]
        ],
        [
          [
            1,
            0,
            2,
            2
          ],
          [
            0,
            5,
            0,
            2
          ],
          [
            1,
            2,
            1,
            2
          ],
          [
            1,
            0,
            0,
            0
          ]
        ]
      ]);
  var state = create(4, 4, 2, 2, 2, 2);
  return [
          inputs,
          state
        ];
}

function test(param) {
  var outputMap = forward(param[1], param[0]);
  Log$8_cnn.printForDebug([
        "f:",
        outputMap
      ]);
  
}

var Test = {
  init: init,
  test: test
};

test(init(undefined));

exports.create = create;
exports.forward = forward;
exports.bpDeltaMap = bpDeltaMap;
exports.Test = Test;
/*  Not a pure module */
