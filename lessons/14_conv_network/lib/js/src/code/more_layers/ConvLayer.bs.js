'use strict';

var Curry = require("rescript/lib/js/curry.js");
var NP$Cnn = require("../NP.bs.js");
var Filter$Cnn = require("./Filter.bs.js");
var Matrix$Cnn = require("../Matrix.bs.js");
var Random$Cnn = require("../Random.bs.js");
var ArraySt$Cnn = require("../ArraySt.bs.js");
var Caml_option = require("rescript/lib/js/caml_option.js");
var OptionSt$Cnn = require("../OptionSt.bs.js");
var LayerUtils$Cnn = require("./LayerUtils.bs.js");
var MatrixUtils$Cnn = require("../MatrixUtils.bs.js");
var ImmutableSparseMap$Cnn = require("../sparse_map/ImmutableSparseMap.bs.js");

function create(inputWidth, inputHeight, filterWidth, filterHeight, filterNumberOpt, zeroPaddingOpt, strideOpt, depthNumberOpt, initValueMethodOpt, randomFuncOpt, param) {
  var filterNumber = filterNumberOpt !== undefined ? filterNumberOpt : 1;
  var zeroPadding = zeroPaddingOpt !== undefined ? zeroPaddingOpt : 1;
  var stride = strideOpt !== undefined ? strideOpt : 1;
  var depthNumber = depthNumberOpt !== undefined ? depthNumberOpt : 1;
  var initValueMethod = initValueMethodOpt !== undefined ? initValueMethodOpt : Random$Cnn.random;
  var randomFunc = randomFuncOpt !== undefined ? randomFuncOpt : (function (prim) {
        return Math.random();
      });
  var outputWidth = LayerUtils$Cnn.computeOutputSize(inputWidth, filterWidth, zeroPadding, stride);
  var outputHeight = LayerUtils$Cnn.computeOutputSize(inputHeight, filterHeight, zeroPadding, stride);
  var fanIn = Math.imul(Math.imul(filterWidth, filterHeight), depthNumber);
  var fanOut = Math.imul(Math.imul(filterWidth, filterHeight), filterNumber);
  return {
          inputWidth: inputWidth,
          inputHeight: inputHeight,
          depthNumber: depthNumber,
          filterWidth: filterWidth,
          filterHeight: filterHeight,
          filterNumber: filterNumber,
          filterStates: ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, filterNumber - 1 | 0), (function (map, filterIndex) {
                  return ImmutableSparseMap$Cnn.set(map, filterIndex, Filter$Cnn.create([
                                  initValueMethod,
                                  randomFunc,
                                  fanIn,
                                  fanOut
                                ], filterWidth, filterHeight, depthNumber));
                }), ImmutableSparseMap$Cnn.createEmpty(undefined, undefined)),
          zeroPadding: zeroPadding,
          stride: stride,
          outputWidth: outputWidth,
          outputHeight: outputHeight
        };
}

function _padding(matrixMap, zeroPadding) {
  if (zeroPadding === 0) {
    return matrixMap;
  }
  var match = NP$Cnn.getMatrixMapSize(matrixMap);
  var paddingMatrixMap = NP$Cnn.zeroMatrixMap(match[2], match[1] + (zeroPadding << 1) | 0, match[0] + (zeroPadding << 1) | 0);
  return ImmutableSparseMap$Cnn.mapi(paddingMatrixMap, (function (paddingMatrix, i) {
                return NP$Cnn.fillMatrix(zeroPadding, zeroPadding, paddingMatrix, ImmutableSparseMap$Cnn.getExn(matrixMap, i));
              }));
}

function _crossCorrelation2D(data, filter, param, stride, bias) {
  var outputHeight = param[1];
  var outputWidth = param[0];
  var filterWidth = Matrix$Cnn.getColCount(filter);
  var filterHeight = Matrix$Cnn.getRowCount(filter);
  return NP$Cnn.reduceMatrix(NP$Cnn.zeroMatrix(outputHeight, outputWidth), (function (output, param, rowIndex, colIndex) {
                return MatrixUtils$Cnn.setValue(output, NP$Cnn.sum(NP$Cnn.dot(filter, LayerUtils$Cnn.getConvolutionRegion2D(data, rowIndex, colIndex, filterWidth, filterHeight, stride))) + bias, rowIndex, colIndex);
              }), Matrix$Cnn.create(outputHeight, outputWidth, []));
}

function _crossCorrelation3D(data, filters, param, stride, bias) {
  var outputHeight = param[1];
  var outputWidth = param[0];
  var match = NP$Cnn.getMatrixMapSize(filters);
  var filterHeight = match[1];
  var filterWidth = match[0];
  return NP$Cnn.reduceMatrix(NP$Cnn.zeroMatrix(outputHeight, outputWidth), (function (output, param, rowIndex, colIndex) {
                return MatrixUtils$Cnn.setValue(output, NP$Cnn.sumMatrixMap(ImmutableSparseMap$Cnn.mapi(LayerUtils$Cnn.getConvolutionRegion3D(data, rowIndex, colIndex, filterWidth, filterHeight, stride), (function (convolutionRegion, i) {
                                      return NP$Cnn.dot(ImmutableSparseMap$Cnn.getExn(filters, i), convolutionRegion);
                                    }))) + bias, rowIndex, colIndex);
              }), Matrix$Cnn.create(outputHeight, outputWidth, []));
}

var _elementWiseOp = Matrix$Cnn.map;

function forward(state, activatorData, inputs, param) {
  var match = OptionSt$Cnn.getExn(activatorData);
  var forwardMatrix = match.forwardMatrix;
  var paddedInputs = _padding(inputs, state.zeroPadding);
  var match$1 = ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, state.filterNumber - 1 | 0), (function (param, i) {
          var filterState = ImmutableSparseMap$Cnn.getExn(state.filterStates, i);
          var net = _crossCorrelation3D(paddedInputs, Filter$Cnn.getWeights(filterState), [
                state.outputWidth,
                state.outputHeight
              ], state.stride, Filter$Cnn.getBias(filterState));
          var output = Curry._1(forwardMatrix, net);
          return [
                  ImmutableSparseMap$Cnn.set(param[0], i, net),
                  ImmutableSparseMap$Cnn.set(param[1], i, output)
                ];
        }), [
        ImmutableSparseMap$Cnn.createEmpty(undefined, undefined),
        ImmutableSparseMap$Cnn.createEmpty(undefined, undefined)
      ]);
  return [
          state,
          match$1[0],
          match$1[1]
        ];
}

function _expandDeltaMapByStride(deltaMap, param) {
  var stride = param.stride;
  var zeroPadding = param.zeroPadding;
  var match = NP$Cnn.getMatrixMapSize(deltaMap);
  var expandDeltaWidth = LayerUtils$Cnn.computeOutputSize(param.inputWidth, param.filterWidth, zeroPadding, 1);
  var expandDeltaHeight = LayerUtils$Cnn.computeOutputSize(param.inputHeight, param.filterHeight, zeroPadding, 1);
  var expandDeltaMap = NP$Cnn.zeroMatrixMap(match[2], expandDeltaHeight, expandDeltaWidth);
  return ImmutableSparseMap$Cnn.mapi(deltaMap, (function (delta, i) {
                var expandDelta = ImmutableSparseMap$Cnn.getExn(expandDeltaMap, i);
                return NP$Cnn.reduceMatrix(delta, (function (expandDelta, value, rowIndex, colIndex) {
                              return MatrixUtils$Cnn.setValue(expandDelta, value, Math.imul(rowIndex, stride), Math.imul(colIndex, stride));
                            }), expandDelta);
              }));
}

function _paddingDeltaMap(expandDeltaMap, param) {
  var match = NP$Cnn.getMatrixMapSize(expandDeltaMap);
  return _padding(expandDeltaMap, (((param.inputWidth + param.filterWidth | 0) - 1 | 0) - match[0] | 0) / 2 | 0);
}

function _compute(param, padExpandDeltaMap, state, previousLayerNets) {
  var backward = param.backward;
  var previousLayerDeltaMap = ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, state.filterNumber - 1 | 0), (function (previousLayerDeltaMap, filterIndex) {
          var padExpandDelta = ImmutableSparseMap$Cnn.getExn(padExpandDeltaMap, filterIndex);
          var filterState = ImmutableSparseMap$Cnn.getExn(state.filterStates, filterIndex);
          var flippedWeights = ImmutableSparseMap$Cnn.map(Filter$Cnn.getWeights(filterState), NP$Cnn.rotate180);
          return NP$Cnn.addMatrixMap(previousLayerDeltaMap, ImmutableSparseMap$Cnn.mapi(LayerUtils$Cnn.createPreviousLayerDeltaMap([
                              state.depthNumber,
                              state.inputWidth,
                              state.inputHeight
                            ]), (function (delta, depthIndex) {
                            return _crossCorrelation2D(padExpandDelta, ImmutableSparseMap$Cnn.getExn(flippedWeights, depthIndex), [
                                        Matrix$Cnn.getColCount(delta),
                                        Matrix$Cnn.getRowCount(delta)
                                      ], 1, 0);
                          })));
        }), LayerUtils$Cnn.createPreviousLayerDeltaMap([
            state.depthNumber,
            state.inputWidth,
            state.inputHeight
          ]));
  return ImmutableSparseMap$Cnn.mapi(previousLayerDeltaMap, (function (previousLayerDelta, depthIndex) {
                return NP$Cnn.dot(previousLayerDelta, ImmutableSparseMap$Cnn.getExn(NP$Cnn.mapMatrixMap(previousLayerNets, (function (__x) {
                                      return Matrix$Cnn.map(__x, backward);
                                    })), depthIndex));
              }));
}

function bpDelta(previousLayerActivatorData, param, layerExpandDelta, state) {
  var __x = _paddingDeltaMap(layerExpandDelta, state);
  return _compute(OptionSt$Cnn.getExn(previousLayerActivatorData), __x, state, OptionSt$Cnn.getExn(param[1]));
}

function computeGradient(inputs, expandDeltaMap, state) {
  var state$1 = OptionSt$Cnn.getExn(state);
  var paddedInputs = _padding(inputs, state$1.zeroPadding);
  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, state$1.filterNumber - 1 | 0), (function (param, filterIndex) {
                ImmutableSparseMap$Cnn.getExn(state$1.filterStates, filterIndex);
                var expandDelta = ImmutableSparseMap$Cnn.getExn(expandDeltaMap, filterIndex);
                var weightGradients = ImmutableSparseMap$Cnn.mapi(NP$Cnn.zeroMatrixMap(state$1.depthNumber, state$1.filterHeight, state$1.filterWidth), (function (weightGradient, depthIndex) {
                        return _crossCorrelation2D(ImmutableSparseMap$Cnn.getExn(paddedInputs, depthIndex), expandDelta, [
                                    Matrix$Cnn.getColCount(weightGradient),
                                    Matrix$Cnn.getRowCount(weightGradient)
                                  ], 1, 0);
                      }));
                var biasGradient = NP$Cnn.sum(expandDelta);
                return [
                        ImmutableSparseMap$Cnn.set(param[0], filterIndex, weightGradients),
                        ImmutableSparseMap$Cnn.set(param[1], filterIndex, biasGradient)
                      ];
              }), [
              ImmutableSparseMap$Cnn.createEmpty(undefined, undefined),
              ImmutableSparseMap$Cnn.createEmpty(undefined, undefined)
            ]);
}

function createGradientDataSum(state) {
  var filterNumber = state.filterNumber;
  var filterHeight = state.filterHeight;
  var filterWidth = state.filterWidth;
  var depthNumber = state.depthNumber;
  return [
          ImmutableSparseMap$Cnn.createFromArr(ArraySt$Cnn.map(ArraySt$Cnn.range(0, filterNumber - 1 | 0), (function (param) {
                      return NP$Cnn.zeroMatrixMap(depthNumber, filterHeight, filterWidth);
                    }))),
          ImmutableSparseMap$Cnn.createFromArr(ArraySt$Cnn.map(ArraySt$Cnn.range(0, filterNumber - 1 | 0), (function (param) {
                      return 0;
                    })))
        ];
}

function getPreviousLayerNet(param, previousLayerOutput) {
  var invert = param.invert;
  return NP$Cnn.mapMatrixMap(previousLayerOutput, (function (__x) {
                return Matrix$Cnn.map(__x, invert);
              }));
}

function backward(previousLayerActivatorData, param, layerDelta, state) {
  var layerExpandDelta = _expandDeltaMapByStride(layerDelta, state);
  var previousLayerDeltaMap = bpDelta(previousLayerActivatorData, [
        undefined,
        param[1]
      ], layerExpandDelta, state);
  var gradientData = computeGradient(OptionSt$Cnn.getExn(param[0]), layerExpandDelta, Caml_option.some(state));
  return [
          previousLayerDeltaMap,
          gradientData
        ];
}

function addToGradientDataSum(gradientDataSum, gradientData) {
  var match = OptionSt$Cnn.getExn(gradientDataSum);
  var match$1 = OptionSt$Cnn.getExn(gradientData);
  var biasGradientData = match$1[1];
  var weightGradientData = match$1[0];
  return [
          ImmutableSparseMap$Cnn.mapi(match[0], (function (oneFilterWeightGradientsSum, filterIndex) {
                  return NP$Cnn.addMatrixMap(oneFilterWeightGradientsSum, ImmutableSparseMap$Cnn.getExn(weightGradientData, filterIndex));
                })),
          ImmutableSparseMap$Cnn.mapi(match[1], (function (oneFilterBiasGradientSum, filterIndex) {
                  return oneFilterBiasGradientSum + ImmutableSparseMap$Cnn.getExn(biasGradientData, filterIndex);
                }))
        ];
}

function update(state, optimizerData, param) {
  var miniBatchSize = param[0];
  var match = OptionSt$Cnn.getExn(param[1]);
  var biasGradientDataSum = match[1];
  var weightGradientDataSum = match[0];
  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, state.filterNumber - 1 | 0), (function (state, filterIndex) {
                var filterState = ImmutableSparseMap$Cnn.getExn(state.filterStates, filterIndex);
                return {
                        inputWidth: state.inputWidth,
                        inputHeight: state.inputHeight,
                        depthNumber: state.depthNumber,
                        filterWidth: state.filterWidth,
                        filterHeight: state.filterHeight,
                        filterNumber: state.filterNumber,
                        filterStates: ImmutableSparseMap$Cnn.set(state.filterStates, filterIndex, Filter$Cnn.update(filterState, [
                                  miniBatchSize,
                                  ImmutableSparseMap$Cnn.getExn(weightGradientDataSum, filterIndex),
                                  ImmutableSparseMap$Cnn.getExn(biasGradientDataSum, filterIndex)
                                ], optimizerData)),
                        zeroPadding: state.zeroPadding,
                        stride: state.stride,
                        outputWidth: state.outputWidth,
                        outputHeight: state.outputHeight
                      };
              }), state);
}

function createLayerData(state, activatorData) {
  return {
          layerName: "conv",
          state: state,
          forward: forward,
          backward: backward,
          update: update,
          createGradientDataSum: createGradientDataSum,
          addToGradientDataSum: addToGradientDataSum,
          activatorData: activatorData,
          getWeight: (function (state) {
              return Caml_option.some(ImmutableSparseMap$Cnn.map(state.filterStates, Filter$Cnn.getWeights));
            }),
          getBias: (function (state) {
              return Caml_option.some(ImmutableSparseMap$Cnn.map(state.filterStates, Filter$Cnn.getBias));
            })
        };
}

exports.create = create;
exports._padding = _padding;
exports._crossCorrelation2D = _crossCorrelation2D;
exports._crossCorrelation3D = _crossCorrelation3D;
exports._elementWiseOp = _elementWiseOp;
exports.forward = forward;
exports._expandDeltaMapByStride = _expandDeltaMapByStride;
exports._paddingDeltaMap = _paddingDeltaMap;
exports._compute = _compute;
exports.bpDelta = bpDelta;
exports.computeGradient = computeGradient;
exports.createGradientDataSum = createGradientDataSum;
exports.getPreviousLayerNet = getPreviousLayerNet;
exports.backward = backward;
exports.addToGradientDataSum = addToGradientDataSum;
exports.update = update;
exports.createLayerData = createLayerData;
/* No side effect */
