

import * as NP$8_cnn from "./NP.bs.js";
import * as Log$8_cnn from "./Log.bs.js";
import * as Filter$8_cnn from "./Filter.bs.js";
import * as Matrix$8_cnn from "./Matrix.bs.js";
import * as ArraySt$8_cnn from "./ArraySt.bs.js";
import * as FloatUtils$8_cnn from "./FloatUtils.bs.js";
import * as LayerUtils$8_cnn from "./LayerUtils.bs.js";
import * as MatrixUtils$8_cnn from "./MatrixUtils.bs.js";
import * as ReluActivator$8_cnn from "./ReluActivator.bs.js";
import * as ImmutableSparseMap$8_cnn from "./sparse_map/ImmutableSparseMap.bs.js";

function create(inputWidth, inputHeight, depthNumber, filterWidth, filterHeight, filterNumber, zeroPadding, stride, leraningRate) {
  var outputWidth = LayerUtils$8_cnn.computeOutputSize(inputWidth, filterWidth, zeroPadding, stride);
  var outputHeight = LayerUtils$8_cnn.computeOutputSize(inputHeight, filterHeight, zeroPadding, stride);
  return {
          inputWidth: inputWidth,
          inputHeight: inputHeight,
          depthNumber: depthNumber,
          filterWidth: filterWidth,
          filterHeight: filterHeight,
          filterNumber: filterNumber,
          filterStates: ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, filterNumber - 1 | 0), (function (map, filterIndex) {
                  return ImmutableSparseMap$8_cnn.set(map, filterIndex, Filter$8_cnn.create(filterWidth, filterHeight, depthNumber));
                }), ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined)),
          zeroPadding: zeroPadding,
          stride: stride,
          outputWidth: outputWidth,
          outputHeight: outputHeight,
          leraningRate: leraningRate
        };
}

function _padding(matrixMap, zeroPadding) {
  if (zeroPadding === 0) {
    return matrixMap;
  }
  var match = NP$8_cnn.getMatrixMapSize(matrixMap);
  var paddingMatrixMap = NP$8_cnn.zeroMatrixMap(match[2], match[1] + (zeroPadding << 1) | 0, match[0] + (zeroPadding << 1) | 0);
  return ImmutableSparseMap$8_cnn.mapi(paddingMatrixMap, (function (paddingMatrix, i) {
                return NP$8_cnn.fillMatrix(zeroPadding, zeroPadding, paddingMatrix, ImmutableSparseMap$8_cnn.getExn(matrixMap, i));
              }));
}

function _crossCorrelation2D(input, weight, param, stride, bias) {
  var outputHeight = param[1];
  var outputWidth = param[0];
  var filterWidth = Matrix$8_cnn.getColCount(weight);
  var filterHeight = Matrix$8_cnn.getRowCount(weight);
  return NP$8_cnn.reduceMatrix(NP$8_cnn.zeroMatrix(outputHeight, outputWidth), (function (output, param, rowIndex, colIndex) {
                return MatrixUtils$8_cnn.setValue(output, NP$8_cnn.sum(NP$8_cnn.dot(weight, LayerUtils$8_cnn.getConvolutionRegion2D(input, rowIndex, colIndex, filterWidth, filterHeight, stride))) + bias, rowIndex, colIndex);
              }), Matrix$8_cnn.create(outputHeight, outputWidth, []));
}

function _crossCorrelation3D(inputs, weights, param, stride, bias) {
  var outputHeight = param[1];
  var outputWidth = param[0];
  var match = NP$8_cnn.getMatrixMapSize(weights);
  var filterHeight = match[1];
  var filterWidth = match[0];
  return NP$8_cnn.reduceMatrix(NP$8_cnn.zeroMatrix(outputHeight, outputWidth), (function (output, param, rowIndex, colIndex) {
                return MatrixUtils$8_cnn.setValue(output, NP$8_cnn.sumMatrixMap(ImmutableSparseMap$8_cnn.mapi(LayerUtils$8_cnn.getConvolutionRegion3D(inputs, rowIndex, colIndex, filterWidth, filterHeight, stride), (function (convolutionRegion, i) {
                                      return NP$8_cnn.dot(ImmutableSparseMap$8_cnn.getExn(weights, i), convolutionRegion);
                                    }))) + bias, rowIndex, colIndex);
              }), Matrix$8_cnn.create(outputHeight, outputWidth, []));
}

var _elementWiseOp = Matrix$8_cnn.map;

function forward(activate, state, inputs) {
  var paddedInputs = _padding(inputs, state.zeroPadding);
  var match = ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, state.filterNumber - 1 | 0), (function (param, i) {
          var filterState = ImmutableSparseMap$8_cnn.getExn(state.filterStates, i);
          var net = _crossCorrelation3D(paddedInputs, Filter$8_cnn.getWeights(filterState), [
                state.outputWidth,
                state.outputHeight
              ], state.stride, Filter$8_cnn.getBias(filterState));
          var output = Matrix$8_cnn.map(net, activate);
          return [
                  ImmutableSparseMap$8_cnn.set(param[0], i, net),
                  ImmutableSparseMap$8_cnn.set(param[1], i, output)
                ];
        }), [
        ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined),
        ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined)
      ]);
  return [
          paddedInputs,
          [
            match[0],
            match[1]
          ]
        ];
}

function _expandDeltaMapByStride(deltaMap, param) {
  var stride = param.stride;
  var zeroPadding = param.zeroPadding;
  var match = NP$8_cnn.getMatrixMapSize(deltaMap);
  var expandDeltaWidth = LayerUtils$8_cnn.computeOutputSize(param.inputWidth, param.filterWidth, zeroPadding, 1);
  var expandDeltaHeight = LayerUtils$8_cnn.computeOutputSize(param.inputHeight, param.filterHeight, zeroPadding, 1);
  var expandDeltaMap = NP$8_cnn.zeroMatrixMap(match[2], expandDeltaHeight, expandDeltaWidth);
  return ImmutableSparseMap$8_cnn.mapi(deltaMap, (function (delta, i) {
                var expandDelta = ImmutableSparseMap$8_cnn.getExn(expandDeltaMap, i);
                return NP$8_cnn.reduceMatrix(delta, (function (expandDelta, value, rowIndex, colIndex) {
                              return MatrixUtils$8_cnn.setValue(expandDelta, value, Math.imul(rowIndex, stride), Math.imul(colIndex, stride));
                            }), expandDelta);
              }));
}

function _paddingDeltaMap(expandDeltaMap, param) {
  var match = NP$8_cnn.getMatrixMapSize(expandDeltaMap);
  return _padding(expandDeltaMap, (((param.inputWidth + param.filterWidth | 0) - 1 | 0) - match[0] | 0) / 2 | 0);
}

function _compute(padExpandDeltaMap, state, inputs) {
  var lastLayerNets = NP$8_cnn.mapMatrixMap(inputs, (function (__x) {
          return Matrix$8_cnn.map(__x, ReluActivator$8_cnn.invert);
        }));
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, state.filterNumber - 1 | 0), (function (lastLayerDeltaMap, filterIndex) {
                var padExpandDelta = ImmutableSparseMap$8_cnn.getExn(padExpandDeltaMap, filterIndex);
                var filterState = ImmutableSparseMap$8_cnn.getExn(state.filterStates, filterIndex);
                var flippedWeights = ImmutableSparseMap$8_cnn.map(Filter$8_cnn.getWeights(filterState), NP$8_cnn.rotate180);
                var lastLayerDeltaMap$1 = NP$8_cnn.addMatrixMap(lastLayerDeltaMap, ImmutableSparseMap$8_cnn.mapi(LayerUtils$8_cnn.createLastLayerDeltaMap([
                              state.depthNumber,
                              state.inputWidth,
                              state.inputHeight
                            ]), (function (delta, depthIndex) {
                            return _crossCorrelation2D(padExpandDelta, ImmutableSparseMap$8_cnn.getExn(flippedWeights, depthIndex), [
                                        Matrix$8_cnn.getColCount(delta),
                                        Matrix$8_cnn.getRowCount(delta)
                                      ], 1, 0);
                          })));
                return ImmutableSparseMap$8_cnn.mapi(lastLayerDeltaMap$1, (function (lastLayerDelta, depthIndex) {
                              return NP$8_cnn.dot(lastLayerDelta, ImmutableSparseMap$8_cnn.getExn(NP$8_cnn.mapMatrixMap(lastLayerNets, (function (__x) {
                                                    return Matrix$8_cnn.map(__x, ReluActivator$8_cnn.backward);
                                                  })), depthIndex));
                            }));
              }), LayerUtils$8_cnn.createLastLayerDeltaMap([
                  state.depthNumber,
                  state.inputWidth,
                  state.inputHeight
                ]));
}

function bpDeltaMap(state, inputs, deltaMap) {
  return _compute(_paddingDeltaMap(_expandDeltaMapByStride(deltaMap, state), state), state, inputs);
}

function bpGradient(state, paddedInputs, deltaMap) {
  var expandDeltaMap = _expandDeltaMapByStride(deltaMap, state);
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, state.filterNumber - 1 | 0), (function (state, filterIndex) {
                var filterState = ImmutableSparseMap$8_cnn.getExn(state.filterStates, filterIndex);
                var expandDelta = ImmutableSparseMap$8_cnn.getExn(expandDeltaMap, filterIndex);
                var weightGradients = ImmutableSparseMap$8_cnn.mapi(filterState.weightGradients, (function (weightGradient, depthIndex) {
                        return _crossCorrelation2D(ImmutableSparseMap$8_cnn.getExn(paddedInputs, depthIndex), expandDelta, [
                                    Matrix$8_cnn.getColCount(weightGradient),
                                    Matrix$8_cnn.getRowCount(weightGradient)
                                  ], 1, 0);
                      }));
                var biasGradient = NP$8_cnn.sum(expandDelta);
                return {
                        inputWidth: state.inputWidth,
                        inputHeight: state.inputHeight,
                        depthNumber: state.depthNumber,
                        filterWidth: state.filterWidth,
                        filterHeight: state.filterHeight,
                        filterNumber: state.filterNumber,
                        filterStates: ImmutableSparseMap$8_cnn.set(state.filterStates, filterIndex, {
                              weights: filterState.weights,
                              bias: filterState.bias,
                              weightGradients: weightGradients,
                              biasGradient: biasGradient
                            }),
                        zeroPadding: state.zeroPadding,
                        stride: state.stride,
                        outputWidth: state.outputWidth,
                        outputHeight: state.outputHeight,
                        leraningRate: state.leraningRate
                      };
              }), state);
}

function backward(state, inputs, deltaMap) {
  var match = forward(ReluActivator$8_cnn.forward, state, inputs);
  bpDeltaMap(state, inputs, deltaMap);
  return bpGradient(state, match[0], deltaMap);
}

function update(state) {
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, state.filterNumber - 1 | 0), (function (state, i) {
                var filterState = ImmutableSparseMap$8_cnn.getExn(state.filterStates, i);
                return {
                        inputWidth: state.inputWidth,
                        inputHeight: state.inputHeight,
                        depthNumber: state.depthNumber,
                        filterWidth: state.filterWidth,
                        filterHeight: state.filterHeight,
                        filterNumber: state.filterNumber,
                        filterStates: ImmutableSparseMap$8_cnn.set(state.filterStates, i, Filter$8_cnn.update(filterState, state.leraningRate)),
                        zeroPadding: state.zeroPadding,
                        stride: state.stride,
                        outputWidth: state.outputWidth,
                        outputHeight: state.outputHeight,
                        leraningRate: state.leraningRate
                      };
              }), state);
}

function init(param) {
  var inputs = NP$8_cnn.createMatrixMapByDataArr([[
          [
            0,
            1,
            1,
            0,
            2
          ],
          [
            2,
            2,
            2,
            2,
            1
          ],
          [
            1,
            0,
            0,
            2,
            0
          ],
          [
            0,
            1,
            1,
            0,
            0
          ],
          [
            1,
            2,
            0,
            0,
            2
          ]
        ]]);
  var deltaMap = NP$8_cnn.createMatrixMapByDataArr([[
          [
            0,
            1,
            1
          ],
          [
            2,
            2,
            2
          ],
          [
            1,
            0,
            0
          ]
        ]]);
  var state = create(5, 5, 1, 3, 3, 1, 1, 2, 0.001);
  var init$1 = ImmutableSparseMap$8_cnn.getExn(state.filterStates, 0);
  var state_inputWidth = state.inputWidth;
  var state_inputHeight = state.inputHeight;
  var state_depthNumber = state.depthNumber;
  var state_filterWidth = state.filterWidth;
  var state_filterHeight = state.filterHeight;
  var state_filterNumber = state.filterNumber;
  var state_filterStates = ImmutableSparseMap$8_cnn.set(ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined), 0, {
        weights: NP$8_cnn.createMatrixMapByDataArr([
              [
                [
                  -1,
                  1,
                  0
                ],
                [
                  0,
                  1,
                  0
                ],
                [
                  0,
                  1,
                  1
                ]
              ],
              [
                [
                  -1,
                  -1,
                  0
                ],
                [
                  0,
                  0,
                  0
                ],
                [
                  0,
                  -1,
                  0
                ]
              ],
              [
                [
                  0,
                  0,
                  -1
                ],
                [
                  0,
                  1,
                  0
                ],
                [
                  1,
                  -1,
                  -1
                ]
              ]
            ]),
        bias: 1,
        weightGradients: init$1.weightGradients,
        biasGradient: init$1.biasGradient
      });
  var state_zeroPadding = state.zeroPadding;
  var state_stride = state.stride;
  var state_outputWidth = state.outputWidth;
  var state_outputHeight = state.outputHeight;
  var state_leraningRate = state.leraningRate;
  var state$1 = {
    inputWidth: state_inputWidth,
    inputHeight: state_inputHeight,
    depthNumber: state_depthNumber,
    filterWidth: state_filterWidth,
    filterHeight: state_filterHeight,
    filterNumber: state_filterNumber,
    filterStates: state_filterStates,
    zeroPadding: state_zeroPadding,
    stride: state_stride,
    outputWidth: state_outputWidth,
    outputHeight: state_outputHeight,
    leraningRate: state_leraningRate
  };
  return [
          inputs,
          deltaMap,
          state$1
        ];
}

function test(param) {
  var state = param[2];
  var inputs = param[0];
  var match = forward(ReluActivator$8_cnn.forward, state, inputs);
  Log$8_cnn.printForDebug([
        "f:",
        match[1][0]
      ]);
  var state$1 = backward(state, inputs, param[1]);
  var state$2 = update(state$1);
  Log$8_cnn.printForDebug(state$2.filterStates);
  
}

function checkGradient(param) {
  var match = init(undefined);
  var state = match[2];
  var inputs = match[0];
  var match$1 = forward(ReluActivator$8_cnn.forward, state, inputs);
  var match$2 = match$1[1];
  var outputMap = match$2[1];
  console.log(outputMap);
  console.log(inputs);
  NP$8_cnn.getMatrixMapSize(match[1]);
  var nextLayerDeltaMap = NP$8_cnn.mapMatrixMap(match$2[0], (function (matrix) {
          return Matrix$8_cnn.map(matrix, (function (value) {
                        if (value > 0) {
                          return 1;
                        } else {
                          return 0;
                        }
                      }));
        }));
  var layerDeltaMap = bpDeltaMap(state, outputMap, nextLayerDeltaMap);
  var state$1 = bpGradient(state, match$1[0], layerDeltaMap);
  var filterState = ImmutableSparseMap$8_cnn.getExn(state$1.filterStates, 0);
  var weights = filterState.weights;
  return ImmutableSparseMap$8_cnn.forEachi(filterState.weightGradients, (function (weightGradient, depthIndex) {
                return NP$8_cnn.forEachMatrix(weightGradient, (function (actualGradient, rowIndex, colIndex) {
                              var weight = ImmutableSparseMap$8_cnn.getExn(weights, depthIndex);
                              var weightValue = MatrixUtils$8_cnn.getValue(rowIndex, colIndex, weight);
                              var state1_inputWidth = state$1.inputWidth;
                              var state1_inputHeight = state$1.inputHeight;
                              var state1_depthNumber = state$1.depthNumber;
                              var state1_filterWidth = state$1.filterWidth;
                              var state1_filterHeight = state$1.filterHeight;
                              var state1_filterNumber = state$1.filterNumber;
                              var state1_filterStates = ImmutableSparseMap$8_cnn.set(state$1.filterStates, 0, {
                                    weights: ImmutableSparseMap$8_cnn.set(weights, depthIndex, MatrixUtils$8_cnn.setValue(NP$8_cnn.copyMatrix(weight), weightValue + 10e-4, rowIndex, colIndex)),
                                    bias: filterState.bias,
                                    weightGradients: filterState.weightGradients,
                                    biasGradient: filterState.biasGradient
                                  });
                              var state1_zeroPadding = state$1.zeroPadding;
                              var state1_stride = state$1.stride;
                              var state1_outputWidth = state$1.outputWidth;
                              var state1_outputHeight = state$1.outputHeight;
                              var state1_leraningRate = state$1.leraningRate;
                              var state1 = {
                                inputWidth: state1_inputWidth,
                                inputHeight: state1_inputHeight,
                                depthNumber: state1_depthNumber,
                                filterWidth: state1_filterWidth,
                                filterHeight: state1_filterHeight,
                                filterNumber: state1_filterNumber,
                                filterStates: state1_filterStates,
                                zeroPadding: state1_zeroPadding,
                                stride: state1_stride,
                                outputWidth: state1_outputWidth,
                                outputHeight: state1_outputHeight,
                                leraningRate: state1_leraningRate
                              };
                              var _activate_linear = function (net) {
                                return net;
                              };
                              var match = forward(ReluActivator$8_cnn.forward, state1, inputs);
                              var match$1 = forward(_activate_linear, state$1, match[1][1]);
                              var err1 = NP$8_cnn.sumMatrixMap(match$1[1][1]);
                              var state2_inputWidth = state$1.inputWidth;
                              var state2_inputHeight = state$1.inputHeight;
                              var state2_depthNumber = state$1.depthNumber;
                              var state2_filterWidth = state$1.filterWidth;
                              var state2_filterHeight = state$1.filterHeight;
                              var state2_filterNumber = state$1.filterNumber;
                              var state2_filterStates = ImmutableSparseMap$8_cnn.set(state$1.filterStates, 0, {
                                    weights: ImmutableSparseMap$8_cnn.set(weights, depthIndex, MatrixUtils$8_cnn.setValue(NP$8_cnn.copyMatrix(weight), weightValue - 10e-4, rowIndex, colIndex)),
                                    bias: filterState.bias,
                                    weightGradients: filterState.weightGradients,
                                    biasGradient: filterState.biasGradient
                                  });
                              var state2_zeroPadding = state$1.zeroPadding;
                              var state2_stride = state$1.stride;
                              var state2_outputWidth = state$1.outputWidth;
                              var state2_outputHeight = state$1.outputHeight;
                              var state2_leraningRate = state$1.leraningRate;
                              var state2 = {
                                inputWidth: state2_inputWidth,
                                inputHeight: state2_inputHeight,
                                depthNumber: state2_depthNumber,
                                filterWidth: state2_filterWidth,
                                filterHeight: state2_filterHeight,
                                filterNumber: state2_filterNumber,
                                filterStates: state2_filterStates,
                                zeroPadding: state2_zeroPadding,
                                stride: state2_stride,
                                outputWidth: state2_outputWidth,
                                outputHeight: state2_outputHeight,
                                leraningRate: state2_leraningRate
                              };
                              var match$2 = forward(ReluActivator$8_cnn.forward, state2, inputs);
                              var match$3 = forward(_activate_linear, state$1, match$2[1][1]);
                              var err2 = NP$8_cnn.sumMatrixMap(match$3[1][1]);
                              var expectedGradient = (err1 - err2) / (2 * 10e-4);
                              var result = FloatUtils$8_cnn.truncateFloatValue(expectedGradient, 4) === FloatUtils$8_cnn.truncateFloatValue(actualGradient, 4);
                              console.log("check gradient -> weights(" + depthIndex + "), " + rowIndex + ", " + colIndex + "): " + result);
                              
                            }));
              }));
}

var Test = {
  init: init,
  test: test,
  checkGradient: checkGradient
};

checkGradient(undefined);

export {
  create ,
  _padding ,
  _crossCorrelation2D ,
  _crossCorrelation3D ,
  _elementWiseOp ,
  forward ,
  _expandDeltaMapByStride ,
  _paddingDeltaMap ,
  _compute ,
  bpDeltaMap ,
  bpGradient ,
  backward ,
  update ,
  Test ,
  
}
/*  Not a pure module */
