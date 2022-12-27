

import * as NP$8_cnn from "../NP.bs.js";
import * as Log$8_cnn from "../Log.bs.js";
import * as Matrix$8_cnn from "../Matrix.bs.js";
import * as ArraySt$8_cnn from "../ArraySt.bs.js";
import * as LayerUtils$8_cnn from "../LayerUtils.bs.js";
import * as MatrixUtils$8_cnn from "../MatrixUtils.bs.js";
import * as Filter_answer$8_cnn from "./Filter_answer.bs.js";
import * as ReluActivator$8_cnn from "../ReluActivator.bs.js";
import * as ImmutableSparseMap$8_cnn from "../sparse_map/ImmutableSparseMap.bs.js";

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
                  return ImmutableSparseMap$8_cnn.set(map, filterIndex, Filter_answer$8_cnn.create(filterWidth, filterHeight, depthNumber));
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
          var net = _crossCorrelation3D(paddedInputs, Filter_answer$8_cnn.getWeights(filterState), [
                state.outputWidth,
                state.outputHeight
              ], state.stride, Filter_answer$8_cnn.getBias(filterState));
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

function _expandDeltaMapByStride(nextLayerDeltaMap, param) {
  var stride = param.stride;
  var zeroPadding = param.zeroPadding;
  var match = NP$8_cnn.getMatrixMapSize(nextLayerDeltaMap);
  var expandDeltaWidth = LayerUtils$8_cnn.computeOutputSize(param.inputWidth, param.filterWidth, zeroPadding, 1);
  var expandDeltaHeight = LayerUtils$8_cnn.computeOutputSize(param.inputHeight, param.filterHeight, zeroPadding, 1);
  var expandDeltaMap = NP$8_cnn.zeroMatrixMap(match[2], expandDeltaHeight, expandDeltaWidth);
  return ImmutableSparseMap$8_cnn.mapi(nextLayerDeltaMap, (function (delta, i) {
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

function _compute(padExpandDeltaMap, state, inputNets) {
  var currentLayerDeltaMap = ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, state.filterNumber - 1 | 0), (function (currentLayerDeltaMap, filterIndex) {
          var padExpandDelta = ImmutableSparseMap$8_cnn.getExn(padExpandDeltaMap, filterIndex);
          var filterState = ImmutableSparseMap$8_cnn.getExn(state.filterStates, filterIndex);
          var flippedWeights = ImmutableSparseMap$8_cnn.map(Filter_answer$8_cnn.getWeights(filterState), NP$8_cnn.rotate180);
          return NP$8_cnn.addMatrixMap(currentLayerDeltaMap, ImmutableSparseMap$8_cnn.mapi(LayerUtils$8_cnn.createCurrentLayerDeltaMap([
                              state.depthNumber,
                              state.inputWidth,
                              state.inputHeight
                            ]), (function (delta, depthIndex) {
                            return _crossCorrelation2D(padExpandDelta, ImmutableSparseMap$8_cnn.getExn(flippedWeights, depthIndex), [
                                        Matrix$8_cnn.getColCount(delta),
                                        Matrix$8_cnn.getRowCount(delta)
                                      ], 1, 0);
                          })));
        }), LayerUtils$8_cnn.createCurrentLayerDeltaMap([
            state.depthNumber,
            state.inputWidth,
            state.inputHeight
          ]));
  return ImmutableSparseMap$8_cnn.mapi(currentLayerDeltaMap, (function (currentLayerDelta, depthIndex) {
                return NP$8_cnn.dot(currentLayerDelta, ImmutableSparseMap$8_cnn.getExn(NP$8_cnn.mapMatrixMap(inputNets, (function (__x) {
                                      return Matrix$8_cnn.map(__x, ReluActivator$8_cnn.backward);
                                    })), depthIndex));
              }));
}

function bpDeltaMap(state, inputNets, nextLayerDeltaMap) {
  return _compute(_paddingDeltaMap(_expandDeltaMapByStride(nextLayerDeltaMap, state), state), state, inputNets);
}

function computeGradient(state, inputs, currentLayerDeltaMap) {
  var paddedInputs = _padding(inputs, state.zeroPadding);
  var expandDeltaMap = _expandDeltaMapByStride(currentLayerDeltaMap, state);
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

function init(param) {
  var inputs = NP$8_cnn.createMatrixMapByDataArr([
        [
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
        ],
        [
          [
            1,
            0,
            2,
            2,
            0
          ],
          [
            0,
            0,
            0,
            2,
            0
          ],
          [
            1,
            2,
            1,
            2,
            1
          ],
          [
            1,
            0,
            0,
            0,
            0
          ],
          [
            1,
            2,
            1,
            1,
            1
          ]
        ],
        [
          [
            2,
            1,
            2,
            0,
            0
          ],
          [
            1,
            0,
            0,
            1,
            0
          ],
          [
            0,
            2,
            1,
            0,
            1
          ],
          [
            0,
            1,
            2,
            2,
            2
          ],
          [
            2,
            1,
            0,
            0,
            1
          ]
        ]
      ]);
  var state = create(5, 5, 3, 3, 3, 2, 1, 2, 0.001);
  var init$1 = ImmutableSparseMap$8_cnn.getExn(state.filterStates, 0);
  var init$2 = ImmutableSparseMap$8_cnn.getExn(state.filterStates, 1);
  var state_inputWidth = state.inputWidth;
  var state_inputHeight = state.inputHeight;
  var state_depthNumber = state.depthNumber;
  var state_filterWidth = state.filterWidth;
  var state_filterHeight = state.filterHeight;
  var state_filterNumber = state.filterNumber;
  var state_filterStates = ImmutableSparseMap$8_cnn.set(ImmutableSparseMap$8_cnn.set(ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined), 0, {
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
          }), 1, {
        weights: NP$8_cnn.createMatrixMapByDataArr([
              [
                [
                  1,
                  1,
                  -1
                ],
                [
                  -1,
                  -1,
                  1
                ],
                [
                  0,
                  -1,
                  1
                ]
              ],
              [
                [
                  0,
                  1,
                  0
                ],
                [
                  -1,
                  0,
                  -1
                ],
                [
                  -1,
                  1,
                  0
                ]
              ],
              [
                [
                  -1,
                  0,
                  0
                ],
                [
                  -1,
                  0,
                  1
                ],
                [
                  -1,
                  0,
                  0
                ]
              ]
            ]),
        bias: 0,
        weightGradients: init$2.weightGradients,
        biasGradient: init$2.biasGradient
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
          state$1
        ];
}

function test(param) {
  var match = forward(ReluActivator$8_cnn.forward, param[1], param[0]);
  Log$8_cnn.printForDebug([
        "f:",
        match[1][0]
      ]);
  
}

var Test = {
  init: init,
  test: test
};

test(init(undefined));

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
  computeGradient ,
  Test ,
  
}
/*  Not a pure module */
