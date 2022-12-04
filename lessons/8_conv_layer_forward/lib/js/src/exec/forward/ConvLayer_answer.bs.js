'use strict';

var NP$8_cnn = require("../NP.bs.js");
var Log$8_cnn = require("../Log.bs.js");
var Matrix$8_cnn = require("../Matrix.bs.js");
var ArraySt$8_cnn = require("../ArraySt.bs.js");
var LayerUtils$8_cnn = require("../LayerUtils.bs.js");
var MatrixUtils$8_cnn = require("../MatrixUtils.bs.js");
var Filter_answer$8_cnn = require("./Filter_answer.bs.js");
var ImmutableSparseMap$8_cnn = require("../sparse_map/ImmutableSparseMap.bs.js");
var ReluActivator_answer$8_cnn = require("../relu/ReluActivator_answer.bs.js");

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
            bias: 1
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
        bias: 0
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
  var match = forward(ReluActivator_answer$8_cnn.forward, param[1], param[0]);
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

exports.create = create;
exports._padding = _padding;
exports._crossCorrelation3D = _crossCorrelation3D;
exports._elementWiseOp = _elementWiseOp;
exports.forward = forward;
exports.Test = Test;
/*  Not a pure module */
