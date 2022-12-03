'use strict';

var NP$8_cnn = require("./NP.bs.js");
var Log$8_cnn = require("./Log.bs.js");
var Matrix$8_cnn = require("./Matrix.bs.js");
var ConvLayer$8_cnn = require("./ConvLayer.bs.js");
var FloatUtils$8_cnn = require("./FloatUtils.bs.js");
var MatrixUtils$8_cnn = require("./MatrixUtils.bs.js");
var ReluActivator$8_cnn = require("./ReluActivator.bs.js");
var ImmutableSparseMap$8_cnn = require("./sparse_map/ImmutableSparseMap.bs.js");

function checkGradient(param) {
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
  var stateForConvLayer1 = ConvLayer$8_cnn.create(5, 5, 1, 3, 3, 1, 0, 1, 0.001);
  var init = ImmutableSparseMap$8_cnn.getExn(stateForConvLayer1.filterStates, 0);
  var stateForConvLayer1_inputWidth = stateForConvLayer1.inputWidth;
  var stateForConvLayer1_inputHeight = stateForConvLayer1.inputHeight;
  var stateForConvLayer1_depthNumber = stateForConvLayer1.depthNumber;
  var stateForConvLayer1_filterWidth = stateForConvLayer1.filterWidth;
  var stateForConvLayer1_filterHeight = stateForConvLayer1.filterHeight;
  var stateForConvLayer1_filterNumber = stateForConvLayer1.filterNumber;
  var stateForConvLayer1_filterStates = ImmutableSparseMap$8_cnn.set(ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined), 0, {
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
        weightGradients: init.weightGradients,
        biasGradient: init.biasGradient
      });
  var stateForConvLayer1_zeroPadding = stateForConvLayer1.zeroPadding;
  var stateForConvLayer1_stride = stateForConvLayer1.stride;
  var stateForConvLayer1_outputWidth = stateForConvLayer1.outputWidth;
  var stateForConvLayer1_outputHeight = stateForConvLayer1.outputHeight;
  var stateForConvLayer1_leraningRate = stateForConvLayer1.leraningRate;
  var stateForConvLayer1$1 = {
    inputWidth: stateForConvLayer1_inputWidth,
    inputHeight: stateForConvLayer1_inputHeight,
    depthNumber: stateForConvLayer1_depthNumber,
    filterWidth: stateForConvLayer1_filterWidth,
    filterHeight: stateForConvLayer1_filterHeight,
    filterNumber: stateForConvLayer1_filterNumber,
    filterStates: stateForConvLayer1_filterStates,
    zeroPadding: stateForConvLayer1_zeroPadding,
    stride: stateForConvLayer1_stride,
    outputWidth: stateForConvLayer1_outputWidth,
    outputHeight: stateForConvLayer1_outputHeight,
    leraningRate: stateForConvLayer1_leraningRate
  };
  var stateForConvLayer2 = ConvLayer$8_cnn.create(3, 3, 1, 2, 2, 1, 0, 1, 0.001);
  var init$1 = ImmutableSparseMap$8_cnn.getExn(stateForConvLayer2.filterStates, 0);
  var stateForConvLayer2_inputWidth = stateForConvLayer2.inputWidth;
  var stateForConvLayer2_inputHeight = stateForConvLayer2.inputHeight;
  var stateForConvLayer2_depthNumber = stateForConvLayer2.depthNumber;
  var stateForConvLayer2_filterWidth = stateForConvLayer2.filterWidth;
  var stateForConvLayer2_filterHeight = stateForConvLayer2.filterHeight;
  var stateForConvLayer2_filterNumber = stateForConvLayer2.filterNumber;
  var stateForConvLayer2_filterStates = ImmutableSparseMap$8_cnn.set(ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined), 0, {
        weights: NP$8_cnn.createMatrixMapByDataArr([
              [
                [
                  1,
                  0
                ],
                [
                  1,
                  0
                ]
              ],
              [
                [
                  -1,
                  0
                ],
                [
                  0,
                  0
                ]
              ]
            ]),
        bias: 1,
        weightGradients: init$1.weightGradients,
        biasGradient: init$1.biasGradient
      });
  var stateForConvLayer2_zeroPadding = stateForConvLayer2.zeroPadding;
  var stateForConvLayer2_stride = stateForConvLayer2.stride;
  var stateForConvLayer2_outputWidth = stateForConvLayer2.outputWidth;
  var stateForConvLayer2_outputHeight = stateForConvLayer2.outputHeight;
  var stateForConvLayer2_leraningRate = stateForConvLayer2.leraningRate;
  var stateForConvLayer2$1 = {
    inputWidth: stateForConvLayer2_inputWidth,
    inputHeight: stateForConvLayer2_inputHeight,
    depthNumber: stateForConvLayer2_depthNumber,
    filterWidth: stateForConvLayer2_filterWidth,
    filterHeight: stateForConvLayer2_filterHeight,
    filterNumber: stateForConvLayer2_filterNumber,
    filterStates: stateForConvLayer2_filterStates,
    zeroPadding: stateForConvLayer2_zeroPadding,
    stride: stateForConvLayer2_stride,
    outputWidth: stateForConvLayer2_outputWidth,
    outputHeight: stateForConvLayer2_outputHeight,
    leraningRate: stateForConvLayer2_leraningRate
  };
  var match = ConvLayer$8_cnn.forward(ReluActivator$8_cnn.forward, stateForConvLayer1$1, inputs);
  var match$1 = match[1];
  var convLayer2DeltaMap = NP$8_cnn.mapMatrixMap(match$1[0], (function (matrix) {
          return Matrix$8_cnn.map(matrix, (function (value) {
                        if (value > 0) {
                          return 1;
                        } else {
                          return 0;
                        }
                      }));
        }));
  var convLayer1DeltaMap = ConvLayer$8_cnn.bpDeltaMap(stateForConvLayer2$1, match$1[1], convLayer2DeltaMap);
  var stateForConvLayer1$2 = ConvLayer$8_cnn.bpGradient(stateForConvLayer1$1, match[0], convLayer1DeltaMap);
  var filterState = ImmutableSparseMap$8_cnn.getExn(stateForConvLayer1$2.filterStates, 0);
  var weights = filterState.weights;
  return ImmutableSparseMap$8_cnn.forEachi(filterState.weightGradients, (function (weightGradient, depthIndex) {
                return NP$8_cnn.forEachMatrix(weightGradient, (function (actualGradient, rowIndex, colIndex) {
                              var weight = ImmutableSparseMap$8_cnn.getExn(weights, depthIndex);
                              var weightValue = MatrixUtils$8_cnn.getValue(rowIndex, colIndex, weight);
                              var stateForConvLayer1_1_inputWidth = stateForConvLayer1$2.inputWidth;
                              var stateForConvLayer1_1_inputHeight = stateForConvLayer1$2.inputHeight;
                              var stateForConvLayer1_1_depthNumber = stateForConvLayer1$2.depthNumber;
                              var stateForConvLayer1_1_filterWidth = stateForConvLayer1$2.filterWidth;
                              var stateForConvLayer1_1_filterHeight = stateForConvLayer1$2.filterHeight;
                              var stateForConvLayer1_1_filterNumber = stateForConvLayer1$2.filterNumber;
                              var stateForConvLayer1_1_filterStates = ImmutableSparseMap$8_cnn.set(stateForConvLayer1$2.filterStates, 0, {
                                    weights: ImmutableSparseMap$8_cnn.set(weights, depthIndex, MatrixUtils$8_cnn.setValue(NP$8_cnn.copyMatrix(weight), weightValue + 10e-4, rowIndex, colIndex)),
                                    bias: filterState.bias,
                                    weightGradients: filterState.weightGradients,
                                    biasGradient: filterState.biasGradient
                                  });
                              var stateForConvLayer1_1_zeroPadding = stateForConvLayer1$2.zeroPadding;
                              var stateForConvLayer1_1_stride = stateForConvLayer1$2.stride;
                              var stateForConvLayer1_1_outputWidth = stateForConvLayer1$2.outputWidth;
                              var stateForConvLayer1_1_outputHeight = stateForConvLayer1$2.outputHeight;
                              var stateForConvLayer1_1_leraningRate = stateForConvLayer1$2.leraningRate;
                              var stateForConvLayer1_1 = {
                                inputWidth: stateForConvLayer1_1_inputWidth,
                                inputHeight: stateForConvLayer1_1_inputHeight,
                                depthNumber: stateForConvLayer1_1_depthNumber,
                                filterWidth: stateForConvLayer1_1_filterWidth,
                                filterHeight: stateForConvLayer1_1_filterHeight,
                                filterNumber: stateForConvLayer1_1_filterNumber,
                                filterStates: stateForConvLayer1_1_filterStates,
                                zeroPadding: stateForConvLayer1_1_zeroPadding,
                                stride: stateForConvLayer1_1_stride,
                                outputWidth: stateForConvLayer1_1_outputWidth,
                                outputHeight: stateForConvLayer1_1_outputHeight,
                                leraningRate: stateForConvLayer1_1_leraningRate
                              };
                              var _activate_linear = function (net) {
                                return net;
                              };
                              var match = ConvLayer$8_cnn.forward(ReluActivator$8_cnn.forward, stateForConvLayer1_1, inputs);
                              var match$1 = ConvLayer$8_cnn.forward(_activate_linear, stateForConvLayer2$1, match[1][1]);
                              var err1 = NP$8_cnn.sumMatrixMap(match$1[1][1]);
                              var stateForConvLayer1_2_inputWidth = stateForConvLayer1$2.inputWidth;
                              var stateForConvLayer1_2_inputHeight = stateForConvLayer1$2.inputHeight;
                              var stateForConvLayer1_2_depthNumber = stateForConvLayer1$2.depthNumber;
                              var stateForConvLayer1_2_filterWidth = stateForConvLayer1$2.filterWidth;
                              var stateForConvLayer1_2_filterHeight = stateForConvLayer1$2.filterHeight;
                              var stateForConvLayer1_2_filterNumber = stateForConvLayer1$2.filterNumber;
                              var stateForConvLayer1_2_filterStates = ImmutableSparseMap$8_cnn.set(stateForConvLayer1$2.filterStates, 0, {
                                    weights: ImmutableSparseMap$8_cnn.set(weights, depthIndex, MatrixUtils$8_cnn.setValue(NP$8_cnn.copyMatrix(weight), weightValue - 10e-4, rowIndex, colIndex)),
                                    bias: filterState.bias,
                                    weightGradients: filterState.weightGradients,
                                    biasGradient: filterState.biasGradient
                                  });
                              var stateForConvLayer1_2_zeroPadding = stateForConvLayer1$2.zeroPadding;
                              var stateForConvLayer1_2_stride = stateForConvLayer1$2.stride;
                              var stateForConvLayer1_2_outputWidth = stateForConvLayer1$2.outputWidth;
                              var stateForConvLayer1_2_outputHeight = stateForConvLayer1$2.outputHeight;
                              var stateForConvLayer1_2_leraningRate = stateForConvLayer1$2.leraningRate;
                              var stateForConvLayer1_2 = {
                                inputWidth: stateForConvLayer1_2_inputWidth,
                                inputHeight: stateForConvLayer1_2_inputHeight,
                                depthNumber: stateForConvLayer1_2_depthNumber,
                                filterWidth: stateForConvLayer1_2_filterWidth,
                                filterHeight: stateForConvLayer1_2_filterHeight,
                                filterNumber: stateForConvLayer1_2_filterNumber,
                                filterStates: stateForConvLayer1_2_filterStates,
                                zeroPadding: stateForConvLayer1_2_zeroPadding,
                                stride: stateForConvLayer1_2_stride,
                                outputWidth: stateForConvLayer1_2_outputWidth,
                                outputHeight: stateForConvLayer1_2_outputHeight,
                                leraningRate: stateForConvLayer1_2_leraningRate
                              };
                              var match$2 = ConvLayer$8_cnn.forward(ReluActivator$8_cnn.forward, stateForConvLayer1_2, inputs);
                              var match$3 = ConvLayer$8_cnn.forward(_activate_linear, stateForConvLayer2$1, match$2[1][1]);
                              var err2 = NP$8_cnn.sumMatrixMap(match$3[1][1]);
                              var expectedGradient = (err1 - err2) / (2 * 10e-4);
                              var result = FloatUtils$8_cnn.truncateFloatValue(expectedGradient, 4) === FloatUtils$8_cnn.truncateFloatValue(actualGradient, 4);
                              console.log("check gradient -> weights(" + depthIndex + "), " + rowIndex + ", " + colIndex + "): " + result);
                              Log$8_cnn.printForDebug([
                                    expectedGradient,
                                    actualGradient
                                  ]);
                              
                            }));
              }));
}

checkGradient(undefined);

exports.checkGradient = checkGradient;
/*  Not a pure module */
