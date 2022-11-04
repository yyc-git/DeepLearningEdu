'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Js_math = require("rescript/lib/js/js_math.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var Matrix$Gender_analyze = require("./Matrix.bs.js");
var Vector$Gender_analyze = require("./Vector.bs.js");
var ArraySt$Gender_analyze = require("./ArraySt.bs.js");
var FloatUtils$Gender_analyze = require("./FloatUtils.bs.js");
var MatrixUtils$Gender_analyze = require("./MatrixUtils.bs.js");

function _createWMatrix(getValueFunc, firstLayerNodeCount, secondLayerNodeCount) {
  var col = firstLayerNodeCount + 1 | 0;
  return Matrix$Gender_analyze.create(secondLayerNodeCount, col, ArraySt$Gender_analyze.map(ArraySt$Gender_analyze.range(0, Math.imul(secondLayerNodeCount, col) - 1 | 0), (function (param) {
                    return Curry._1(getValueFunc, undefined);
                  })));
}

function createState(layer1NodeCount, layer2NodeCount, layer3NodeCount) {
  return {
          wMatrixBetweenLayer1Layer2: _createWMatrix((function (prim) {
                  return Math.random();
                }), layer1NodeCount, layer2NodeCount),
          wMatrixBetweenLayer2Layer3: _createWMatrix((function (prim) {
                  return Math.random();
                }), layer2NodeCount, layer3NodeCount)
        };
}

function _activateFunc(x) {
  return 1 / (1 + Math.exp(-x));
}

function _deriv_Sigmoid(x) {
  var fx = _activateFunc(x);
  return fx * (1 - fx);
}

function _forwardLayer2(inputVector, state) {
  var layerNet = Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector);
  var layerOutputVector = Vector$Gender_analyze.map(layerNet, _activateFunc);
  return [
          layerNet,
          layerOutputVector
        ];
}

function _forwardLayer3(layer2OutputVector, state) {
  var layerNet = Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer2Layer3, Vector$Gender_analyze.push(layer2OutputVector, 1.0));
  var layerOutputVector = Vector$Gender_analyze.map(layerNet, _activateFunc);
  return [
          layerNet,
          layerOutputVector
        ];
}

function forward(inputVector, state) {
  var match = _forwardLayer2(inputVector, state);
  var layer2OutputVector = match[1];
  var match$1 = _forwardLayer3(layer2OutputVector, state);
  return [
          [
            match[0],
            layer2OutputVector
          ],
          [
            match$1[0],
            match$1[1]
          ]
        ];
}

function _bpLayer3Delta(layer3Net, layer3OutputVector, n, labelVector) {
  return Vector$Gender_analyze.mapi(layer3OutputVector, (function (layer3OutputValue, i) {
                var d_E_d_value = -2 / n * (Vector$Gender_analyze.getExn(labelVector, i) - layer3OutputValue);
                var d_y_net_value = _deriv_Sigmoid(Vector$Gender_analyze.getExn(layer3Net, i));
                return d_E_d_value * d_y_net_value;
              }));
}

function _bpLayer2Delta(layer2Net, layer3Delta, state) {
  return Vector$Gender_analyze.mapi(layer2Net, (function (layer2NetValue, i) {
                return Vector$Gender_analyze.dot(layer3Delta, MatrixUtils$Gender_analyze.getCol(Matrix$Gender_analyze.getRowCount(state.wMatrixBetweenLayer2Layer3), Matrix$Gender_analyze.getColCount(state.wMatrixBetweenLayer2Layer3), i, Matrix$Gender_analyze.getData(state.wMatrixBetweenLayer2Layer3))) * _deriv_Sigmoid(layer2NetValue);
              }));
}

function backward(param, n, label, inputVector, state) {
  var match = param[1];
  var match$1 = param[0];
  var labelVector = Vector$Gender_analyze.create([label]);
  var layer3Delta = _bpLayer3Delta(match[0], match[1], n, labelVector);
  var layer2Delta = _bpLayer2Delta(match$1[0], layer3Delta, state);
  var layer2Gradient = Matrix$Gender_analyze.multiply(Matrix$Gender_analyze.create(Vector$Gender_analyze.length(layer2Delta), 1, layer2Delta), Matrix$Gender_analyze.create(1, Vector$Gender_analyze.length(inputVector), inputVector));
  var layer2OutputVector = Vector$Gender_analyze.push(match$1[1], 1.0);
  var layer3Gradient = Matrix$Gender_analyze.multiply(Matrix$Gender_analyze.create(Vector$Gender_analyze.length(layer3Delta), 1, layer3Delta), Matrix$Gender_analyze.create(1, Vector$Gender_analyze.length(layer2OutputVector), layer2OutputVector));
  return [
          layer2Gradient,
          layer3Gradient
        ];
}

function _convertLabelToFloat(label) {
  if (label) {
    return 1;
  } else {
    return 0;
  }
}

function _computeLoss(labels, outputs) {
  return ArraySt$Gender_analyze.reduceOneParami(labels, (function (result, label, i) {
                return result + Math.pow(label - Caml_array.get(outputs, i), 2.0);
              }), 0) / ArraySt$Gender_analyze.length(labels);
}

function _createInputVector(feature) {
  return Vector$Gender_analyze.create([
              feature.height,
              feature.weight,
              1.0
            ]);
}

function train(state, features, labels) {
  var n = ArraySt$Gender_analyze.length(features);
  return ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, 999), (function (state, epoch) {
                var state$1 = ArraySt$Gender_analyze.reduceOneParami(features, (function (state, feature, i) {
                        var label = Caml_array.get(labels, i) ? 1 : 0;
                        var inputVector = _createInputVector(feature);
                        var match = backward(forward(inputVector, state), n, label, inputVector, state);
                        return {
                                wMatrixBetweenLayer1Layer2: Matrix$Gender_analyze.sub(state.wMatrixBetweenLayer1Layer2, Matrix$Gender_analyze.multiplyScalar(0.1, match[0])),
                                wMatrixBetweenLayer2Layer3: Matrix$Gender_analyze.sub(state.wMatrixBetweenLayer2Layer3, Matrix$Gender_analyze.multiplyScalar(0.1, match[1]))
                              };
                      }), state);
                if (epoch % 10 === 0) {
                  console.log([
                        "loss: ",
                        _computeLoss(ArraySt$Gender_analyze.map(labels, _convertLabelToFloat), ArraySt$Gender_analyze.map(features, (function (feature) {
                                    var inputVector = _createInputVector(feature);
                                    var match = forward(inputVector, state$1);
                                    return Vector$Gender_analyze.getExn(match[1][1], 0);
                                  })))
                      ]);
                  return state$1;
                } else {
                  return state$1;
                }
              }), state);
}

function inference(state, feature) {
  var inputVector = _createInputVector(feature);
  var match = forward(inputVector, state);
  return Vector$Gender_analyze.getExn(match[1][1], 0);
}

function _mean(values) {
  return ArraySt$Gender_analyze.reduceOneParam(values, (function (sum, value) {
                return sum + value;
              }), 0) / ArraySt$Gender_analyze.length(values);
}

function _zeroMean(features) {
  var weightMean = Js_math.floor(_mean(ArraySt$Gender_analyze.map(features, (function (feature) {
                  return feature.weight;
                }))));
  var heightMean = Js_math.floor(_mean(ArraySt$Gender_analyze.map(features, (function (feature) {
                  return feature.height;
                }))));
  return ArraySt$Gender_analyze.map(features, (function (feature) {
                return {
                        weight: feature.weight - weightMean,
                        height: feature.height - heightMean
                      };
              }));
}

function checkGradient(inputVector, labelVector) {
  var _checkWeight = function (param, delta, param$1, state) {
    var previousLayerOutputVector = param$1[0];
    var computeErrorFunc = param[1];
    var forwardLayerFunc = param[0];
    var actualGradient = delta * param$1[1];
    var newState1 = Curry._2(param[2], state, 10e-4);
    var match = Curry._2(forwardLayerFunc, previousLayerOutputVector, newState1);
    var error1 = Curry._1(computeErrorFunc, match[1]);
    var newState2 = Curry._2(param[3], state, 10e-4);
    var match$1 = Curry._2(forwardLayerFunc, previousLayerOutputVector, newState2);
    var error2 = Curry._1(computeErrorFunc, match$1[1]);
    var expectedGradient = (error1 - error2) / (2 * 10e-4);
    console.log([
          "check gradient: ",
          FloatUtils$Gender_analyze.truncateFloatValue(expectedGradient, 4) === FloatUtils$Gender_analyze.truncateFloatValue(actualGradient, 4)
        ]);
    
  };
  var _computeErrorForLayer3 = function (labelVector, outputVector) {
    return _computeLoss(Vector$Gender_analyze.toArray(labelVector), Vector$Gender_analyze.toArray(outputVector));
  };
  var state = createState(2, 2, 1);
  var match = _forwardLayer2(inputVector, state);
  var layer2OutputVector = match[1];
  var match$1 = _forwardLayer3(layer2OutputVector, state);
  var layer3Delta = _bpLayer3Delta(match$1[0], match$1[1], 1.0, labelVector);
  var param = [
    (function (state, wMatrix) {
        return {
                wMatrixBetweenLayer1Layer2: state.wMatrixBetweenLayer1Layer2,
                wMatrixBetweenLayer2Layer3: wMatrix
              };
      }),
    _forwardLayer3,
    (function (param) {
        return _computeErrorForLayer3(labelVector, param);
      }),
    _checkWeight
  ];
  var wMatrix = state.wMatrixBetweenLayer2Layer3;
  var checkWeightFunc = param[3];
  var computeErrorFunc = param[2];
  var forwardLayerFunc = param[1];
  var updateWMatrixFunc = param[0];
  return Matrix$Gender_analyze.forEachRow(wMatrix, (function (rowIndex) {
                return Matrix$Gender_analyze.forEachCol(wMatrix, (function (colIndex) {
                              return Curry._4(checkWeightFunc, [
                                          forwardLayerFunc,
                                          computeErrorFunc,
                                          (function (state, epsilon) {
                                              var col = wMatrix[1];
                                              var data = wMatrix[2].slice();
                                              Caml_array.set(data, MatrixUtils$Gender_analyze.computeIndex(col, rowIndex, colIndex), Caml_array.get(data, MatrixUtils$Gender_analyze.computeIndex(col, rowIndex, colIndex)) + epsilon);
                                              return Curry._2(updateWMatrixFunc, state, [
                                                          wMatrix[0],
                                                          col,
                                                          data
                                                        ]);
                                            }),
                                          (function (state, epsilon) {
                                              var col = wMatrix[1];
                                              var data = wMatrix[2].slice();
                                              Caml_array.set(data, MatrixUtils$Gender_analyze.computeIndex(col, rowIndex, colIndex), Caml_array.get(data, MatrixUtils$Gender_analyze.computeIndex(col, rowIndex, colIndex)) - epsilon);
                                              return Curry._2(updateWMatrixFunc, state, [
                                                          wMatrix[0],
                                                          col,
                                                          data
                                                        ]);
                                            })
                                        ], Caml_array.get(layer3Delta, rowIndex), [
                                          layer2OutputVector,
                                          Vector$Gender_analyze.getExn(layer2OutputVector, colIndex)
                                        ], state);
                            }));
              }));
}

function testCheckGradient(param) {
  var inputVector = Vector$Gender_analyze.create([
        -2,
        -1,
        0
      ]);
  var labelVector = Vector$Gender_analyze.map(Vector$Gender_analyze.create([/* Female */1]), _convertLabelToFloat);
  return checkGradient(inputVector, labelVector);
}

console.log("begin test");

testCheckGradient(undefined);

console.log("finish test");

var state = createState(2, 2, 1);

var features = [
  {
    weight: 50,
    height: 150
  },
  {
    weight: 51,
    height: 149
  },
  {
    weight: 60,
    height: 172
  },
  {
    weight: 90,
    height: 188
  }
];

var labels = [
  /* Female */1,
  /* Female */1,
  /* Male */0,
  /* Male */0
];

var features$1 = _zeroMean(features);

var state$1 = train(state, features$1, labels);

var featuresForInference = [
  {
    weight: 89,
    height: 190
  },
  {
    weight: 60,
    height: 155
  }
];

var __x = _zeroMean(featuresForInference);

__x.forEach(function (feature) {
      console.log(inference(state$1, feature));
      
    });

exports._createWMatrix = _createWMatrix;
exports.createState = createState;
exports._activateFunc = _activateFunc;
exports._deriv_Sigmoid = _deriv_Sigmoid;
exports._forwardLayer2 = _forwardLayer2;
exports._forwardLayer3 = _forwardLayer3;
exports.forward = forward;
exports._bpLayer3Delta = _bpLayer3Delta;
exports._bpLayer2Delta = _bpLayer2Delta;
exports.backward = backward;
exports._convertLabelToFloat = _convertLabelToFloat;
exports._computeLoss = _computeLoss;
exports._createInputVector = _createInputVector;
exports.train = train;
exports.inference = inference;
exports._mean = _mean;
exports._zeroMean = _zeroMean;
exports.checkGradient = checkGradient;
exports.testCheckGradient = testCheckGradient;
exports.labels = labels;
exports.features = features$1;
exports.state = state$1;
exports.featuresForInference = featuresForInference;
/*  Not a pure module */
