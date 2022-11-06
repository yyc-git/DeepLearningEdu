'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Mnist = require("mnist");
var Caml_obj = require("rescript/lib/js/caml_obj.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var Mnist$Gender_analyze = require("./mnist.bs.js");
var Matrix$Gender_analyze = require("./Matrix.bs.js");
var Vector$Gender_analyze = require("./Vector.bs.js");
var ArraySt$Gender_analyze = require("./ArraySt.bs.js");
var MatrixUtils$Gender_analyze = require("./MatrixUtils.bs.js");

function _createWMatrix(getValue, firstLayerNodeCount, secondLayerNodeCount) {
  var col = firstLayerNodeCount + 1 | 0;
  return Matrix$Gender_analyze.create(secondLayerNodeCount, col, ArraySt$Gender_analyze.map(ArraySt$Gender_analyze.range(0, Math.imul(secondLayerNodeCount, col) - 1 | 0), (function (param) {
                    return Curry._1(getValue, undefined);
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

function _activate_sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function _deriv_sigmoid(x) {
  var fx = _activate_sigmoid(x);
  return fx * (1 - fx);
}

function _activate_linear(x) {
  return x;
}

function _deriv_linear(x) {
  return 1.0;
}

function _handleInputToAvoidTooLargeForSigmoid(input, max) {
  return ArraySt$Gender_analyze.map(input, (function (v) {
                return v / (max / 10);
              }));
}

function _forwardLayer2(activate, inputVector, state) {
  var layerNet = Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector);
  var layerOutputVector = Vector$Gender_analyze.map(_handleInputToAvoidTooLargeForSigmoid(layerNet, Matrix$Gender_analyze.getColCount(state.wMatrixBetweenLayer1Layer2)), activate);
  return [
          layerNet,
          layerOutputVector
        ];
}

function _forwardLayer3(activate, layer2OutputVector, state) {
  var layerNet = Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer2Layer3, Vector$Gender_analyze.push(layer2OutputVector, 1.0));
  var layerOutputVector = Vector$Gender_analyze.map(_handleInputToAvoidTooLargeForSigmoid(layerNet, Matrix$Gender_analyze.getColCount(state.wMatrixBetweenLayer2Layer3)), activate);
  return [
          layerNet,
          layerOutputVector
        ];
}

function forward(param, inputVector, state) {
  var match = _forwardLayer2(param[0], inputVector, state);
  var layer2OutputVector = match[1];
  var match$1 = _forwardLayer3(param[1], layer2OutputVector, state);
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

function _bpLayer3Delta(deriv, layer3Net, layer3OutputVector, n, labelVector) {
  return Vector$Gender_analyze.mapi(layer3OutputVector, (function (layer3OutputValue, i) {
                var d_E_d_value = -2 / n * (Vector$Gender_analyze.getExn(labelVector, i) - layer3OutputValue);
                var d_y_net_value = Curry._1(deriv, Vector$Gender_analyze.getExn(layer3Net, i));
                return d_E_d_value * d_y_net_value;
              }));
}

function _bpLayer2Delta(deriv, layer2Net, layer3Delta, state) {
  return Vector$Gender_analyze.mapi(layer2Net, (function (layer2NetValue, i) {
                return Vector$Gender_analyze.dot(layer3Delta, MatrixUtils$Gender_analyze.getCol(Matrix$Gender_analyze.getRowCount(state.wMatrixBetweenLayer2Layer3), Matrix$Gender_analyze.getColCount(state.wMatrixBetweenLayer2Layer3), i, Matrix$Gender_analyze.getData(state.wMatrixBetweenLayer2Layer3))) * Curry._1(deriv, layer2NetValue);
              }));
}

function backward(param, n, labelVector, inputVector, state) {
  var match = param[1];
  var match$1 = param[0];
  var layer3Delta = _bpLayer3Delta(_deriv_sigmoid, match[0], match[1], n, labelVector);
  var layer2Delta = _bpLayer2Delta(_deriv_sigmoid, match$1[0], layer3Delta, state);
  var layer2Gradient = Matrix$Gender_analyze.multiply(Matrix$Gender_analyze.create(Vector$Gender_analyze.length(layer2Delta), 1, layer2Delta), Matrix$Gender_analyze.create(1, Vector$Gender_analyze.length(inputVector), inputVector));
  var layer2OutputVector = Vector$Gender_analyze.push(match$1[1], 1.0);
  var layer3Gradient = Matrix$Gender_analyze.multiply(Matrix$Gender_analyze.create(Vector$Gender_analyze.length(layer3Delta), 1, layer3Delta), Matrix$Gender_analyze.create(1, Vector$Gender_analyze.length(layer2OutputVector), layer2OutputVector));
  return [
          layer2Gradient,
          layer3Gradient
        ];
}

function _computeLoss(labels, outputs) {
  return ArraySt$Gender_analyze.reduceOneParami(labels, (function (result, label, i) {
                return result + Math.pow(label - Caml_array.get(outputs, i), 2.0);
              }), 0) / ArraySt$Gender_analyze.length(labels);
}

function _createInputVector(feature) {
  return Vector$Gender_analyze.push(Vector$Gender_analyze.create(feature), 1.0);
}

function _getOutputNumber(outputVector) {
  return Vector$Gender_analyze.reducei(outputVector, (function (param, value, index) {
                    var maxValue = param[0];
                    if (value >= maxValue) {
                      return [
                              value,
                              index
                            ];
                    } else {
                      return [
                              maxValue,
                              param[1]
                            ];
                    }
                  }))([
                0,
                undefined
              ])[1];
}

function _isCorrectInference(labelVector, predictVector) {
  return Caml_obj.caml_equal(_getOutputNumber(labelVector), _getOutputNumber(predictVector));
}

function _getCorrectRate(correctCount, errorCount) {
  return correctCount / (correctCount + errorCount) * 100 + "%";
}

function train(state, sampleCount) {
  var mnistData = Mnist.set(sampleCount, 1);
  var features = Mnist$Gender_analyze.getMnistData(mnistData.training);
  var labels = Mnist$Gender_analyze.getMnistLabels(mnistData.training);
  var n = ArraySt$Gender_analyze.length(features);
  return ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, 9), (function (state, epoch) {
                var match = ArraySt$Gender_analyze.reduceOneParami(features, (function (param, feature, i) {
                        var match = param[1];
                        var errorCount = match[1];
                        var correctCount = match[0];
                        var state = param[0];
                        var labelVector = Vector$Gender_analyze.create(Caml_array.get(labels, i));
                        var inputVector = Vector$Gender_analyze.push(Vector$Gender_analyze.create(feature), 1.0);
                        var forwardOutput = forward([
                              _activate_sigmoid,
                              _activate_sigmoid
                            ], inputVector, state);
                        var match$1 = backward(forwardOutput, n, labelVector, inputVector, state);
                        return [
                                {
                                  wMatrixBetweenLayer1Layer2: Matrix$Gender_analyze.sub(state.wMatrixBetweenLayer1Layer2, Matrix$Gender_analyze.multiplyScalar(0.1, match$1[0])),
                                  wMatrixBetweenLayer2Layer3: Matrix$Gender_analyze.sub(state.wMatrixBetweenLayer2Layer3, Matrix$Gender_analyze.multiplyScalar(0.1, match$1[1]))
                                },
                                _isCorrectInference(labelVector, forwardOutput[1][1]) ? [
                                    correctCount + 1 | 0,
                                    errorCount
                                  ] : [
                                    correctCount,
                                    errorCount + 1 | 0
                                  ]
                              ];
                      }), [
                      state,
                      [
                        0,
                        0
                      ]
                    ]);
                var match$1 = match[1];
                console.log([
                      "getCorrectRate:",
                      _getCorrectRate(match$1[0], match$1[1])
                    ]);
                return match[0];
              }), state);
}

function inference(state, feature) {
  var inputVector = Vector$Gender_analyze.push(Vector$Gender_analyze.create(feature), 1.0);
  var match = forward([
        _activate_sigmoid,
        _activate_sigmoid
      ], inputVector, state);
  return match[1][1];
}

var state = createState(784, 30, 10);

var state$1 = train(state, 200);

exports._createWMatrix = _createWMatrix;
exports.createState = createState;
exports._activate_sigmoid = _activate_sigmoid;
exports._deriv_sigmoid = _deriv_sigmoid;
exports._activate_linear = _activate_linear;
exports._deriv_linear = _deriv_linear;
exports._handleInputToAvoidTooLargeForSigmoid = _handleInputToAvoidTooLargeForSigmoid;
exports._forwardLayer2 = _forwardLayer2;
exports._forwardLayer3 = _forwardLayer3;
exports.forward = forward;
exports._bpLayer3Delta = _bpLayer3Delta;
exports._bpLayer2Delta = _bpLayer2Delta;
exports.backward = backward;
exports._computeLoss = _computeLoss;
exports._createInputVector = _createInputVector;
exports._getOutputNumber = _getOutputNumber;
exports._isCorrectInference = _isCorrectInference;
exports._getCorrectRate = _getCorrectRate;
exports.train = train;
exports.inference = inference;
exports.state = state$1;
/* state Not a pure module */
