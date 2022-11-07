

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Mnist from "mnist";
import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
import * as Log$Gender_analyze from "./Log.bs.js";
import * as Mnist$Gender_analyze from "./mnist.bs.js";
import * as Matrix$Gender_analyze from "./Matrix.bs.js";
import * as Vector$Gender_analyze from "./Vector.bs.js";
import * as ArraySt$Gender_analyze from "./ArraySt.bs.js";
import * as OptionSt$Gender_analyze from "./OptionSt.bs.js";
import * as MatrixUtils$Gender_analyze from "./MatrixUtils.bs.js";

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
  var layerNet = _handleInputToAvoidTooLargeForSigmoid(Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector), Matrix$Gender_analyze.getColCount(state.wMatrixBetweenLayer1Layer2));
  var layerOutputVector = Vector$Gender_analyze.map(layerNet, activate);
  return [
          layerNet,
          layerOutputVector
        ];
}

function _forwardLayer3(activate, layer2OutputVector, state) {
  var layerNet = _handleInputToAvoidTooLargeForSigmoid(Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer2Layer3, Vector$Gender_analyze.push(layer2OutputVector, 1.0)), Matrix$Gender_analyze.getColCount(state.wMatrixBetweenLayer2Layer3));
  var layerOutputVector = Vector$Gender_analyze.map(layerNet, activate);
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
  var match = Vector$Gender_analyze.reducei(outputVector, (function (param, value, index) {
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
      ]);
  return OptionSt$Gender_analyze.getExn(match[1]);
}

function _isCorrectInference(labelVector, predictVector) {
  return _getOutputNumber(labelVector) === _getOutputNumber(predictVector);
}

function _getCorrectRate(correctCount, errorCount) {
  return correctCount / (correctCount + errorCount) * 100 + "%";
}

function train(state, sampleCount) {
  var mnistData = Mnist.set(sampleCount, 1);
  var features = Mnist$Gender_analyze.getMnistData(mnistData.training);
  var labels = Mnist$Gender_analyze.getMnistLabels(mnistData.training);
  var n = ArraySt$Gender_analyze.length(features);
  return ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, 99), (function (state, epoch) {
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
                                  wMatrixBetweenLayer1Layer2: Matrix$Gender_analyze.sub(state.wMatrixBetweenLayer1Layer2, Matrix$Gender_analyze.multiplyScalar(10, match$1[0])),
                                  wMatrixBetweenLayer2Layer3: Matrix$Gender_analyze.sub(state.wMatrixBetweenLayer2Layer3, Matrix$Gender_analyze.multiplyScalar(10, match$1[1]))
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

function inferenceWithSampleCount(state, sampleCount) {
  var mnistData = Mnist.set(0, sampleCount);
  var testData = Mnist$Gender_analyze.getMnistData(mnistData.test);
  var testLabels = Mnist$Gender_analyze.getMnistLabels(mnistData.test);
  var match = Log$Gender_analyze.printForDebug(ArraySt$Gender_analyze.reduceOneParami(testData, (function (param, data, i) {
              var errorCount = param[1];
              var correctCount = param[0];
              if (_isCorrectInference(Vector$Gender_analyze.create(Caml_array.get(testLabels, i)), inference(state, data))) {
                return [
                        correctCount + 1 | 0,
                        errorCount
                      ];
              } else {
                return [
                        correctCount,
                        errorCount + 1 | 0
                      ];
              }
            }), [
            0,
            0
          ]));
  return _getCorrectRate(match[0], match[1]);
}

var state = createState(784, 30, 10);

var state$1 = train(state, 10);

console.log([
      "inference correctRate:",
      inferenceWithSampleCount(state$1, 100)
    ]);

export {
  _createWMatrix ,
  createState ,
  _activate_sigmoid ,
  _deriv_sigmoid ,
  _activate_linear ,
  _deriv_linear ,
  _handleInputToAvoidTooLargeForSigmoid ,
  _forwardLayer2 ,
  _forwardLayer3 ,
  forward ,
  _bpLayer3Delta ,
  _bpLayer2Delta ,
  backward ,
  _computeLoss ,
  _createInputVector ,
  _getOutputNumber ,
  _isCorrectInference ,
  _getCorrectRate ,
  train ,
  inference ,
  inferenceWithSampleCount ,
  state$1 as state,
  
}
/* state Not a pure module */
