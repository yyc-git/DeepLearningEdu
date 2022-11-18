'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var Matrix$Gender_analyze = require("../Matrix.bs.js");
var Vector$Gender_analyze = require("../Vector.bs.js");
var ArraySt$Gender_analyze = require("../ArraySt.bs.js");
var FloatUtils$Gender_analyze = require("../FloatUtils.bs.js");
var MatrixUtils$Gender_analyze = require("../MatrixUtils.bs.js");

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

function _deriv_sigmoid(x) {
  var fx = _activateFunc(x);
  return fx * (1 - fx);
}

function forward(inputVector, state) {
  var layer2Net = Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector);
  var layer2OutputVector = Vector$Gender_analyze.map(layer2Net, _activateFunc);
  var layer3Net = Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer2Layer3, Vector$Gender_analyze.push(layer2OutputVector, 1.0));
  var layer3OutputVector = Vector$Gender_analyze.map(layer3Net, _activateFunc);
  return [
          [
            layer2Net,
            layer2OutputVector
          ],
          [
            layer3Net,
            layer3OutputVector
          ]
        ];
}

function backward(param, n, label, inputVector, state) {
  var match = param[1];
  var layer3Net = match[0];
  var match$1 = param[0];
  var labelVector = Vector$Gender_analyze.create([label]);
  var layer3Delta = Vector$Gender_analyze.mapi(match[1], (function (layer3OutputValue, i) {
          var d_E_d_value = -2 / n * (Vector$Gender_analyze.getExn(labelVector, i) - layer3OutputValue);
          var d_y_net_value = _deriv_sigmoid(Vector$Gender_analyze.getExn(layer3Net, i));
          return d_E_d_value * d_y_net_value;
        }));
  var layer2Delta = Vector$Gender_analyze.mapi(match$1[0], (function (layer2NetValue, i) {
          return Vector$Gender_analyze.dot(layer3Delta, MatrixUtils$Gender_analyze.getCol(Matrix$Gender_analyze.getRowCount(state.wMatrixBetweenLayer2Layer3), Matrix$Gender_analyze.getColCount(state.wMatrixBetweenLayer2Layer3), i, Matrix$Gender_analyze.getData(state.wMatrixBetweenLayer2Layer3))) * _deriv_sigmoid(layer2NetValue);
        }));
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

function checkGradient(inputVector, labelVector) {
  var _computeErrorForLayer3 = function (labelVector, outputVector) {
    return _computeLoss(Vector$Gender_analyze.toArray(labelVector), Vector$Gender_analyze.toArray(outputVector));
  };
  var state = createState(2, 2, 1);
  var forwardOutput = forward(inputVector, state);
  var match = backward(forwardOutput, 1.0, Vector$Gender_analyze.getExn(labelVector, 0), inputVector, state);
  var actualGradient = MatrixUtils$Gender_analyze.getValue(0, 0, match[1]);
  var wMatrix = state.wMatrixBetweenLayer2Layer3;
  var col = wMatrix[1];
  var data = wMatrix[2].slice();
  Caml_array.set(data, MatrixUtils$Gender_analyze.computeIndex(col, 0, 0), Caml_array.get(data, MatrixUtils$Gender_analyze.computeIndex(col, 0, 0)) + 10e-4);
  var newState1_wMatrixBetweenLayer1Layer2 = state.wMatrixBetweenLayer1Layer2;
  var newState1_wMatrixBetweenLayer2Layer3 = [
    wMatrix[0],
    col,
    data
  ];
  var newState1 = {
    wMatrixBetweenLayer1Layer2: newState1_wMatrixBetweenLayer1Layer2,
    wMatrixBetweenLayer2Layer3: newState1_wMatrixBetweenLayer2Layer3
  };
  var match$1 = forward(inputVector, newState1);
  var error1 = _computeErrorForLayer3(labelVector, match$1[1][1]);
  var col$1 = wMatrix[1];
  var data$1 = wMatrix[2].slice();
  Caml_array.set(data$1, MatrixUtils$Gender_analyze.computeIndex(col$1, 0, 0), Caml_array.get(data$1, MatrixUtils$Gender_analyze.computeIndex(col$1, 0, 0)) - 10e-4);
  var newState2_wMatrixBetweenLayer1Layer2 = state.wMatrixBetweenLayer1Layer2;
  var newState2_wMatrixBetweenLayer2Layer3 = [
    wMatrix[0],
    col$1,
    data$1
  ];
  var newState2 = {
    wMatrixBetweenLayer1Layer2: newState2_wMatrixBetweenLayer1Layer2,
    wMatrixBetweenLayer2Layer3: newState2_wMatrixBetweenLayer2Layer3
  };
  var match$2 = forward(inputVector, newState2);
  var error2 = _computeErrorForLayer3(labelVector, match$2[1][1]);
  var expectedGradient = (error1 - error2) / (2 * 10e-4);
  console.log([
        "check gradient: ",
        FloatUtils$Gender_analyze.truncateFloatValue(expectedGradient, 4) === FloatUtils$Gender_analyze.truncateFloatValue(actualGradient, 4)
      ]);
  
}

function testCheckOutputLayerGradient(param) {
  var inputVector = Vector$Gender_analyze.create([
        -2,
        -1,
        1
      ]);
  var labelVector = Vector$Gender_analyze.map(Vector$Gender_analyze.create([/* Female */1]), _convertLabelToFloat);
  return checkGradient(inputVector, labelVector);
}

console.log("begin test");

testCheckOutputLayerGradient(undefined);

console.log("finish test");

exports._createWMatrix = _createWMatrix;
exports.createState = createState;
exports._activateFunc = _activateFunc;
exports._deriv_sigmoid = _deriv_sigmoid;
exports.forward = forward;
exports.backward = backward;
exports._convertLabelToFloat = _convertLabelToFloat;
exports._computeLoss = _computeLoss;
exports._createInputVector = _createInputVector;
exports.train = train;
exports.inference = inference;
exports.checkGradient = checkGradient;
exports.testCheckOutputLayerGradient = testCheckOutputLayerGradient;
/*  Not a pure module */
