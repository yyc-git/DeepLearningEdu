'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var Matrix$Gender_analyze = require("./Matrix.bs.js");
var Vector$Gender_analyze = require("./Vector.bs.js");
var ArraySt$Gender_analyze = require("./ArraySt.bs.js");

function _createWMatrix(getValueFunc, firstLayerNodeCount, secondLayerNodeCount) {
  var col = firstLayerNodeCount + 1 | 0;
  return Matrix$Gender_analyze.create(secondLayerNodeCount, col, ArraySt$Gender_analyze.map(ArraySt$Gender_analyze.range(0, Math.imul(secondLayerNodeCount, col) - 1 | 0), (function (param) {
                    return Curry._1(getValueFunc, undefined);
                  })));
}

function createState(layer1NodeCount, layer2NodeCount, layer3NodeCount) {
  return {
          wMatrixBetweenLayer1Layer2: _createWMatrix((function (param) {
                  return 0.1;
                }), layer1NodeCount, layer2NodeCount),
          wMatrixBetweenLayer2Layer3: _createWMatrix((function (param) {
                  return 0.1;
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
  return 1;
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

var state$1 = train(state, features, labels);

exports._createWMatrix = _createWMatrix;
exports.createState = createState;
exports._activateFunc = _activateFunc;
exports._deriv_Sigmoid = _deriv_Sigmoid;
exports.forward = forward;
exports.backward = backward;
exports._convertLabelToFloat = _convertLabelToFloat;
exports._computeLoss = _computeLoss;
exports._createInputVector = _createInputVector;
exports.train = train;
exports.features = features;
exports.labels = labels;
exports.state = state$1;
/* state Not a pure module */
