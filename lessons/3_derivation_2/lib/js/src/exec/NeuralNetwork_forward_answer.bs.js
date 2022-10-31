'use strict';

var Curry = require("rescript/lib/js/curry.js");
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

function forward(state, feature) {
  var inputVector = Vector$Gender_analyze.create([
        feature.height,
        feature.weight,
        1.0
      ]);
  var layer2OutputVector = Vector$Gender_analyze.map(Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector), _activateFunc);
  var layer3OutputVector = Vector$Gender_analyze.map(Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer2Layer3, Vector$Gender_analyze.push(layer2OutputVector, 1.0)), _activateFunc);
  return [
          layer2OutputVector,
          layer3OutputVector
        ];
}

var state = createState(2, 2, 1);

var feature = {
  weight: 50,
  height: 150
};

console.log(forward(state, feature));

exports._createWMatrix = _createWMatrix;
exports.createState = createState;
exports._activateFunc = _activateFunc;
exports.forward = forward;
exports.state = state;
exports.feature = feature;
/* state Not a pure module */
