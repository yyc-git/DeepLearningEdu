'use strict';

var Vector$Cnn = require("./Vector.bs.js");
var ArraySt$Cnn = require("./ArraySt.bs.js");
var Exception$Cnn = require("./Exception.bs.js");
var DebugUtils$Cnn = require("./DebugUtils.bs.js");

function _handleInputValueToAvoidTooLarge(inputValue, maxOfhandleSigmoidInputToAvoidTooLarge) {
  return inputValue / maxOfhandleSigmoidInputToAvoidTooLarge;
}

function _handleInputVectorToAvoidTooLarge(input, maxOfhandleSigmoidInputToAvoidTooLarge) {
  if (maxOfhandleSigmoidInputToAvoidTooLarge !== undefined) {
    return ArraySt$Cnn.map(input, (function (v) {
                  return v / maxOfhandleSigmoidInputToAvoidTooLarge;
                }));
  } else {
    return input;
  }
}

function _forward(x) {
  DebugUtils$Cnn.checkSigmoidInputTooLarge(x);
  return 1 / (1 + Math.exp(-x));
}

function forwardNet(net) {
  var net$1 = _handleInputVectorToAvoidTooLarge(net, 784 / 10);
  return Vector$Cnn.map(net$1, _forward);
}

function forwardMatrix(matrix) {
  return Exception$Cnn.throwErr("softmax not support matrix");
}

function backward(x) {
  var fx = _forward(x / (784 / 10));
  return fx * (1 - fx);
}

function invert(y) {
  return Exception$Cnn.notImplement(undefined);
}

function buildData(param) {
  return {
          forwardNet: forwardNet,
          forwardMatrix: forwardMatrix,
          backward: backward,
          invert: invert
        };
}

exports._handleInputValueToAvoidTooLarge = _handleInputValueToAvoidTooLarge;
exports._handleInputVectorToAvoidTooLarge = _handleInputVectorToAvoidTooLarge;
exports._forward = _forward;
exports.forwardNet = forwardNet;
exports.forwardMatrix = forwardMatrix;
exports.backward = backward;
exports.invert = invert;
exports.buildData = buildData;
/* No side effect */
