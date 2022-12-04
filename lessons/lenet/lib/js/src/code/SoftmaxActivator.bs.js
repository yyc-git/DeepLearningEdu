'use strict';

var Vector$Cnn = require("./Vector.bs.js");
var Exception$Cnn = require("./Exception.bs.js");
var NumberUtils$Cnn = require("./NumberUtils.bs.js");

function _handleInputToAvoidUnderflowOrOverflow(net) {
  var maxValue = Vector$Cnn.max(net);
  return Vector$Cnn.map(net, (function (netValue) {
                return netValue - maxValue;
              }));
}

function _forward(netValue, net) {
  if (Math.exp(netValue) > NumberUtils$Cnn.getMaxNumber(undefined)) {
    Exception$Cnn.throwErr("netValue: " + netValue + " is too large");
  }
  return Math.exp(netValue) / Vector$Cnn.reducei(net, (function (sum, netValue, i) {
                  return sum + Math.exp(netValue);
                }))(0);
}

function forwardNet(net) {
  var net$1 = _handleInputToAvoidUnderflowOrOverflow(net);
  return Vector$Cnn.map(net$1, (function (netValue) {
                return _forward(netValue, net$1);
              }));
}

function forwardMatrix(matrix) {
  return Exception$Cnn.throwErr("softmax not support matrix");
}

function backward(x) {
  return Exception$Cnn.notImplement(undefined);
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

exports._handleInputToAvoidUnderflowOrOverflow = _handleInputToAvoidUnderflowOrOverflow;
exports._forward = _forward;
exports.forwardNet = forwardNet;
exports.forwardMatrix = forwardMatrix;
exports.backward = backward;
exports.invert = invert;
exports.buildData = buildData;
/* No side effect */
