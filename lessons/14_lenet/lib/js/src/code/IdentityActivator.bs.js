'use strict';

var Matrix$Cnn = require("./Matrix.bs.js");
var Vector$Cnn = require("./Vector.bs.js");

function _forward(x) {
  return x;
}

function forwardNet(net) {
  return Vector$Cnn.map(net, _forward);
}

function forwardMatrix(matrix) {
  return Matrix$Cnn.map(matrix, _forward);
}

function backward(x) {
  return 1;
}

function invert(y) {
  return y;
}

function buildData(param) {
  return {
          forwardNet: forwardNet,
          forwardMatrix: forwardMatrix,
          backward: backward,
          invert: invert
        };
}

exports._forward = _forward;
exports.forwardNet = forwardNet;
exports.forwardMatrix = forwardMatrix;
exports.backward = backward;
exports.invert = invert;
exports.buildData = buildData;
/* No side effect */
