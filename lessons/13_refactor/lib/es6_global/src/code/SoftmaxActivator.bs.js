

import * as Vector$Cnn from "./Vector.bs.js";
import * as Exception$Cnn from "./Exception.bs.js";
import * as NumberUtils$Cnn from "./NumberUtils.bs.js";

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

export {
  _handleInputToAvoidUnderflowOrOverflow ,
  _forward ,
  forwardNet ,
  forwardMatrix ,
  backward ,
  invert ,
  buildData ,
  
}
/* No side effect */
