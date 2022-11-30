

import * as Vector$Cnn from "./Vector.bs.js";
import * as ArraySt$Cnn from "./ArraySt.bs.js";
import * as Exception$Cnn from "./Exception.bs.js";
import * as DebugUtils$Cnn from "./DebugUtils.bs.js";

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
  var net$1 = _handleInputVectorToAvoidTooLarge(net, 784 / 30);
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

export {
  _handleInputValueToAvoidTooLarge ,
  _handleInputVectorToAvoidTooLarge ,
  _forward ,
  forwardNet ,
  forwardMatrix ,
  backward ,
  invert ,
  buildData ,
  
}
/* No side effect */
