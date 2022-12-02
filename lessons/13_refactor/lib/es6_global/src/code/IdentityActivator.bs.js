

import * as Matrix$Cnn from "./Matrix.bs.js";
import * as Vector$Cnn from "./Vector.bs.js";

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

export {
  _forward ,
  forwardNet ,
  forwardMatrix ,
  backward ,
  invert ,
  buildData ,
  
}
/* No side effect */
