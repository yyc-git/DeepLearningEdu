'use strict';


function forward(x) {
  return Math.max(0, x);
}

function backward(x) {
  if (x > 0) {
    return 1;
  } else {
    return 0;
  }
}

function invert(y) {
  return y;
}

exports.forward = forward;
exports.backward = backward;
exports.invert = invert;
/* No side effect */
