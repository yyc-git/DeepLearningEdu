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

exports.forward = forward;
exports.backward = backward;
/* No side effect */
