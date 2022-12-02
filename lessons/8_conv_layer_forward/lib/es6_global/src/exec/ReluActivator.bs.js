


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

export {
  forward ,
  backward ,
  invert ,
  
}
/* No side effect */
