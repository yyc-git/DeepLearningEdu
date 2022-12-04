open ActivatorType

let _forward = x => {
  x
}

let forwardNet = net => {
  net->Vector.map(_forward)
}

let forwardMatrix = matrix => {
  matrix->Matrix.map(_forward)
}

let backward = x => {
  1.
}

/* ! get x => x = f-1(y) */
let invert = y => {
  y
}

let buildData = () => {
  forwardNet: forwardNet,
  forwardMatrix: forwardMatrix,
  backward: backward,
  invert: invert,
}
