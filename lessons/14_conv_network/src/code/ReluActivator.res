open ActivatorType

let _forward = x => {
  Js.Math.max_float(0., x)
}

let forwardNet = net => {
  net->Vector.map(_forward)
}

let forwardMatrix = matrix => {
  matrix->Matrix.map(_forward)
}

let backward = x => {
  x > 0. ? 1. : 0.
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
