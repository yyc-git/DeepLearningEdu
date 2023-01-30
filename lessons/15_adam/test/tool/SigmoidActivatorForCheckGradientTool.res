open ActivatorType

let _forward = x => {
  DebugUtils.checkSigmoidInputTooLarge(x)

  1. /. (1. +. Js.Math.exp(-.x))
}

let forwardNet = net => {
  net->Vector.map(netValue => {
    _forward(netValue)
  })
}

let forwardMatrix = matrix => {
  Exception.throwErr({j`softmax not support matrix`})
}

let backward = x => {
  let fx = _forward(x)

  fx *. (1. -. fx)
}

let invert = y => {
  Exception.notImplement()
}

let buildData = () => {
  forwardNet: forwardNet,
  forwardMatrix: forwardMatrix,
  backward: backward,
  invert: invert,
}
