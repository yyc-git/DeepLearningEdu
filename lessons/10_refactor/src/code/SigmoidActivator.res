open ActivatorType

let _handleInputValueToAvoidTooLarge = (inputValue, maxOfhandleSigmoidInputToAvoidTooLarge) => {
  inputValue /. maxOfhandleSigmoidInputToAvoidTooLarge
}

let _handleInputVectorToAvoidTooLarge = (input, maxOfhandleSigmoidInputToAvoidTooLarge) => {
  switch maxOfhandleSigmoidInputToAvoidTooLarge {
  | None => input
  | Some(maxOfhandleSigmoidInputToAvoidTooLarge) =>
    input->(
      x =>
        x->ArraySt.map(v => {
          v->_handleInputValueToAvoidTooLarge(maxOfhandleSigmoidInputToAvoidTooLarge)
        })
    )
  }
}

let _forward = x => {
  DebugUtils.checkSigmoidInputTooLarge(x)

  1. /. (1. +. Js.Math.exp(-.x))
}

let forwardNet = net => {
  // TODO should pass maxOfhandleSigmoidInputToAvoidTooLarge
  let net = net->_handleInputVectorToAvoidTooLarge(Some(784. /. 10.))

  net->Vector.map(netValue => {
    _forward(netValue)
  })
}

let forwardMatrix = matrix => {
  Exception.throwErr({j`softmax not support matrix`})
}

let backward = x => {
  // TODO should pass maxOfhandleSigmoidInputToAvoidTooLarge
  let fx = _forward(x->_handleInputValueToAvoidTooLarge(784. /. 10.))

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
