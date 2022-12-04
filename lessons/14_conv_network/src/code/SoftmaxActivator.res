open ActivatorType

let _handleInputToAvoidUnderflowOrOverflow = net => {
  let maxValue = Vector.max(net)

  net->Vector.map(netValue => {
    netValue -. maxValue
  })
}

let _forward = (netValue, net) => {
  // TODO use requireCheck
  Js.Math.exp(netValue) > NumberUtils.getMaxNumber()
    ? Exception.throwErr({j`netValue: $netValue is too large`})
    : ()

  Js.Math.exp(netValue) /. net->Vector.reducei((. sum, netValue, i) => {
    sum +. Js.Math.exp(netValue)
  }, 0.)
}

let forwardNet = net => {
  let net = net->_handleInputToAvoidUnderflowOrOverflow

  net->Vector.map(netValue => {
    _forward(netValue, net)
  })
}

let forwardMatrix = matrix => {
  Exception.throwErr({j`softmax not support matrix`})
}

let backward = x => {
  Exception.notImplement()
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
