let computeDelta = (output, label) => {
  Vector.sub(output, label)
}

let compute = (~output, ~label, ~epsion=1e-10, ()) => {
  -.Vector.dot(
    label,
    output
    ->Vector.addScalar(epsion)
    ->Vector.map(Js.Math.log)
    ->Vector.multiplyScalar(1. /. (1. +. epsion)),
  )
}

let buildData = (): LossType.data => {
  computeDelta: computeDelta,
}
