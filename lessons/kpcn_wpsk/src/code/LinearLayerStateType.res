type adamData = {
  vWeight: Matrix.t,
  vBias: Vector.t,
  sWeight: Matrix.t,
  sBias: Vector.t,
}

type state = {
  weight: Matrix.t,
  bias: Vector.t,
  adamData: adamData,
}
