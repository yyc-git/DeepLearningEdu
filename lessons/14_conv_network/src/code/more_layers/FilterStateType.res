// type depthIndex = int

type adamData = {
  vWeight: MatrixMap.t,
  vBias: float,
  sWeight: MatrixMap.t,
  sBias: float,
}

type state = {
  weights: MatrixMap.t,
  bias: float,
  adamData: adamData,
}
