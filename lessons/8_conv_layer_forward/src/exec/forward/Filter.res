type depthIndex = int

type state = {
  weights: ImmutableSparseMapType.t<depthIndex, Matrix.t>,
  bias: float,
}
