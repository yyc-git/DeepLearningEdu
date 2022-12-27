type depthIndex = int

type state = {
  weights: ImmutableSparseMapType.t<depthIndex, Matrix.t>,
  bias: float,
  weightGradients: ImmutableSparseMapType.t<depthIndex, Matrix.t>,
  biasGradient: float,
}

let create = (width, height, depth) => {
  weights: NP.createMatrixMap(() => (Js.Math.random() *. 2. -. 1.) *. 1e-4, depth, height, width),
  bias: 0.,
  weightGradients: NP.zeroMatrixMap(depth, height, width),
  biasGradient: 0.,

}

let getWeights = state => {
  state.weights
}

let getBias = state => {
  state.bias
}
