type state = {
  wMatrixBetweenLayer1Layer2: Matrix.t,
  wMatrixBetweenLayer2Layer3: Matrix.t,
}

type feature = {
  weight: float,
  height: float,
}

let _createWMatrix = (getValueFunc, firstLayerNodeCount, secondLayerNodeCount) => {
  let row = secondLayerNodeCount
  let col = firstLayerNodeCount + 1

  Matrix.create(row, col, ArraySt.range(0, row * col - 1)->ArraySt.map(_ => getValueFunc()))
}

let createState = (layer1NodeCount, layer2NodeCount, layer3NodeCount): state => {
  wMatrixBetweenLayer1Layer2: _createWMatrix(() => 0.1, layer1NodeCount, layer2NodeCount),
  wMatrixBetweenLayer2Layer3: _createWMatrix(() => 0.1, layer2NodeCount, layer3NodeCount),
}

let _activateFunc = x => {
  1. /. (1. +. Js.Math.exp(-.x))
}

let forward = (state: state, feature: feature) => {
  let inputVector = Vector.create([feature.height, feature.weight, 1.0])

  let layer2OutputVector =
    Vector.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector)->Vector.map(_activateFunc)

  let layer3OutputVector = Vector.transformMatrix(
    state.wMatrixBetweenLayer2Layer3,
    /* ! 注意：此处push 1.0 */
    layer2OutputVector->Vector.push(1.0),
  )->Vector.map(_activateFunc)

  (layer2OutputVector, layer3OutputVector)
}

let state = createState(2, 2, 1)

let feature = {
  weight: 50.,
  height: 150.,
}

forward(state, feature)->Js.log
