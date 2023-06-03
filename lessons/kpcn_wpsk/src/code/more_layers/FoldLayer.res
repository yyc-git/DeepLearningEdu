type state = {
  outputWidth: int,
  outputHeight: int,
}

let forward: LayerAbstractType.forward<Vector.t, MatrixMap.t> = (state, _, input, _) => {
  let {outputWidth, outputHeight} = state->Obj.magic

  let output =
    ImmutableSparseMap.createEmpty()->ImmutableSparseMap.set(
      0,
      Matrix.create(outputHeight, outputWidth, input->Vector.toArray),
    )

  (state, output->Some, output)
}

let bpDelta: LayerAbstractType.bpDelta<Vector.t, MatrixMap.t> = (_, _, _, _) => {
  None
}

let backward: LayerAbstractType.backward<
  MatrixMap.t,
  Vector.t,
  LayerAbstractType.none,
  LayerAbstractType.none,
> = (_, previousLayerData, nextLayerDelta, state) => {
  (None, None)
}

let createGradientDataSum: LayerAbstractType.createGradientDataSum<
  LayerAbstractType.none,
  LayerAbstractType.none,
> = state => {
  None
}

let addToGradientDataSum: LayerAbstractType.addToGradientDataSum<
  LayerAbstractType.none,
  LayerAbstractType.none,
> = (_, _) => {
  None
}

let update: LayerAbstractType.update<LayerAbstractType.none, LayerAbstractType.none> = (
  state,
  _,
  _,
) => {
  state
}

let create = (outputWidth, outputHeight) => {
  {
    outputWidth: outputWidth,
    outputHeight: outputHeight,
  }
}

let createLayerData: LayerAbstractType.createLayerData = (state, activatorData) => {
  {
    layerName: #fold,
    state: state,
    forward: forward->Obj.magic,
    backward: backward->Obj.magic,
    update: update->Obj.magic,
    createGradientDataSum: createGradientDataSum->Obj.magic,
    addToGradientDataSum: addToGradientDataSum->Obj.magic,
    activatorData: activatorData,
    getWeight: state => {
      None
    },
    getBias: state => {
      None
    },
  }
}
