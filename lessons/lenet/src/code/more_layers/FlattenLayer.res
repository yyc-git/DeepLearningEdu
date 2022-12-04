type state = {
  inputWidth: int,
  inputHeight: int,
  depthNumber: int,
}

let _flattenMatrix = matrix => {
  //   matrix->NP.reduceMatrix((vectorData, value, _, _) => {
  //     vectorData->ArraySt.push(value)
  //   }, [])
  matrix->Matrix.getData
}

let _flatten = (inputMap): Vector.t => {
  inputMap->ImmutableSparseMap.reducei((. vectorData, output, _) => {
    Js.Array.concat(_flattenMatrix(output), vectorData)
  }, [])->Vector.create
}

let forward: LayerAbstractType.forward<MatrixMap.t, Vector.t> = (state, _, inputMap, _) => {
  let output = inputMap->_flatten

  (state, output->Some, output)
}

let bpDelta: LayerAbstractType.bpDelta<MatrixMap.t, Vector.t> = (
  _,
  _,
  layerDelta: Vector.t,
  state,
) => {
  let {depthNumber, inputWidth, inputHeight} = state->Obj.magic

  //   DO use requireCheck: length should equal
  layerDelta->Vector.length == depthNumber * inputWidth * inputHeight
    ? ()
    : Exception.throwErr("error")

  let oneDeltaMapCount = inputWidth * inputHeight
  let (totalDepthDelta, depthDelta) =
    layerDelta->Vector.reducei((. (totalDepthDelta, depthDelta), value, index) => {
      index !== 0 && mod(index, oneDeltaMapCount) == 0
        ? {
            (totalDepthDelta->ArraySt.push(depthDelta), [value])
          }
        : {
            (totalDepthDelta, depthDelta->ArraySt.push(value))
          }
    }, ([], []))
  let totalDepthDelta = totalDepthDelta->ArraySt.push(depthDelta)

  totalDepthDelta
  ->ArraySt.map(depthDelta => {
    Matrix.create(inputHeight, inputWidth, depthDelta)
  })
  ->ImmutableSparseMap.createFromArr
  ->Some
}

let backward: LayerAbstractType.backward<
  MatrixMap.t,
  Vector.t,
  LayerAbstractType.none,
  LayerAbstractType.none,
> = (_, previousLayerData, layerDelta, state) => {
  (bpDelta(None, previousLayerData, layerDelta, state), None)
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

let create = (~inputWidth, ~inputHeight, ~depthNumber=1, ()) => {
  {
    inputWidth: inputWidth,
    inputHeight: inputHeight,
    depthNumber: depthNumber,
  }
}

let createLayerData: LayerAbstractType.createLayerData = (state, activatorData) => {
  {
    layerName: #flatten,
    state: state,
    forward: forward->Obj.magic,
    backward: backward->Obj.magic,
    // TODO refactor(all): change update, ... to None(e.g. update: None)
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
