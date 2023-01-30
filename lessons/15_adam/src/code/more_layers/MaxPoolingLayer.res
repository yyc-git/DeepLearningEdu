type depthIndex = int

type state = {
  inputWidth: int,
  inputHeight: int,
  depthNumber: int,
  filterWidth: int,
  filterHeight: int,
  stride: int,
  outputWidth: int,
  outputHeight: int,
}

let create = (
  ~inputWidth,
  ~inputHeight,
  ~filterWidth,
  ~filterHeight,
  ~stride=1,
  ~depthNumber=1,
  (),
) => {
  let outputWidth = LayerUtils.computeOutputSize(inputWidth, filterWidth, 0, stride)
  let outputHeight = LayerUtils.computeOutputSize(inputHeight, filterHeight, 0, stride)

  {
    inputWidth: inputWidth,
    inputHeight: inputHeight,
    depthNumber: depthNumber,
    filterWidth: filterWidth,
    filterHeight: filterHeight,
    stride: stride,
    outputWidth: outputWidth,
    outputHeight: outputHeight,
  }
}

let forward: LayerAbstractType.forward<MatrixMap.t, MatrixMap.t> = (
  state,
  _,
  inputs: MatrixMap.t,
  _
) => {
  let state = state->Obj.magic

  let outputRow = state.outputHeight
  let outputCol = state.outputWidth

  let output =
    ArraySt.range(0, state.depthNumber - 1)->ArraySt.reduceOneParam((. outputMap, depthIndex) => {
      let input = inputs->ImmutableSparseMap.getExn(depthIndex)

      let output = ArraySt.range(0, outputRow - 1)->ArraySt.reduceOneParam((. output, rowIndex) => {
        ArraySt.range(0, outputCol - 1)->ArraySt.reduceOneParam((. output, colIndex) => {
          output->MatrixUtils.setValue(
            LayerUtils.getConvolutionRegion2D(
              input,
              rowIndex,
              colIndex,
              state.filterWidth,
              state.filterHeight,
              state.stride,
            )->NP.max,
            rowIndex,
            colIndex,
          )
        }, output)
      }, Matrix.create(outputRow, outputCol, []))

      outputMap->ImmutableSparseMap.set(depthIndex, output)
    }, ImmutableSparseMap.createEmpty())

  (state, output->Some, output)
}

let bpDelta: LayerAbstractType.bpDelta<MatrixMap.t, MatrixMap.t> = (
  _,
  (previousLayerOutput, _),
  nextLayerDelta: MatrixMap.t,
  state,
) => {
  let state = state->Obj.magic
  let previousLayerOutput = previousLayerOutput->OptionSt.getExn

  let outputRow = state.outputHeight
  let outputCol = state.outputWidth

  ArraySt.range(0, state.depthNumber - 1)
  ->ArraySt.reduceOneParam((. currentLayerDeltaMap, depthIndex) => {
    let input = previousLayerOutput->ImmutableSparseMap.getExn(depthIndex)

    let currentLayerDelta = currentLayerDeltaMap->ImmutableSparseMap.getExn(depthIndex)

    let currentLayerDelta =
      ArraySt.range(0, outputRow - 1)->ArraySt.reduceOneParam((. currentLayerDelta, rowIndex) => {
        ArraySt.range(0, outputCol - 1)->ArraySt.reduceOneParam(
          (. currentLayerDelta, colIndex) => {
            let (_, maxValueRowIndex, maxValueColIndex) =
              LayerUtils.getConvolutionRegion2D(
                input,
                rowIndex,
                colIndex,
                state.filterWidth,
                state.filterHeight,
                state.stride,
              )->NP.getMaxIndex

            currentLayerDelta->MatrixUtils.setValue(
              nextLayerDelta->NP.getMatrixMapValue(depthIndex, rowIndex, colIndex),
              rowIndex * state.stride + maxValueRowIndex,
              colIndex * state.stride + maxValueColIndex,
            )
          },
          currentLayerDelta,
        )
      }, currentLayerDelta)

    currentLayerDeltaMap->ImmutableSparseMap.set(depthIndex, currentLayerDelta)
  }, LayerUtils.createPreviousLayerDeltaMap((
    state.depthNumber,
    state.inputWidth,
    state.inputHeight,
  )))
  ->Some
}

let backward: LayerAbstractType.backward<
  MatrixMap.t,
  MatrixMap.t,
  LayerAbstractType.none,
  LayerAbstractType.none,
> = (_, previousLayerData, nextLayerDelta, state) => {
  (bpDelta(None, previousLayerData, nextLayerDelta, state), None)
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

// TODO refactor: duplicate
let createLayerData: LayerAbstractType.createLayerData = (state, activatorData) => {
  {
    layerName: #maxPooling,
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
