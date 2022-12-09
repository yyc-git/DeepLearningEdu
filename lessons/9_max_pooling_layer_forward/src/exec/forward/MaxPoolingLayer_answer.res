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

let create = (inputWidth, inputHeight, depthNumber, filterWidth, filterHeight, stride) => {
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

let forward = (state, inputs: ImmutableSparseMapType.t<depthIndex, Matrix.t>) => {
  let outputRow = state.outputHeight
  let outputCol = state.outputWidth

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
}

module Test = {
  let init = () => {
    let inputs =
      [
        [[0., 1., 1., 0.], [2., 3., 2., 2.], [1., 0., 0., 2.], [0., 1., 1., 0.]],
        [[1., 0., 2., 2.], [0., 5., 0., 2.], [1., 2., 1., 2.], [1., 0., 0., 0.]],
      ]->NP.createMatrixMapByDataArr

    // inputWidth,
    // inputHeight,
    // depthNumber,
    // filterWidth,
    // filterHeight,
    // stride,
    let state = create(4, 4, 2, 2, 2, 2)

    (inputs, state)
  }

  let test = ((inputs, state)) => {
    let outputMap = forward(state, inputs)

    ("f:", outputMap)->Log.printForDebug->ignore
  }
}

Test.init()->Test.test
