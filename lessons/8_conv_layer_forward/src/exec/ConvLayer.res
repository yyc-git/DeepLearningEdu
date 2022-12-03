type filterIndex = int

type depthIndex = int

type state = {
  inputWidth: int,
  inputHeight: int,
  depthNumber: int,
  filterWidth: int,
  filterHeight: int,
  filterNumber: int,
  filterStates: ImmutableSparseMapType.t<filterIndex, Filter.state>,
  zeroPadding: int,
  stride: int,
  outputWidth: int,
  outputHeight: int,
  // outputMap: array<Matrix.t>,
  // nets: array<Matrix.t>,
  leraningRate: float,
}

let create = (
  inputWidth,
  inputHeight,
  depthNumber,
  filterWidth,
  filterHeight,
  filterNumber,
  zeroPadding,
  stride,
  leraningRate,
) => {
  let outputWidth = LayerUtils.computeOutputSize(inputWidth, filterWidth, zeroPadding, stride)
  let outputHeight = LayerUtils.computeOutputSize(inputHeight, filterHeight, zeroPadding, stride)

  {
    inputWidth: inputWidth,
    inputHeight: inputHeight,
    depthNumber: depthNumber,
    filterWidth: filterWidth,
    filterHeight: filterHeight,
    filterNumber: filterNumber,
    zeroPadding: zeroPadding,
    stride: stride,
    outputWidth: outputWidth,
    outputHeight: outputHeight,
    filterStates: ArraySt.range(0, filterNumber - 1)->ArraySt.reduceOneParam(
      (. map, filterIndex) => {
        map->ImmutableSparseMap.set(
          filterIndex,
          Filter.create(filterWidth, filterHeight, depthNumber),
        )
      },
      ImmutableSparseMap.createEmpty(),
    ),
    leraningRate: leraningRate,
  }
}

let _padding = (matrixMap, zeroPadding) => {
  switch zeroPadding {
  | 0 => matrixMap
  | zeroPadding =>
    let (width, height, depth) = NP.getMatrixMapSize(matrixMap)

    // let paddingMatrix = NP.zeroMatrixMap(depth, row + 2 * zeroPadding, col + 2 * zeroPadding)
    let paddingMatrixMap = NP.zeroMatrixMap(
      depth,
      height + 2 * zeroPadding,
      width + 2 * zeroPadding,
    )

    paddingMatrixMap->ImmutableSparseMap.mapi((. paddingMatrix, i) => {
      NP.fillMatrix(
        zeroPadding,
        zeroPadding,
        paddingMatrix,
        matrixMap->ImmutableSparseMap.getExn(i),
      )
    })
  }
}

let _crossCorrelation2D = (input, weight, (outputWidth, outputHeight), stride, bias) => {
  let (filterWidth, filterHeight) = (Matrix.getColCount(weight), Matrix.getRowCount(weight))

  let outputRow = outputHeight
  let outputCol = outputWidth

  NP.zeroMatrix(outputRow, outputCol)->NP.reduceMatrix((output, _, rowIndex, colIndex) => {
    output->MatrixUtils.setValue(
      NP.dot(
        weight,
        LayerUtils.getConvolutionRegion2D(
          input,
          rowIndex,
          colIndex,
          filterWidth,
          filterHeight,
          stride,
        ),
      )->NP.sum +. bias,
      rowIndex,
      colIndex,
    )
  }, Matrix.create(outputRow, outputCol, []))
}

let _crossCorrelation3D = (inputs, weights, (outputWidth, outputHeight), stride, bias) => {
  let (filterWidth, filterHeight, _) = NP.getMatrixMapSize(weights)

  //  let   NP.zeroMatrix( state.outputHeight, state.outputWidth)

  // let outputCol = Matrix.getColCount(output)
  let outputRow = outputHeight
  let outputCol = outputWidth

  // (filterWidth, filterHeight, outputHeight, outputWidth)->Log.printForDebug->ignore

  NP.zeroMatrix(outputRow, outputCol)->NP.reduceMatrix((output, _, rowIndex, colIndex) => {
    output->MatrixUtils.setValue(
      LayerUtils.getConvolutionRegion3D(
        inputs,
        rowIndex,
        colIndex,
        filterWidth,
        filterHeight,
        stride,
      )
      ->ImmutableSparseMap.mapi((. convolutionRegion, i) => {
        NP.dot(weights->ImmutableSparseMap.getExn(i), convolutionRegion)
      })
      ->NP.sumMatrixMap +. bias,
      rowIndex,
      colIndex,
    )
  }, Matrix.create(outputRow, outputCol, []))
}

let _elementWiseOp = (matrix, opFunc) => {
  matrix->Matrix.map(opFunc)
}

// let forward = (state, inputs: ImmutableSparseMapType.t<depthIndex, Matrix.t>) => {
// let forward = (state, inputs: ImmutableSparseMapType.t<depthIndex, Matrix.t>) => {
let forward = (activate, state, inputs: ImmutableSparseMapType.t<depthIndex, Matrix.t>) => {
  let paddedInputs = _padding(inputs, state.zeroPadding)

  // let outputMap = NP.zeroMatrixMap(state.filterNumber, state.outputHeight, state.outputWidth)

  let (nets, outputMap) =
    ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParam((. (nets, outputMap), i) => {
      let filterState = state.filterStates->ImmutableSparseMap.getExn(i)

      let net = _crossCorrelation3D(
        paddedInputs,
        Filter.getWeights(filterState),
        (state.outputWidth, state.outputHeight),
        state.stride,
        Filter.getBias(filterState),
      )

      // net->Log.printForDebug->ignore

      let output = net->_elementWiseOp(
        // ReluActivator_answer.forward
        activate,
      )

      (nets->ImmutableSparseMap.set(i, net), outputMap->ImmutableSparseMap.set(i, output))
    }, (ImmutableSparseMap.createEmpty(), ImmutableSparseMap.createEmpty()))

  (paddedInputs, (nets, outputMap))
}

module Test = {
  let init = () => {
    let inputs =
      [
        [
          [0., 1., 1., 0., 2.],
          [2., 2., 2., 2., 1.],
          [1., 0., 0., 2., 0.],
          [0., 1., 1., 0., 0.],
          [1., 2., 0., 0., 2.],
        ],
        [
          [1., 0., 2., 2., 0.],
          [0., 0., 0., 2., 0.],
          [1., 2., 1., 2., 1.],
          [1., 0., 0., 0., 0.],
          [1., 2., 1., 1., 1.],
        ],
        [
          [2., 1., 2., 0., 0.],
          [1., 0., 0., 1., 0.],
          [0., 2., 1., 0., 1.],
          [0., 1., 2., 2., 2.],
          [2., 1., 0., 0., 1.],
        ],
      ]->NP.createMatrixMapByDataArr

    // inputWidth,
    // inputHeight,
    // depthNumber,
    // filterWidth,
    // filterHeight,
    // filterNumber,
    // zeroPadding,
    // stride,
    // leraningRate,
    let state = create(5, 5, 3, 3, 3, 2, 1, 2, 0.001)

    let state = {
      ...state,
      filterStates: ImmutableSparseMap.createEmpty()
      ->ImmutableSparseMap.set(
        0,
        (
          {
            ...state.filterStates->ImmutableSparseMap.getExn(0),
            weights: [
              [[-1., 1., 0.], [0., 1., 0.], [0., 1., 1.]],
              [[-1., -1., 0.], [0., 0., 0.], [0., -1., 0.]],
              [[0., 0., -1.], [0., 1., 0.], [1., -1., -1.]],
            ]->NP.createMatrixMapByDataArr,
            bias: 1.,
          }: Filter.state
        ),
      )
      ->ImmutableSparseMap.set(
        1,
        {
          ...state.filterStates->ImmutableSparseMap.getExn(1),
          weights: [
            [[1., 1., -1.], [-1., -1., 1.], [0., -1., 1.]],
            [[0., 1., 0.], [-1., 0., -1.], [-1., 1., 0.]],
            [[-1., 0., 0.], [-1., 0., 1.], [-1., 0., 0.]],
          ]->NP.createMatrixMapByDataArr,
          bias: 0.,
        },
      ),
    }

    (inputs, state)
  }

  let test = ((inputs, state)) => {
    let (paddedInputs, (nets, outputMap)) = forward(ReluActivator_answer.forward, state, inputs)

    ("f:", nets)->Log.printForDebug->ignore
  }
}

Test.init()->Test.test
