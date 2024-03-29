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
        // ReluActivator.forward
        activate,
      )

      (nets->ImmutableSparseMap.set(i, net), outputMap->ImmutableSparseMap.set(i, output))
    }, (ImmutableSparseMap.createEmpty(), ImmutableSparseMap.createEmpty()))

  (paddedInputs, (nets, outputMap))
}

let _expandDeltaMapByStride = (
  nextLayerDeltaMap,
  {inputWidth, inputHeight, filterWidth, filterHeight, stride, zeroPadding},
) => {
  let (_, _, deltaMapDepth) = NP.getMatrixMapSize(nextLayerDeltaMap)

  let expandDeltaWidth = LayerUtils.computeOutputSize(inputWidth, filterWidth, zeroPadding, 1)
  let expandDeltaHeight = LayerUtils.computeOutputSize(inputHeight, filterHeight, zeroPadding, 1)

  // "ssss"->Log.printForDebug->ignore

  // let expandDelta = NP.zeroMatrix(expandDeltaHeight, expandDeltaWidth)
  let expandDeltaMap = NP.zeroMatrixMap(deltaMapDepth, expandDeltaHeight, expandDeltaWidth)

  nextLayerDeltaMap->ImmutableSparseMap.mapi((. delta, i) => {
    let expandDelta = expandDeltaMap->ImmutableSparseMap.getExn(i)

    delta->NP.reduceMatrix((expandDelta, value, rowIndex, colIndex) => {
      expandDelta->MatrixUtils.setValue(value, rowIndex * stride, colIndex * stride)
    }, expandDelta)
  })
}

let _paddingDeltaMap = (expandDeltaMap, {inputWidth, filterWidth}) => {
  let (expandDeltaWidth, _, _) = NP.getMatrixMapSize(expandDeltaMap)

  _padding(expandDeltaMap, (inputWidth + filterWidth - 1 - expandDeltaWidth) / 2)
}

// let _computeDeltaWeightConv = (expandDelta, state) => {
//   ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParami((. totalDelta, _, i) => {
//     let filterState = ImmutableSparseMap.getExn(i)

//     let invertedWeights = NP.rotate180(Filter.getWeights(filterState))

//     let delta = _crossCorrelation3D(expandDelta, invertedWeights, LayerUtils. createCurrentLayerDeltaMap(state), 1, 0)

//     Matrix.add(totalDelta, delta)
//   }, LayerUtils. createCurrentLayerDeltaMap(state))
// }

let _compute = (padExpandDeltaMap, state, inputNets) => {
  let lastLayerNets = inputNets->NP.mapMatrixMap(Matrix.map(_, ReluActivator.invert))

  let currentLayerDeltaMap =
    ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParam(
      (. currentLayerDeltaMap, filterIndex) => {
        let padExpandDelta = padExpandDeltaMap->ImmutableSparseMap.getExn(filterIndex)
        let filterState = state.filterStates->ImmutableSparseMap.getExn(filterIndex)

        let flippedWeights =
          Filter.getWeights(filterState)->ImmutableSparseMap.map((. weight) => weight->NP.rotate180)

        NP.addMatrixMap(
          currentLayerDeltaMap,
          LayerUtils.createCurrentLayerDeltaMap((
            state.depthNumber,
            state.inputWidth,
            state.inputHeight,
          ))->ImmutableSparseMap.mapi((. delta, depthIndex) => {
            _crossCorrelation2D(
              padExpandDelta,
              flippedWeights->ImmutableSparseMap.getExn(depthIndex),
              (Matrix.getColCount(delta), Matrix.getRowCount(delta)),
              1,
              0.,
            )
          }),
        )
      },
      LayerUtils.createCurrentLayerDeltaMap((
        state.depthNumber,
        state.inputWidth,
        state.inputHeight,
      )),
    )

  currentLayerDeltaMap->ImmutableSparseMap.mapi((. lastLayerDelta, depthIndex) => {
    NP.dot(
      lastLayerDelta,
      lastLayerNets
      ->NP.mapMatrixMap(Matrix.map(_, ReluActivator.backward))
      ->ImmutableSparseMap.getExn(depthIndex),
    )
  })
}

// let _multiplyActivatorDeriv = (delta, input) => {
//   NP.dot(delta, input->_elementWiseOp(ReluActivator.backward))
// }

// 计算传递到上一层的delta map
let bpDeltaMap = (
  state,
  inputNets: ImmutableSparseMapType.t<depthIndex, Matrix.t>,
  nextLayerDeltaMap: ImmutableSparseMapType.t<filterIndex, Matrix.t>,
) => {
  _expandDeltaMapByStride(nextLayerDeltaMap, state)
  // ->Log.printForDebug
  ->_paddingDeltaMap(state)
  // ->Log.printForDebug
  // ->_computeDeltaWeightConv(state)
  // ->_multiplyActivatorDeriv(nets)
  ->_compute(state, inputNets)
}

// let computeGradient = (state, outputMap, nextLayerDeltaMap) => {
let computeGradient = (state, paddedInputs, nextLayerDeltaMap) => {
  // ("bbb", nextLayerDeltaMap)->Log.printForDebug->ignore
  let expandDeltaMap = _expandDeltaMapByStride(nextLayerDeltaMap, state)

  ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParam((. state, filterIndex) => {
    let filterState: Filter.state = state.filterStates->ImmutableSparseMap.getExn(filterIndex)

    let expandDelta = expandDeltaMap->ImmutableSparseMap.getExn(filterIndex)

    let weightGradients = filterState.weightGradients->ImmutableSparseMap.mapi((.
      weightGradient,
      depthIndex,
    ) => {
      _crossCorrelation2D(
        paddedInputs->ImmutableSparseMap.getExn(depthIndex),
        expandDelta,
        (Matrix.getColCount(weightGradient), Matrix.getRowCount(weightGradient)),
        1,
        0.,
      )
    })

    let biasGradient = NP.sum(expandDelta)

    {
      ...state,
      filterStates: state.filterStates->ImmutableSparseMap.set(
        filterIndex,
        {
          ...filterState,
          weightGradients: weightGradients,
          biasGradient: biasGradient,
        },
      ),
    }
  }, state)
}

let backward = (state, (inputs, inputNets), nextLayerDeltaMap) => {
  // TODO remove forward
  let (paddedInputs, _) = forward(ReluActivator.forward, state, inputs)

  // let updatedDelta = bpDeltaMap(state, nets, nextLayerDeltaMap)
  let currentLayerDeltaMap = bpDeltaMap(state, inputNets, nextLayerDeltaMap)

  let state = computeGradient(state, paddedInputs, currentLayerDeltaMap)

  state
}

let update = state => {
  ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParam((. state, i) => {
    let filterState = state.filterStates->ImmutableSparseMap.getExn(i)

    {
      ...state,
      filterStates: state.filterStates->ImmutableSparseMap.set(
        i,
        Filter.update(filterState, state.leraningRate),
      ),
    }
  }, state)
}
