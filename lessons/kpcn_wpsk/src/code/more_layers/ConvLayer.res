open ActivatorType

open ConvLayerStateType

// type createConfig = {
//   inputWidth: int,
//   inputHeight: int,
//   depthNumber: int,
//   filterWidth: int,
//   filterHeight: int,
//   filterNumber: int,
//   zeroPadding: int,
//   stride: int,
//   leraningRate: float,
// }

let create = (
  ~inputWidth,
  ~inputHeight,
  ~filterWidth,
  ~filterHeight,
  ~filterNumber=1,
  ~zeroPadding=1,
  ~stride=1,
  ~depthNumber=1,
  // ~initValueMethod=NormalHe.normal,
  ~initValueMethod=Random.random,
  ~randomFunc=Js.Math.random,
  (),
) => {
  let outputWidth = LayerUtils.computeOutputSize(inputWidth, filterWidth, zeroPadding, stride)
  let outputHeight = LayerUtils.computeOutputSize(inputHeight, filterHeight, zeroPadding, stride)

  let fanIn = filterWidth * filterHeight * depthNumber
  let fanOut = filterWidth * filterHeight * filterNumber

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
          Filter.create(
            (initValueMethod, randomFunc, fanIn, fanOut),
            filterWidth,
            filterHeight,
            depthNumber,
          ),
        )
      },
      ImmutableSparseMap.createEmpty(),
    ),
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

// let padding = (matrixMap, {zeroPadding}) => {
//   _padding(matrixMap, zeroPadding)
// }

let _crossCorrelation2D = (data, filter, (outputWidth, outputHeight), stride, bias) => {
  let (filterWidth, filterHeight) = (Matrix.getColCount(filter), Matrix.getRowCount(filter))

  let outputRow = outputHeight
  let outputCol = outputWidth

  NP.zeroMatrix(outputRow, outputCol)->NP.reduceMatrix((output, _, rowIndex, colIndex) => {
    output->MatrixUtils.setValue(
      NP.dot(
        filter,
        LayerUtils.getConvolutionRegion2D(
          data,
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

let _crossCorrelation3D = (data, filters, (outputWidth, outputHeight), stride, bias) => {
  let (filterWidth, filterHeight, _) = NP.getMatrixMapSize(filters)

  //  let   NP.zeroMatrix( state.outputHeight, state.outputWidth)

  // let outputCol = Matrix.getColCount(output)
  let outputRow = outputHeight
  let outputCol = outputWidth

  // (filterWidth, filterHeight, outputHeight, outputWidth)->Log.printForDebug->ignore

  NP.zeroMatrix(outputRow, outputCol)->NP.reduceMatrix((output, _, rowIndex, colIndex) => {
    output->MatrixUtils.setValue(
      LayerUtils.getConvolutionRegion3D(data, rowIndex, colIndex, filterWidth, filterHeight, stride)
      ->ImmutableSparseMap.mapi((. convolutionRegion, i) => {
        NP.dot(filters->ImmutableSparseMap.getExn(i), convolutionRegion)
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

let forward: LayerAbstractType.forward<MatrixMap.t, MatrixMap.t> = (
  state,
  activatorData,
  inputs,
  _,
) => {
  let state = state->Obj.magic
  let {forwardMatrix} = activatorData->OptionSt.getExn

  let paddedInputs = inputs
  // ->_convertInputVectorToInputs(state.inputWidth, state.inputHeight)
  ->_padding(state.zeroPadding)

  // let paddedInputs = padding(inputs, state.zeroPadding)

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

      // let output = net-> _elementWiseOp(forward)
      let output = net->forwardMatrix

      (nets->ImmutableSparseMap.set(i, net), outputMap->ImmutableSparseMap.set(i, output))
    }, (ImmutableSparseMap.createEmpty(), ImmutableSparseMap.createEmpty()))

  // "forward"->Log.printForDebug->ignore
  // (paddedInputs, (nets, outputMap))
  // (nets->Some, outputMap->Log.printForDebug)
  (state, nets->Some, outputMap)
}

// let forwardGetOutput = (activatorData, inputs, state) => {
//   let (paddedInputs, (nets, outputMap)) = forward(activatorData, inputs, state)

//   outputMap
// }

let _expandDeltaMapByStride = (
  deltaMap,
  {inputWidth, inputHeight, filterWidth, filterHeight, stride, zeroPadding},
) => {
  let (_, _, deltaMapDepth) = NP.getMatrixMapSize(deltaMap)

  let expandDeltaWidth = LayerUtils.computeOutputSize(inputWidth, filterWidth, zeroPadding, 1)
  let expandDeltaHeight = LayerUtils.computeOutputSize(inputHeight, filterHeight, zeroPadding, 1)

  // "ssss"->Log.printForDebug->ignore

  // let expandDelta = NP.zeroMatrix(expandDeltaHeight, expandDeltaWidth)
  let expandDeltaMap = NP.zeroMatrixMap(deltaMapDepth, expandDeltaHeight, expandDeltaWidth)

  deltaMap->ImmutableSparseMap.mapi((. delta, i) => {
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

//     let delta = _crossCorrelation3D(expandDelta, invertedWeights, LayerUtils. createPreviousLayerDeltaMap(state), 1, 0)

//     Matrix.add(totalDelta, delta)
//   }, LayerUtils. createPreviousLayerDeltaMap(state))
// }

// TODO rename padExpandDeltaMap to padExpandDelta,   previousLayerNets to previousLayerNet
let _compute = ({backward}, padExpandDeltaMap, state, previousLayerNets) => {
  // let previousLayerNets = inputs->NP.mapMatrixMap(Matrix.map(_, invert))

  let currentLayerDeltaMap =
    ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParam(
      (. currentLayerDeltaMap, filterIndex) => {
        let padExpandDelta = padExpandDeltaMap->ImmutableSparseMap.getExn(filterIndex)
        let filterState = state.filterStates->ImmutableSparseMap.getExn(filterIndex)

        let flippedWeights =
          Filter.getWeights(filterState)->ImmutableSparseMap.map((. weight) => weight->NP.rotate180)

        // "aaa"->Log.printForDebug->ignore
        // let currentLayerDeltaMap = NP.addMatrixMap(
        NP.addMatrixMap(
          currentLayerDeltaMap,
          LayerUtils.createPreviousLayerDeltaMap((
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

        // currentLayerDeltaMap->ImmutableSparseMap.mapi((. currentLayerDelta, depthIndex) => {
        //   NP.dot(
        //     currentLayerDelta,
        //     previousLayerNets
        //     // ->NP.mapMatrixMap(Matrix.map(_, ReluActivator.backward))
        //     ->NP.mapMatrixMap(Matrix.map(_, backward))
        //     ->ImmutableSparseMap.getExn(depthIndex),
        //   )
        // })
      },
      LayerUtils.createPreviousLayerDeltaMap((
        state.depthNumber,
        state.inputWidth,
        state.inputHeight,
      )),
    )

  currentLayerDeltaMap->ImmutableSparseMap.mapi((. currentLayerDelta, depthIndex) => {
    NP.dot(
      currentLayerDelta,
      previousLayerNets
      // ->NP.mapMatrixMap(Matrix.map(_, ReluActivator.backward))
      ->NP.mapMatrixMap(Matrix.map(_, backward))
      ->ImmutableSparseMap.getExn(depthIndex),
    )
  })
}

// let _multiplyActivatorDeriv = (delta, input) => {
//   NP.dot(delta, input->_elementWiseOp(ReluActivator.backward))
// }

let bpDelta: LayerAbstractType.bpDelta<MatrixMap.t, MatrixMap.t> = (
  previousLayerActivatorData,
  (_, previousLayerNet),
  nextLayerExpandDelta: ImmutableSparseMapType.t<filterIndex, Matrix.t>,
  state,
) => {
  let state = state->Obj.magic

  nextLayerExpandDelta
  ->_paddingDeltaMap(state)
  ->_compute(
    previousLayerActivatorData->OptionSt.getExn,
    _,
    state,
    previousLayerNet->OptionSt.getExn,
  )
  ->Some
}

let computeGradient: LayerAbstractType.computeGradient<
  MatrixMap.t,
  MatrixMap.t,
  ImmutableSparseMapType.t<filterIndex, MatrixMap.t>,
  ImmutableSparseMapType.t<filterIndex, float>,
> = (inputs, expandDeltaMap, state) => {
  let state = state->OptionSt.getExn->Obj.magic

  let paddedInputs = inputs->_padding(state.zeroPadding)

  ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParam(
    (. (weightGradientData, biasGradientData), filterIndex) => {
      let filterState: FilterStateType.state =
        state.filterStates->ImmutableSparseMap.getExn(filterIndex)

      let expandDelta = expandDeltaMap->ImmutableSparseMap.getExn(filterIndex)

      let weightGradients = NP.zeroMatrixMap(
        state.depthNumber,
        state.filterHeight,
        state.filterWidth,
      )->ImmutableSparseMap.mapi((. weightGradient, depthIndex) => {
        _crossCorrelation2D(
          paddedInputs->ImmutableSparseMap.getExn(depthIndex),
          expandDelta,
          (Matrix.getColCount(weightGradient), Matrix.getRowCount(weightGradient)),
          1,
          0.,
        )
      })

      let biasGradient = NP.sum(expandDelta)

      (
        weightGradientData->ImmutableSparseMap.set(filterIndex, weightGradients),
        biasGradientData->ImmutableSparseMap.set(filterIndex, biasGradient),
      )
    },
    (ImmutableSparseMap.createEmpty(), ImmutableSparseMap.createEmpty()),
  )
}

let createGradientDataSum: LayerAbstractType.createGradientDataSum<
  ImmutableSparseMapType.t<filterIndex, MatrixMap.t>,
  ImmutableSparseMapType.t<filterIndex, float>,
> = state => {
  let {filterNumber, filterWidth, filterHeight, depthNumber}: state = state->Obj.magic

  (
    ArraySt.range(0, filterNumber - 1)
    ->ArraySt.map(_ => NP.zeroMatrixMap(depthNumber, filterHeight, filterWidth))
    ->ImmutableSparseMap.createFromArr,
    ArraySt.range(0, filterNumber - 1)->ArraySt.map(_ => 0.)->ImmutableSparseMap.createFromArr,
  )->Some
}

let getPreviousLayerNet = ({invert}, previousLayerOutput) => {
  previousLayerOutput->NP.mapMatrixMap(Matrix.map(_, invert))
}

let backward: LayerAbstractType.backward<
  MatrixMap.t,
  MatrixMap.t,
  ImmutableSparseMapType.t<filterIndex, MatrixMap.t>,
  ImmutableSparseMapType.t<filterIndex, float>,
> = (
  previousLayerActivatorData,
  // (previousLayerPaddedOutput, previousLayerNet),
  (previousLayerOutput, previousLayerNet),
  nextLayerDelta: ImmutableSparseMapType.t<filterIndex, Matrix.t>,
  state,
) => {
  // ("nextLayerDelta:", nextLayerDelta )->Log.printForDebug->ignore

  let nextLayerExpandDelta = _expandDeltaMapByStride(nextLayerDelta, state->Obj.magic)

  let currentLayerDeltaMap = bpDelta(
    previousLayerActivatorData,
    (None, previousLayerNet),
    nextLayerExpandDelta,
    state,
  )

  let gradientData = computeGradient(
    previousLayerOutput->OptionSt.getExn,
    nextLayerExpandDelta,
    state->Some,
  )

  (currentLayerDeltaMap, gradientData->Some)
}

let addToGradientDataSum: LayerAbstractType.addToGradientDataSum<
  ImmutableSparseMapType.t<filterIndex, MatrixMap.t>,
  ImmutableSparseMapType.t<filterIndex, float>,
> = (gradientDataSum, gradientData) => {
  let (weightGradientDataSum, biasGradientDataSum) = gradientDataSum->OptionSt.getExn
  let (weightGradientData, biasGradientData) = gradientData->OptionSt.getExn

  (
    weightGradientDataSum->ImmutableSparseMap.mapi((. oneFilterWeightGradientsSum, filterIndex) => {
      NP.addMatrixMap(
        oneFilterWeightGradientsSum,
        weightGradientData->ImmutableSparseMap.getExn(filterIndex),
      )
    }),
    biasGradientDataSum->ImmutableSparseMap.mapi((. oneFilterBiasGradientSum, filterIndex) => {
      oneFilterBiasGradientSum +. biasGradientData->ImmutableSparseMap.getExn(filterIndex)
    }),
  )->Some
}

let update: LayerAbstractType.update<
  ImmutableSparseMapType.t<filterIndex, MatrixMap.t>,
  ImmutableSparseMapType.t<filterIndex, float>,
> = (state, optimizerData, (miniBatchSize, gradientDataSum)) => {
  let state = state->Obj.magic
  let (weightGradientDataSum, biasGradientDataSum) = gradientDataSum->OptionSt.getExn

  ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParam((. state, filterIndex) => {
    let filterState = state.filterStates->ImmutableSparseMap.getExn(filterIndex)

    {
      ...state,
      filterStates: state.filterStates->ImmutableSparseMap.set(
        filterIndex,
        Filter.update(
          filterState,
          (
            miniBatchSize,
            weightGradientDataSum->ImmutableSparseMap.getExn(filterIndex),
            biasGradientDataSum->ImmutableSparseMap.getExn(filterIndex),
          ),
          optimizerData,
        ),
      ),
    }
  }, state)->Obj.magic
}

let createLayerData: LayerAbstractType.createLayerData = (state, activatorData) => {
  {
    layerName: #conv,
    state: state,
    forward: forward->Obj.magic,
    backward: backward->Obj.magic,
    update: update->Obj.magic,
    createGradientDataSum: createGradientDataSum->Obj.magic,
    addToGradientDataSum: addToGradientDataSum->Obj.magic,
    activatorData: activatorData,
    getWeight: state => {
      let state = state->Obj.magic

      state.filterStates
      ->ImmutableSparseMap.map((. filterState) => {
        Filter.getWeights(filterState)
      })
      ->Obj.magic
      ->Some
    },
    getBias: state => {
      let state = state->Obj.magic

      state.filterStates
      ->ImmutableSparseMap.map((. filterState) => {
        Filter.getBias(filterState)
      })
      ->Obj.magic
      ->Some
    },
  }
}
