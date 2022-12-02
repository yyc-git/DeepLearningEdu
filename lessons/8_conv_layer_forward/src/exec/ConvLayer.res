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

//     let delta = _crossCorrelation3D(expandDelta, invertedWeights, LayerUtils. createLastLayerDeltaMap(state), 1, 0)

//     Matrix.add(totalDelta, delta)
//   }, LayerUtils. createLastLayerDeltaMap(state))
// }

let _compute = (padExpandDeltaMap, state, inputs) => {
  let lastLayerNets = inputs->NP.mapMatrixMap(Matrix.map(_, ReluActivator.invert))

  ArraySt.range(0, state.filterNumber - 1)->ArraySt.reduceOneParam(
    (. lastLayerDeltaMap, filterIndex) => {
      let padExpandDelta = padExpandDeltaMap->ImmutableSparseMap.getExn(filterIndex)
      let filterState = state.filterStates->ImmutableSparseMap.getExn(filterIndex)

      let flippedWeights =
        Filter.getWeights(filterState)->ImmutableSparseMap.map((. weight) => weight->NP.rotate180)

      // "aaa"->Log.printForDebug->ignore
      let lastLayerDeltaMap = NP.addMatrixMap(
        lastLayerDeltaMap,
        LayerUtils.createLastLayerDeltaMap((
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

      lastLayerDeltaMap->ImmutableSparseMap.mapi((. lastLayerDelta, depthIndex) => {
        NP.dot(
          lastLayerDelta,
          lastLayerNets
          ->NP.mapMatrixMap(Matrix.map(_, ReluActivator.backward))
          ->ImmutableSparseMap.getExn(depthIndex),
        )
      })
    },
    LayerUtils.createLastLayerDeltaMap((state.depthNumber, state.inputWidth, state.inputHeight)),
  )
}

// let _multiplyActivatorDeriv = (delta, input) => {
//   NP.dot(delta, input->_elementWiseOp(ReluActivator.backward))
// }

// 计算传递到上一层的delta map
let bpDeltaMap = (
  state,
  inputs: ImmutableSparseMapType.t<depthIndex, Matrix.t>,
  deltaMap: ImmutableSparseMapType.t<filterIndex, Matrix.t>,
) => {
  _expandDeltaMapByStride(deltaMap, state)
  // ->Log.printForDebug
  ->_paddingDeltaMap(state)
  // ->Log.printForDebug
  // ->_computeDeltaWeightConv(state)
  // ->_multiplyActivatorDeriv(nets)
  ->_compute(state, inputs)
}

// let bpGradient = (state, outputMap, deltaMap) => {
let bpGradient = (state, paddedInputs, deltaMap) => {
  // ("bbb", deltaMap)->Log.printForDebug->ignore
  let expandDeltaMap = _expandDeltaMapByStride(deltaMap, state)

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

let backward = (state, inputs, deltaMap) => {
  // let (paddedInputs, _) = forward(state, inputs)
  let (paddedInputs, _) = forward(ReluActivator.forward, state, inputs)

  // let updatedDelta = bpDeltaMap(state, nets, deltaMap)
  let lastLayerDeltaMap = bpDeltaMap(state, inputs, deltaMap)

  let state = bpGradient(state, paddedInputs, deltaMap)

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
        // [
        //   [1., 0., 2., 2., 0.],
        //   [0., 0., 0., 2., 0.],
        //   [1., 2., 1., 2., 1.],
        //   [1., 0., 0., 0., 0.],
        //   [1., 2., 1., 1., 1.],
        // ],
        // [
        //   [2., 1., 2., 0., 0.],
        //   [1., 0., 0., 1., 0.],
        //   [0., 2., 1., 0., 1.],
        //   [0., 1., 2., 2., 2.],
        //   [2., 1., 0., 0., 1.],
        // ],
      ]->NP.createMatrixMapByDataArr

    let deltaMap =
      [
        [[0., 1., 1.], [2., 2., 2.], [1., 0., 0.]],
        // [[1., 0., 2.], [0., 0., 0.], [1., 2., 1.]],
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
    // let state = create(5, 5, 3, 3, 3, 2, 1, 2, 0.001)
    let state = create(5, 5, 1, 3, 3, 1, 1, 2, 0.001)

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
      // ->ImmutableSparseMap.set(
      //   1,
      //   {
      //     ...state.filterStates->ImmutableSparseMap.getExn(1),
      //     weights: [
      //       [[1., 1., -1.], [-1., -1., 1.], [0., -1., 1.]],
      //       [[0., 1., 0.], [-1., 0., -1.], [-1., 1., 0.]],
      //       [[-1., 0., 0.], [-1., 0., 1.], [-1., 0., 0.]],
      //     ]->NP.createMatrixMapByDataArr,
      //     bias: 0.,
      //   },
      // ),
    }

    (inputs, deltaMap, state)
  }

  let test = ((inputs, deltaMap, state)) => {
    // let (paddedInputs, (nets, outputMap)) = forward(state, inputs)
    let (paddedInputs, (nets, outputMap)) = forward(ReluActivator.forward, state, inputs)

    // inputs->Log.printForDebug-> ignore
    // ("f:", paddedInputs)->Log.printForDebug->ignore
    ("f:", nets)->Log.printForDebug->ignore
    // ("f:", outputMap)->Log.printForDebug->ignore

    let state = backward(state, inputs, deltaMap)

    let state = update(state)

    state.filterStates->Log.printForDebug->ignore
  }

  let checkGradient = () => {
    /* ! 设计一个误差函数，取所有节点输出项之和 */
    let _computeError = outputMap => {
      outputMap->NP.sumMatrixMap
    }

    let (inputs, deltaMap, state) = init()

    // let (paddedInputs, (nets, outputMap)) = forward(state, inputs)
    let (paddedInputs, (nets, outputMap)) = forward(ReluActivator.forward, state, inputs)

    // let (paddedInputs, _) = forward(state, inputs)

    Js.log(outputMap)
    Js.log(inputs)

    let (width, height, depth) = deltaMap->NP.getMatrixMapSize
    /* ! 因为误差函数为所有节点输出项之和，所以: if netValue > 0 then delta = 1, else delta = 0 */
    let nextLayerDeltaMap =
      nets->NP.mapMatrixMap(matrix => matrix->Matrix.map(value => value > 0. ? 1. : 0.))

      TODO fix

    // let lastLayerDeltaMap = bpDeltaMap(state, inputs, deltaMap)
    let layerDeltaMap = bpDeltaMap(state, outputMap, nextLayerDeltaMap)

    let state = bpGradient(state, paddedInputs, layerDeltaMap)

    // let state = backward(state, inputs, deltaMap)

    let epsilon = 10e-4
    let {weightGradients, weights} as filterState: Filter.state =
      state.filterStates->ImmutableSparseMap.getExn(0)
    weightGradients->ImmutableSparseMap.forEachi((weightGradient, depthIndex) => {
      weightGradient->NP.forEachMatrix((actualGradient, rowIndex, colIndex) => {
        let weight = weights->ImmutableSparseMap.getExn(depthIndex)

        let weightValue = weight->MatrixUtils.getValue(rowIndex, colIndex, _)

        let state1 = {
          ...state,
          filterStates: state.filterStates->ImmutableSparseMap.set(
            0,
            {
              ...filterState,
              weights: weights->ImmutableSparseMap.set(
                depthIndex,
                weight
                ->NP.copyMatrix
                ->MatrixUtils.setValue(weightValue +. epsilon, rowIndex, colIndex),
              ),
            },
          ),
        }

        let _activate_linear = net => {
          net
        }

        let (_, (_, outputMap1)) = forward(ReluActivator.forward, state1, inputs)
        let (_, (_, outputMap2)) = forward(_activate_linear, state, outputMap1)

        let err1 = _computeError(outputMap2)

        let state2 = {
          ...state,
          filterStates: state.filterStates->ImmutableSparseMap.set(
            0,
            {
              ...filterState,
              weights: weights->ImmutableSparseMap.set(
                depthIndex,
                weight
                ->NP.copyMatrix
                ->MatrixUtils.setValue(weightValue -. epsilon, rowIndex, colIndex),
              ),
            },
          ),
        }

        // let (_, (_, outputMap2)) = forward(state2, inputs)

        let (_, (_, outputMap1)) = forward(ReluActivator.forward, state2, inputs)
        let (_, (_, outputMap2)) = forward(_activate_linear, state, outputMap1)

        let err2 = _computeError(outputMap2)

        let expectedGradient = (err1 -. err2) /. (2. *. epsilon)

        let result =
          FloatUtils.truncateFloatValue(expectedGradient, 4) ==
            FloatUtils.truncateFloatValue(actualGradient, 4)

        Js.log({
          j`check gradient -> weights($depthIndex), $rowIndex, $colIndex): $result`
        })

        // (expectedGradient, actualGradient)->Log.printForDebug->ignore
      })
    })
  }
}

// Test.init()->Test.test

Test.checkGradient()
