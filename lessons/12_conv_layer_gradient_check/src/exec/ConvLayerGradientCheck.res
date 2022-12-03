open ConvLayer

let _initForSimple = () => {
  let inputs = [
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

  // let deltaMap = [
  //   [[0., 1., 1.], [2., 2., 2.], [1., 0., 0.]],
  //   // [[1., 0., 2.], [0., 0., 0.], [1., 2., 1.]],
  // ]->NP.createMatrixMapByDataArr

  // inputWidth,
  // inputHeight,
  // depthNumber,
  // filterWidth,
  // filterHeight,
  // filterNumber,
  // zeroPadding,
  // stride,
  // leraningRate,
  let stateForConvLayer1 = create(5, 5, 1, 3, 3, 1, 0, 1, 0.001)

  let stateForConvLayer1 = {
    ...stateForConvLayer1,
    filterStates: ImmutableSparseMap.createEmpty()->ImmutableSparseMap.set(
      0,
      (
        {
          ...stateForConvLayer1.filterStates->ImmutableSparseMap.getExn(0),
          weights: [
            [[-1., 1., 0.], [0., 1., 0.], [0., 1., 1.]],
            // [[-1., -1., 0.], [0., 0., 0.], [0., -1., 0.]],
            // [[0., 0., -1.], [0., 1., 0.], [1., -1., -1.]],
          ]->NP.createMatrixMapByDataArr,
          bias: 1.,
        }: Filter.state
      ),
    ),
  }

  let stateForConvLayer2 = create(3, 3, 1, 2, 2, 1, 0, 1, 0.001)

  let stateForConvLayer2 = {
    ...stateForConvLayer2,
    filterStates: ImmutableSparseMap.createEmpty()->ImmutableSparseMap.set(
      0,
      (
        {
          ...stateForConvLayer2.filterStates->ImmutableSparseMap.getExn(0),
          weights: [[[1., 0.], [1., 0.]]]->NP.createMatrixMapByDataArr,
          bias: 1.,
        }: Filter.state
      ),
    ),
  }

  ((stateForConvLayer1, stateForConvLayer2), inputs)
}

let _initForMultiDepthsAndFilterNumbers = () => {
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

  // let deltaMap = [
  //   [[0., 1., 1.], [2., 2., 2.], [1., 0., 0.]],
  //   // [[1., 0., 2.], [0., 0., 0.], [1., 2., 1.]],
  // ]->NP.createMatrixMapByDataArr

  // inputWidth,
  // inputHeight,
  // depthNumber,
  // filterWidth,
  // filterHeight,
  // filterNumber,
  // zeroPadding,
  // stride,
  // leraningRate,
  //   let stateForConvLayer1 = create(5, 5, 3, 3, 3, 2, 0, 1, 0.001)
  // let stateForConvLayer1 = create(5, 5, 3, 3, 3, 1, 0, 1, 0.001)
  //   let stateForConvLayer1 = create(5, 5, 1, 3, 3, 1, 0, 1, 0.001)
  //   let stateForConvLayer1 = create(5, 5, 3, 3, 3, 1, 0, 1, 0.001)
  let stateForConvLayer1 = create(5, 5, 3, 3, 3, 2, 0, 1, 0.001)

  let stateForConvLayer1 = {
    ...stateForConvLayer1,
    filterStates: ImmutableSparseMap.createEmpty()
    ->ImmutableSparseMap.set(
      0,
      (
        {
          ...stateForConvLayer1.filterStates->ImmutableSparseMap.getExn(0),
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
        ...stateForConvLayer1.filterStates->ImmutableSparseMap.getExn(1),
        weights: [
          [[1., 1., -1.], [-1., -1., 1.], [0., -1., 1.]],
          [[0., 1., 0.], [-1., 0., -1.], [-1., 1., 0.]],
          [[-1., 0., 0.], [-1., 0., 1.], [-1., 0., 0.]],
        ]->NP.createMatrixMapByDataArr,
        bias: 0.,
      },
    ),
  }

  //   let stateForConvLayer2 = create(3, 3, 2, 2, 2, 1, 0, 1, 0.001)
  let stateForConvLayer2 = create(3, 3, 2, 2, 2, 2, 0, 1, 0.001)
  //   let stateForConvLayer2 = create(3, 3, 1, 2, 2, 1, 0, 1, 0.001)

  let stateForConvLayer2 = {
    ...stateForConvLayer2,
    filterStates: ImmutableSparseMap.createEmpty()
    ->ImmutableSparseMap.set(
      0,
      (
        {
          ...stateForConvLayer2.filterStates->ImmutableSparseMap.getExn(0),
          weights: [[[1., 0.], [1., 0.]], [[-1., 0.], [0., 0.]]]->NP.createMatrixMapByDataArr,
          //   weights: [[[1., 0.], [1., 0.]]]->NP.createMatrixMapByDataArr,
          bias: 1.,
        }: Filter.state
      ),
    )
    ->ImmutableSparseMap.set(
      1,
      (
        {
          ...stateForConvLayer2.filterStates->ImmutableSparseMap.getExn(1),
          weights: [[[0., 1.], [1., 0.]], [[-1., 0.], [0., 0.]]]->NP.createMatrixMapByDataArr,
          bias: 1.,
        }: Filter.state
      ),
    ),
  }

  ((stateForConvLayer1, stateForConvLayer2), inputs)
}

let _initForMultiStridesAndPaddings = () => {
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

  // let deltaMap = [
  //   [[0., 1., 1.], [2., 2., 2.], [1., 0., 0.]],
  //   // [[1., 0., 2.], [0., 0., 0.], [1., 2., 1.]],
  // ]->NP.createMatrixMapByDataArr

  // inputWidth,
  // inputHeight,
  // depthNumber,
  // filterWidth,
  // filterHeight,
  // filterNumber,
  // zeroPadding,
  // stride,
  // leraningRate,
  //   let stateForConvLayer1 = create(5, 5, 3, 3, 3, 2, 0, 1, 0.001)
  // let stateForConvLayer1 = create(5, 5, 3, 3, 3, 1, 0, 1, 0.001)
  //   let stateForConvLayer1 = create(5, 5, 1, 3, 3, 1, 0, 1, 0.001)
  //   let stateForConvLayer1 = create(5, 5, 3, 3, 3, 1, 0, 1, 0.001)
  let stateForConvLayer1 = create(5, 5, 3, 3, 3, 2, 1, 2, 0.001)

  let stateForConvLayer1 = {
    ...stateForConvLayer1,
    filterStates: ImmutableSparseMap.createEmpty()
    ->ImmutableSparseMap.set(
      0,
      (
        {
          ...stateForConvLayer1.filterStates->ImmutableSparseMap.getExn(0),
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
        ...stateForConvLayer1.filterStates->ImmutableSparseMap.getExn(1),
        weights: [
          [[1., 1., -1.], [-1., -1., 1.], [0., -1., 1.]],
          [[0., 1., 0.], [-1., 0., -1.], [-1., 1., 0.]],
          [[-1., 0., 0.], [-1., 0., 1.], [-1., 0., 0.]],
        ]->NP.createMatrixMapByDataArr,
        bias: 0.,
      },
    ),
  }

  //   let stateForConvLayer2 = create(3, 3, 2, 2, 2, 2, 0, 1, 0.001)
  //   let stateForConvLayer2 = create(3, 3, 2, 2, 2, 2, 0, 2, 0.001)
  let stateForConvLayer2 = create(3, 3, 2, 2, 2, 2, 1, 2, 0.001)

  let stateForConvLayer2 = {
    ...stateForConvLayer2,
    filterStates: ImmutableSparseMap.createEmpty()
    ->ImmutableSparseMap.set(
      0,
      (
        {
          ...stateForConvLayer2.filterStates->ImmutableSparseMap.getExn(0),
          weights: [[[1., 0.], [1., 0.]], [[-1., 0.], [0., 0.]]]->NP.createMatrixMapByDataArr,
          //   weights: [[[1., 0.], [1., 0.]]]->NP.createMatrixMapByDataArr,
          bias: 1.,
        }: Filter.state
      ),
    )
    ->ImmutableSparseMap.set(
      1,
      (
        {
          ...stateForConvLayer2.filterStates->ImmutableSparseMap.getExn(1),
          weights: [[[0., 1.], [1., 0.]], [[-1., 0.], [0., 0.]]]->NP.createMatrixMapByDataArr,
          bias: 1.,
        }: Filter.state
      ),
    ),
  }

  ((stateForConvLayer1, stateForConvLayer2), inputs)
}

let checkGradient = (((stateForConvLayer1, stateForConvLayer2), inputs)) => {
  /* ! 设计一个误差函数，取所有节点输出项之和 */
  let _computeError = outputMap => {
    outputMap->NP.sumMatrixMap
  }

  //   let ((stateForConvLayer1, stateForConvLayer2), inputs) = _initForSimple()
  //   let ((stateForConvLayer1, stateForConvLayer2), inputs) = _initForMultiDepthsAndFilterNumbers()
  //   let ((stateForConvLayer1, stateForConvLayer2), inputs) = _initForMultiStridesAndPaddings()

  // let (inputs, deltaMap, state) = init()

  // let (paddedInputs, (nets, outputMap)) = forward(state, inputs)
  let (paddedInputs, (nets, outputMap)) = forward(ReluActivator.forward, stateForConvLayer1, inputs)

  // let (paddedInputs, _) = forward(state, inputs)

  // Js.log(outputMap)
  // Js.log(inputs)
  // Js.log(paddedInputs)
  //   Js.log(nets)

  /* ! because Ed(_computeError) of this layer and previous layer are the same one, so the activator of this layer and previous layer should be the same one(Identity)! */
  // NP.createMatrixMap(() => 0., 1, 2,2)
  //   let convLayer2DeltaMap = nets->NP.mapMatrixMap(matrix => matrix->Matrix.map(value => 1.))
  //   let convLayer2DeltaMap = nets->NP.mapMatrixMap(matrix => matrix->Matrix.map(value => 1.))
  let (col, row, depth) = (
    stateForConvLayer2.outputWidth,
    stateForConvLayer2.outputHeight,
    stateForConvLayer2.filterNumber,
  )
  let convLayer2DeltaMap = NP.createMatrixMap(() => 1., depth, row, col)

  // Js.log(outputMap)
  // Js.log(convLayer2DeltaMap)
  // Js.log("b1")
  // let lastLayerDeltaMap = bpDeltaMap(state, inputs, deltaMap)
  let convLayer1DeltaMap = bpDeltaMap(stateForConvLayer2, outputMap, convLayer2DeltaMap)
  // Js.log("b2")

  let stateForConvLayer1 = bpGradient(stateForConvLayer1, paddedInputs, convLayer1DeltaMap)

  // let state = backward(state, inputs, deltaMap)

  let epsilon = 10e-4
  let {weightGradients, weights} as filterState: Filter.state =
    stateForConvLayer1.filterStates->ImmutableSparseMap.getExn(0)
  weightGradients->ImmutableSparseMap.forEachi((weightGradient, depthIndex) => {
    weightGradient->NP.forEachMatrix((actualGradient, rowIndex, colIndex) => {
      let weight = weights->ImmutableSparseMap.getExn(depthIndex)

      let weightValue = weight->MatrixUtils.getValue(rowIndex, colIndex, _)

      let stateForConvLayer1_1 = {
        ...stateForConvLayer1,
        filterStates: stateForConvLayer1.filterStates->ImmutableSparseMap.set(
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
      // Js.log("bb")

      let (_, (_, outputMap1)) = forward(ReluActivator.forward, stateForConvLayer1_1, inputs)
      let (_, (_, outputMap2)) = forward(_activate_linear, stateForConvLayer2, outputMap1)

      let err1 = _computeError(outputMap2)

      let stateForConvLayer1_2 = {
        ...stateForConvLayer1,
        filterStates: stateForConvLayer1.filterStates->ImmutableSparseMap.set(
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

      let (_, (_, outputMap1)) = forward(ReluActivator.forward, stateForConvLayer1_2, inputs)
      let (_, (_, outputMap2)) = forward(_activate_linear, stateForConvLayer2, outputMap1)

      let err2 = _computeError(outputMap2)

      let expectedGradient = (err1 -. err2) /. (2. *. epsilon)

      let result =
        FloatUtils.truncateFloatValue(expectedGradient, 4) ==
          FloatUtils.truncateFloatValue(actualGradient, 4)

      Js.log({
        j`check gradient -> weights($depthIndex), $rowIndex, $colIndex): $result`
      })

      (expectedGradient, actualGradient)->Log.printForDebug->ignore
    })
  })
}

Js.log("_initForSimple:")
_initForSimple()->checkGradient
Js.log("_initForMultiDepthsAndFilterNumbers:")
_initForMultiDepthsAndFilterNumbers()->checkGradient
Js.log("_initForMultiStridesAndPaddings:")
_initForMultiStridesAndPaddings()->checkGradient
