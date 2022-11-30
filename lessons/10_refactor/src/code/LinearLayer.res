open LinearLayerStateType

open ActivatorType

let getOutputNumber = outputVector => {
  let (_, maxValueIndex) =
    outputVector->Vector.reducei((. (maxValue, maxValueIndex), value, index) => {
      value >= maxValue ? (value, Some(index)) : (maxValue, maxValueIndex)
    }, (0., None))

  maxValueIndex
}

let _createWeight = (
  initValueMethodFunc,
  randomFunc,
  inLinearLayerNodeCount,
  outLinearLayerNodeCount,
) => {
  Matrix.create(
    outLinearLayerNodeCount,
    inLinearLayerNodeCount,
    ArraySt.range(0, outLinearLayerNodeCount * inLinearLayerNodeCount - 1)->ArraySt.map(_ =>
      initValueMethodFunc(randomFunc, inLinearLayerNodeCount, outLinearLayerNodeCount)
    ),
  )
}

let _createBias = outLinearLayerNodeCount => {
  Vector.create(
    ArraySt.range(0, outLinearLayerNodeCount - 1)->ArraySt.map(_ => InitValue.constant(0.0)),
  )
}

let _createZeroWeight = (inLinearLayerNodeCount, outLinearLayerNodeCount) => {
  Matrix.create(
    outLinearLayerNodeCount,
    inLinearLayerNodeCount,
    ArraySt.range(0, outLinearLayerNodeCount * inLinearLayerNodeCount - 1)->ArraySt.map(_ => 0.0),
  )
}

let _createZeroBias = outLinearLayerNodeCount => {
  Vector.create(ArraySt.range(0, outLinearLayerNodeCount - 1)->ArraySt.map(_ => 0.0))
}

// let _initValue = () => {
//   // (Js.Math.random() *. 2. -. 1.) *. 1e-4
//   Js.Math.random()
// }

let getNodeCount = ({weight}) => {
  (Matrix.getColCount(weight), Matrix.getRowCount(weight))
}

// let forwardLayer1: LayerAbstractType.forward<Vector.t, Vector.t> = (_, inputVector, _) => {
//   (inputVector->Some, inputVector)
// }

let forward: LayerAbstractType.forward<Vector.t, Vector.t> = (state, activatorData, input, _) => {
  let {weight, bias} = state->Obj.magic
  let {forwardNet} = activatorData->OptionSt.getExn

  let net = Vector.transformMatrix(weight, input)->Vector.add(_, bias)
  let output = net->forwardNet

  // TODO restore debug
  // DebugUtils.checkOutputVectorExplosion(output)

  (state, net->Some, output)
}

// let forwardLayer3: LayerAbstractType.forward<Vector.t, Vector.t> = (
//   _,
//   layer2OutputVector,
//   state,
// ) => {
//   let state = state->OptionSt.getExn->Obj.magic

//   Layer.Forward.forwardLinear(
//     SoftmaxActivator.buildData(),
//     layer2OutputVector,
//     state.wMatrixBetweenLayer2Layer3,
//     state.layer3BVector,
//   )
// }

// let computeLayer3Delta = (layer3OutputVector, labelVector) => {
//   CrossEntropyLoss.computeDelta(layer3OutputVector, labelVector)
// }

let createGradientDataSum: LayerAbstractType.createGradientDataSum<Matrix.t, Vector.t> = state => {
  let (inLinearLayerNodeCount, outLinearLayerNodeCount) = state->Obj.magic->getNodeCount

  (
    _createZeroWeight(inLinearLayerNodeCount, outLinearLayerNodeCount),
    _createZeroBias(outLinearLayerNodeCount),
  )->Some
}

let bpDelta: LayerAbstractType.bpDelta<Vector.t, Vector.t> = (
  previousLayerActivatorData,
  (_, previousLayerNet),
  currentLayerDelta,
  state,
) => {
  let {backward} = previousLayerActivatorData->OptionSt.getExn
  let previousLayerNet = previousLayerNet->OptionSt.getExn
  let {weight} = state->Obj.magic

  let previousLayerDelta = previousLayerNet->Vector.mapi((previousLayerNetValue, j) => {
    previousLayerNetValue->backward *.
      Vector.dot(
        currentLayerDelta,
        MatrixUtils.getCol(
          Matrix.getRowCount(weight),
          Matrix.getColCount(weight),
          j,
          Matrix.getData(weight),
        ),
      )
  })

  previousLayerDelta->Some
}

let computeGradient: LayerAbstractType.computeGradient<Vector.t, Vector.t, Matrix.t, Vector.t> = (
  input,
  delta,
  _,
) => {
  (
    Matrix.multiply(
      Matrix.create(Vector.length(delta), 1, delta),
      Matrix.create(1, Vector.length(input), input),
    ),
    delta,
  )
}

let backward: LayerAbstractType.backward<Vector.t, Vector.t, Matrix.t, Vector.t> = (
  previousLayerActivatorData,
  (previousLayerOutput, previousLayerNet) as previousLayerData,
  currentLayerDelta,
  state,
) => {
  let previousLayerDelta = bpDelta(
    previousLayerActivatorData,
    previousLayerData,
    currentLayerDelta,
    state,
  )
  let previousLayerOutput = previousLayerOutput->OptionSt.getExn

  (
    previousLayerDelta,
    computeGradient(previousLayerOutput, currentLayerDelta, state->Obj.magic)->Some,
  )
}

let addToGradientDataSum: LayerAbstractType.addToGradientDataSum<Matrix.t, Vector.t> = (
  gradientDataSum,
  gradientData,
) => {
  let (gradientDataSumForWeight, gradientDataSumForBias) = gradientDataSum->OptionSt.getExn
  let (gradientDataForWeight, gradientDataForBias) = gradientData->OptionSt.getExn

  (
    Matrix.add(gradientDataSumForWeight, gradientDataForWeight),
    Vector.add(gradientDataSumForBias, gradientDataForBias),
  )->Some
}

let update: LayerAbstractType.update<Matrix.t, Vector.t> = (
  state,
  // ({updateValueFunc, updateWeightFunc}, {t}, {learnRate, weightDecay, beta1, beta2, epsion}),
  ({updateValueFunc, updateWeightFunc}, {t}, {learnRate, beta1, beta2, epsion}),
  (miniBatchSize, gradientDataSum),
) => {
  let state = state->Obj.magic
  let (gradientDataSumForWeight, gradientDataSumForBias) = gradientDataSum->OptionSt.getExn

  let {vWeight, vBias, sWeight, sBias} = state.adamData

  let (newWeight, (vtForWeigth, stForWeight)) = updateWeightFunc(
    state.weight,
    // (learnRate, t, (beta1, beta2, epsion), weightDecay),
    (learnRate, t, (beta1, beta2, epsion)),
    gradientDataSumForWeight,
    miniBatchSize,
    (vWeight, sWeight),
  )

  let (newBiasValueArr, (vtForBiasArr, stForBiasArr)) =
    state.bias->Vector.reducei((. (newBiasValueArr, (vtArr, stArr)), biasValue, index) => {
      let (newBiasValue, (vt, st)) = updateValueFunc(
        biasValue,
        (learnRate, t, (beta1, beta2, epsion)),
        vBias->Obj.magic->Vector.getExn(index),
        sBias->Obj.magic->Vector.getExn(index),
        gradientDataSumForBias->Vector.getExn(index) /. miniBatchSize->Obj.magic,
      )

      (
        newBiasValueArr->ArraySt.push(newBiasValue),
        (vtArr->ArraySt.push(vt), stArr->ArraySt.push(st)),
      )
    }, ([], ([], [])))

  let (inLinearLayerNodeCount, outLinearLayerNodeCount) = state->getNodeCount

  {
    adamData: {
      vWeight: vtForWeigth,
      sWeight: stForWeight,
      vBias: Vector.create(vtForBiasArr),
      sBias: Vector.create(stForBiasArr),
    },
    weight: newWeight,
    bias: Vector.create(newBiasValueArr),
  }->Obj.magic
}

let _createAdamData = (inLinearLayerNodeCount, outLinearLayerNodeCount) => {
  {
    vWeight: _createZeroWeight(inLinearLayerNodeCount, outLinearLayerNodeCount),
    vBias: _createZeroBias(outLinearLayerNodeCount),
    sWeight: _createZeroWeight(inLinearLayerNodeCount, outLinearLayerNodeCount),
    sBias: _createZeroBias(outLinearLayerNodeCount),
  }
}

let create = (
  ~inLinearLayerNodeCount,
  ~outLinearLayerNodeCount,
  ~initValueMethod=Random.random,
  ~randomFunc=Js.Math.random,
  (),
) => {
  weight: _createWeight(
    initValueMethod,
    randomFunc,
    inLinearLayerNodeCount,
    outLinearLayerNodeCount,
  ),
  bias: _createBias(outLinearLayerNodeCount),
  adamData: _createAdamData(inLinearLayerNodeCount, outLinearLayerNodeCount),
}

let createLayerData: LayerAbstractType.createLayerData = (state, activatorData) => {
  {
    layerName: #linear,
    state: state,
    forward: forward->Obj.magic,
    backward: backward->Obj.magic,
    update: update->Obj.magic,
    createGradientDataSum: createGradientDataSum->Obj.magic,
    addToGradientDataSum: addToGradientDataSum->Obj.magic,
    activatorData: activatorData,
    getWeight: state => {
      let state = state->Obj.magic

      state.weight->Obj.magic->Some
    },
    getBias: state => {
      let state = state->Obj.magic

      state.bias->Obj.magic->Some
    },
  }
}
