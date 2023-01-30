open FilterStateType

let _createAdamData = (depth, height, width) => {
  {
    vWeight: NP.zeroMatrixMap(depth, height, width),
    vBias: 0.0,
    sWeight: NP.zeroMatrixMap(depth, height, width),
    sBias: 0.0,
  }
}

let create = ((initValueMethodFunc, randomFunc, fanIn, fanOut), width, height, depth) => {
  {
    weights: NP.createMatrixMap(
      () => initValueMethodFunc(randomFunc, fanIn, fanOut),
      depth,
      height,
      width,
    ),
    bias: InitValue.constant(0.0),
    adamData: _createAdamData(depth, height, width),
  }
}

let getWeights = state => {
  state.weights
}

let getBias = state => {
  state.bias
}

let update = (
  state,
  (miniBatchSize, gradientDataSumForWeight, gradientDataSumForBias),
  (
    {updateValueFunc, updateWeightFunc}: Optimizer.optimizerFuncs,
    {t}: Optimizer.networkHyperparam,
    // {learnRate, weightDecay, beta1, beta2, epsion}: Optimizer.layerHyperparam,
    {learnRate, beta1, beta2, epsion}: Optimizer.layerHyperparam,
  ),
) => {
  // Js.log(( "filter learnRate:", learnRate ))
  let {vWeight, vBias, sWeight, sBias} = state.adamData

  let (newWeights, vtForWeigths, stForWeights) =
    state.weights->ImmutableSparseMap.reducei(
      (. (newWeights, vtForWeigths, stForWeights), weight, i) => {
        let gradientDataSumForWeight = gradientDataSumForWeight->ImmutableSparseMap.getExn(i)
        let vWeight = vWeight->ImmutableSparseMap.getExn(i)
        let sWeight = sWeight->ImmutableSparseMap.getExn(i)

        let (newWeight, (vtForWeigth, stForWeight)) = updateWeightFunc(
          weight,
          // (learnRate, t, (beta1, beta2, epsion), weightDecay),
          (learnRate, t, (beta1, beta2, epsion)),
          gradientDataSumForWeight,
          miniBatchSize,
          (vWeight, sWeight),
        )

        (
          newWeights->ImmutableSparseMap.set(i, newWeight),
          vtForWeigths->ImmutableSparseMap.set(i, vtForWeigth),
          stForWeights->ImmutableSparseMap.set(i, stForWeight),
        )
      },
      (
        ImmutableSparseMap.createEmpty(),
        ImmutableSparseMap.createEmpty(),
        ImmutableSparseMap.createEmpty(),
      ),
    )

  let (newBias, (vtForBias, stForBias)) = updateValueFunc(
    state.bias,
    (learnRate, t, (beta1, beta2, epsion)),
    vBias->Obj.magic,
    sBias->Obj.magic,
    gradientDataSumForBias /. miniBatchSize->Obj.magic,
  )

  {
    ...state,
    adamData: {
      vWeight: vtForWeigths,
      sWeight: stForWeights,
      vBias: vtForBias,
      sBias: stForBias,
    },
    weights: newWeights,
    bias: newBias,
  }
}
