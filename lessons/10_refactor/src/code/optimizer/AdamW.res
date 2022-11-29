let update = (data, (learnRate, t: int, (beta1, beta2, epsion)), vt_1, st_1, gradient) => {
  let vt = vt_1 *. beta1 +. (1. -. beta1) *. gradient
  let st = st_1 *. beta2 +. (1. -. beta2) *. gradient *. gradient

  let vBiasCorrect = vt /. (1. -. Js.Math.pow_float(~base=beta1, ~exp=t->Obj.magic))
  let sBiasCorrect = st /. (1. -. Js.Math.pow_float(~base=beta2, ~exp=t->Obj.magic))

  (data -. learnRate *. vBiasCorrect /. (Js.Math.sqrt(sBiasCorrect) +. epsion), (vt, st))
}

// let weightDecayForAdamW = (newWeightValue, oldWeightValue, weightDecay, learnRate) => {
//   newWeightValue -. oldWeightValue *. learnRate *. weightDecay
// }

let updateWeight = (
  weight,
  // (learnRate, t, (beta1, beta2, epsion), weightDecay),
  (learnRate, t, (beta1, beta2, epsion)),
  gradientDataSum,
  miniBatchSize,
  (vWeight, sWeight),
) => {
  let (newWeightValueArr, vtForWeightArr, stForWeightArr) =
    weight->NP.reduceMatrix(
      ((newWeightValueArr, vtArr, stArr), weightValue, rowIndex, colIndex) => {
        let (newWeightValue, (vt, st)) = update(
          weightValue,
          (learnRate, t, (beta1, beta2, epsion)),
          vWeight->Obj.magic->MatrixUtils.getValue(rowIndex, colIndex),
          sWeight->Obj.magic->MatrixUtils.getValue(rowIndex, colIndex),
          gradientDataSum->MatrixUtils.getValue(rowIndex, colIndex) /. miniBatchSize->Obj.magic,
        )

        // let newWeightValue =
        //   newWeightValue->weightDecayForAdamW(weightValue, weightDecay, learnRate)

        (
          newWeightValueArr->ArraySt.push(newWeightValue),
          vtArr->ArraySt.push(vt),
          stArr->ArraySt.push(st),
        )
      },
      ([], [], []),
    )

  let (rowCount, colCount) = (Matrix.getRowCount(weight), Matrix.getColCount(weight))

  (
    Matrix.create(rowCount, colCount, newWeightValueArr),
    (
      Matrix.create(rowCount, colCount, vtForWeightArr),
      Matrix.create(rowCount, colCount, stForWeightArr),
    ),
  )
}

let increaseT = t => {
  t->succ
}

let buildData = (): Optimizer.optimizerFuncs => {
  updateNetworkHyperparamFunc: (networkHyperparam: Optimizer.networkHyperparam) => {
    ...networkHyperparam,
    t: networkHyperparam.t->increaseT,
  },
  updateValueFunc: update,
  updateWeightFunc: updateWeight,
}
