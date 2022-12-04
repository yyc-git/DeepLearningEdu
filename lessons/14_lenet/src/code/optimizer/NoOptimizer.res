let update = (data, (learnRate, t: int, _), vt_1, st_1, gradient) => {
  (data -. learnRate *. gradient, (vt_1, st_1))
}

let updateWeight = OptimizerUtils.updateWeight(update)

let buildData = (): Optimizer.optimizerFuncs => {
  updateNetworkHyperparamFunc: (networkHyperparam: Optimizer.networkHyperparam) => {
    networkHyperparam
  },
  updateValueFunc: update,
  updateWeightFunc: updateWeight,
}
