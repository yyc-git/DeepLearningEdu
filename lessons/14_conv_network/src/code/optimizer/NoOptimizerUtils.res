open Optimizer

let buildNetworkNoOptimizerData = (~allLearnRates=[10., 0.1], ()) => {
  {
    optimizerFuncs: NoOptimizer.buildData(),
    networkHyperparam: {t: 0},
    networkLayerHyperparam: None,
    allLayerHyperparams: allLearnRates
    ->ArraySt.map(learnRate => {
      {learnRate: learnRate, beta1: 0., beta2: 0., epsion: 0.}
    })
    ->Some,
  }
}
