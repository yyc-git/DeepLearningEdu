open Optimizer

let buildLayerAdamWOptimizerData = (
  ~learnRate=0.001,
  ~beta1=0.9,
  ~beta2=0.999,
  ~epsion=1e-6,
  // ~weightDecay=0.0,
  (),
) => {
  {
    learnRate: learnRate,
    // weightDecay: weightDecay,
    beta1: beta1,
    beta2: beta2,
    epsion: epsion,
  }
}

let buildNetworkAdamWOptimizerData = (
  ~learnRate=0.001,
  ~t=1,
  ~beta1=0.9,
  ~beta2=0.999,
  ~epsion=1e-6,
  // ~weightDecay=0.0,
  (),
) => {
  {
    optimizerFuncs: AdamW.buildData(),
    networkHyperparam: {t: t},
    networkLayerHyperparam: buildLayerAdamWOptimizerData(
      ~learnRate,
      ~beta1,
      ~beta2,
      ~epsion,
      // ~weightDecay,
      (),
    )->Some,
    allLayerHyperparams: None,
  }
}

let buildNetworkAdamWOptimizerDataWithAllLayerHyperparams = (~allLayerHyperparams, ~t=1, ()) => {
  {
    optimizerFuncs: AdamW.buildData(),
    networkHyperparam: {t: t},
    networkLayerHyperparam: None,
    allLayerHyperparams: allLayerHyperparams->Some,
  }
}
