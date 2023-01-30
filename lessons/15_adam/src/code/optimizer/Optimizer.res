type networkHyperparam = {t: int}

type layerHyperparam = {
  learnRate: float,
  beta1: float,
  beta2: float,
  epsion: float,
  // weightDecay: float,
}

type updateNetworkHyperparamFunc = networkHyperparam => networkHyperparam

type updateValueFunc = (
  float,
  (float, int, (float, float, float)),
  float,
  float,
  float,
) => (float, (float, float))

type updateWeightFunc = (
  Matrix.t,
  // (float, int, (float, float, float), float),
  (float, int, (float, float, float)),
  Matrix.t,
  int,
  (Matrix.t, Matrix.t),
) => (Matrix.t, (Matrix.t, Matrix.t))

type optimizerFuncs = {
  updateNetworkHyperparamFunc: updateNetworkHyperparamFunc,
  updateValueFunc: updateValueFunc,
  updateWeightFunc: updateWeightFunc,
}

type optimizerData = {
  optimizerFuncs: optimizerFuncs,
  networkHyperparam: networkHyperparam,
  networkLayerHyperparam: option<layerHyperparam>,
  allLayerHyperparams: option<array<layerHyperparam>>,
}

let create = (optimizerFuncs, networkHyperparam, networkLayerHyperparam, allLayerHyperparams) => {
  optimizerFuncs: optimizerFuncs,
  networkHyperparam: networkHyperparam,
  networkLayerHyperparam: networkLayerHyperparam,
  allLayerHyperparams: allLayerHyperparams,
}

let getLayerHyperparam = (optimizerData: optimizerData, layerIndex: int) => {
  switch optimizerData.networkLayerHyperparam {
  | Some(networkLayerHyperparam) => networkLayerHyperparam
  | None => optimizerData.allLayerHyperparams->OptionSt.getExn->ArraySt.getExn(layerIndex)
  }
}

let updateNetworkHyperParam = (optimizerData: optimizerData) => {
  ...optimizerData,
  networkHyperparam: optimizerData.optimizerFuncs.updateNetworkHyperparamFunc(
    optimizerData.networkHyperparam,
  ),
}
