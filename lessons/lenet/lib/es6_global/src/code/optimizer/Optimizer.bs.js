

import * as Curry from "../../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as ArraySt$Cnn from "../ArraySt.bs.js";
import * as OptionSt$Cnn from "../OptionSt.bs.js";

function create(optimizerFuncs, networkHyperparam, networkLayerHyperparam, allLayerHyperparams) {
  return {
          optimizerFuncs: optimizerFuncs,
          networkHyperparam: networkHyperparam,
          networkLayerHyperparam: networkLayerHyperparam,
          allLayerHyperparams: allLayerHyperparams
        };
}

function getLayerHyperparam(optimizerData, layerIndex) {
  var networkLayerHyperparam = optimizerData.networkLayerHyperparam;
  if (networkLayerHyperparam !== undefined) {
    return networkLayerHyperparam;
  } else {
    return ArraySt$Cnn.getExn(OptionSt$Cnn.getExn(optimizerData.allLayerHyperparams), layerIndex);
  }
}

function updateNetworkHyperParam(optimizerData) {
  return {
          optimizerFuncs: optimizerData.optimizerFuncs,
          networkHyperparam: Curry._1(optimizerData.optimizerFuncs.updateNetworkHyperparamFunc, optimizerData.networkHyperparam),
          networkLayerHyperparam: optimizerData.networkLayerHyperparam,
          allLayerHyperparams: optimizerData.allLayerHyperparams
        };
}

export {
  create ,
  getLayerHyperparam ,
  updateNetworkHyperParam ,
  
}
/* No side effect */
