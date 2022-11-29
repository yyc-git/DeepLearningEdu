'use strict';

var Curry = require("rescript/lib/js/curry.js");
var ArraySt$Cnn = require("../ArraySt.bs.js");
var OptionSt$Cnn = require("../OptionSt.bs.js");

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

exports.create = create;
exports.getLayerHyperparam = getLayerHyperparam;
exports.updateNetworkHyperParam = updateNetworkHyperParam;
/* No side effect */
