'use strict';

var AdamW$Cnn = require("./AdamW.bs.js");

function buildLayerAdamWOptimizerData(learnRateOpt, beta1Opt, beta2Opt, epsionOpt, param) {
  var learnRate = learnRateOpt !== undefined ? learnRateOpt : 0.001;
  var beta1 = beta1Opt !== undefined ? beta1Opt : 0.9;
  var beta2 = beta2Opt !== undefined ? beta2Opt : 0.999;
  var epsion = epsionOpt !== undefined ? epsionOpt : 1e-6;
  return {
          learnRate: learnRate,
          beta1: beta1,
          beta2: beta2,
          epsion: epsion
        };
}

function buildNetworkAdamWOptimizerData(learnRateOpt, tOpt, beta1Opt, beta2Opt, epsionOpt, param) {
  var learnRate = learnRateOpt !== undefined ? learnRateOpt : 0.001;
  var t = tOpt !== undefined ? tOpt : 1;
  var beta1 = beta1Opt !== undefined ? beta1Opt : 0.9;
  var beta2 = beta2Opt !== undefined ? beta2Opt : 0.999;
  var epsion = epsionOpt !== undefined ? epsionOpt : 1e-6;
  return {
          optimizerFuncs: AdamW$Cnn.buildData(undefined),
          networkHyperparam: {
            t: t
          },
          networkLayerHyperparam: buildLayerAdamWOptimizerData(learnRate, beta1, beta2, epsion, undefined),
          allLayerHyperparams: undefined
        };
}

function buildNetworkAdamWOptimizerDataWithAllLayerHyperparams(allLayerHyperparams, tOpt, param) {
  var t = tOpt !== undefined ? tOpt : 1;
  return {
          optimizerFuncs: AdamW$Cnn.buildData(undefined),
          networkHyperparam: {
            t: t
          },
          networkLayerHyperparam: undefined,
          allLayerHyperparams: allLayerHyperparams
        };
}

exports.buildLayerAdamWOptimizerData = buildLayerAdamWOptimizerData;
exports.buildNetworkAdamWOptimizerData = buildNetworkAdamWOptimizerData;
exports.buildNetworkAdamWOptimizerDataWithAllLayerHyperparams = buildNetworkAdamWOptimizerDataWithAllLayerHyperparams;
/* No side effect */
