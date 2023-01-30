'use strict';

var ArraySt$Cnn = require("../ArraySt.bs.js");
var NoOptimizer$Cnn = require("./NoOptimizer.bs.js");

function buildNetworkNoOptimizerData(allLearnRatesOpt, param) {
  var allLearnRates = allLearnRatesOpt !== undefined ? allLearnRatesOpt : [
      10,
      0.1
    ];
  return {
          optimizerFuncs: NoOptimizer$Cnn.buildData(undefined),
          networkHyperparam: {
            t: 0
          },
          networkLayerHyperparam: undefined,
          allLayerHyperparams: ArraySt$Cnn.map(allLearnRates, (function (learnRate) {
                  return {
                          learnRate: learnRate,
                          beta1: 0,
                          beta2: 0,
                          epsion: 0
                        };
                }))
        };
}

exports.buildNetworkNoOptimizerData = buildNetworkNoOptimizerData;
/* No side effect */
