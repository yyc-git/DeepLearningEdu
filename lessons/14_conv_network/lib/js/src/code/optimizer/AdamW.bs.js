'use strict';

var OptimizerUtils$Cnn = require("./OptimizerUtils.bs.js");

function update(data, param, vt_1, st_1, gradient) {
  var match = param[2];
  var beta2 = match[1];
  var beta1 = match[0];
  var t = param[1];
  var vt = vt_1 * beta1 + (1 - beta1) * gradient;
  var st = st_1 * beta2 + (1 - beta2) * gradient * gradient;
  var vBiasCorrect = vt / (1 - Math.pow(beta1, t));
  var sBiasCorrect = st / (1 - Math.pow(beta2, t));
  return [
          data - param[0] * vBiasCorrect / (Math.sqrt(sBiasCorrect) + match[2]),
          [
            vt,
            st
          ]
        ];
}

function updateWeight(param, param$1, param$2, param$3, param$4) {
  return OptimizerUtils$Cnn.updateWeight(update, param, param$1, param$2, param$3, param$4);
}

function increaseT(t) {
  return t + 1 | 0;
}

function buildData(param) {
  return {
          updateNetworkHyperparamFunc: (function (networkHyperparam) {
              return {
                      t: networkHyperparam.t + 1 | 0
                    };
            }),
          updateValueFunc: update,
          updateWeightFunc: updateWeight
        };
}

exports.update = update;
exports.updateWeight = updateWeight;
exports.increaseT = increaseT;
exports.buildData = buildData;
/* No side effect */
