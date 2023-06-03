

import * as OptimizerUtils$Cnn from "./OptimizerUtils.bs.js";

function update(data, param, vt_1, st_1, gradient) {
  return [
          data - param[0] * gradient,
          [
            vt_1,
            st_1
          ]
        ];
}

function updateWeight(param, param$1, param$2, param$3, param$4) {
  return OptimizerUtils$Cnn.updateWeight(update, param, param$1, param$2, param$3, param$4);
}

function buildData(param) {
  return {
          updateNetworkHyperparamFunc: (function (networkHyperparam) {
              return networkHyperparam;
            }),
          updateValueFunc: update,
          updateWeightFunc: updateWeight
        };
}

export {
  update ,
  updateWeight ,
  buildData ,
  
}
/* No side effect */
