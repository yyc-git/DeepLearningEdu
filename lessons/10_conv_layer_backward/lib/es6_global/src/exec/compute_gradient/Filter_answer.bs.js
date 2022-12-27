

import * as NP$8_cnn from "../NP.bs.js";

function create(width, height, depth) {
  return {
          weights: NP$8_cnn.createMatrixMap((function (param) {
                  return (Math.random() * 2 - 1) * 1e-4;
                }), depth, height, width),
          bias: 0,
          weightGradients: NP$8_cnn.zeroMatrixMap(depth, height, width),
          biasGradient: 0
        };
}

function getWeights(state) {
  return state.weights;
}

function getBias(state) {
  return state.bias;
}

export {
  create ,
  getWeights ,
  getBias ,
  
}
/* No side effect */
