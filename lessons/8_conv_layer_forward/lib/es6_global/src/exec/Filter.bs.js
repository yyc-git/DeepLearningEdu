

import * as NP$8_cnn from "./NP.bs.js";
import * as Matrix$8_cnn from "./Matrix.bs.js";
import * as ImmutableSparseMap$8_cnn from "./sparse_map/ImmutableSparseMap.bs.js";

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

function update(state, learnRate) {
  return {
          weights: ImmutableSparseMap$8_cnn.mapi(state.weights, (function (weights, i) {
                  return Matrix$8_cnn.sub(weights, Matrix$8_cnn.multiplyScalar(learnRate, ImmutableSparseMap$8_cnn.getExn(state.weightGradients, i)));
                })),
          bias: state.bias - learnRate * state.biasGradient,
          weightGradients: state.weightGradients,
          biasGradient: state.biasGradient
        };
}

export {
  create ,
  getWeights ,
  getBias ,
  update ,
  
}
/* No side effect */
