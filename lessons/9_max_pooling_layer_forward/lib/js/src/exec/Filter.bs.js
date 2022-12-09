'use strict';

var NP$8_cnn = require("./NP.bs.js");

function create(width, height, depth) {
  return {
          weights: NP$8_cnn.createMatrixMap((function (param) {
                  return (Math.random() * 2 - 1) * 1e-4;
                }), depth, height, width),
          bias: 0
        };
}

function getWeights(state) {
  return state.weights;
}

function getBias(state) {
  return state.bias;
}

exports.create = create;
exports.getWeights = getWeights;
exports.getBias = getBias;
/* No side effect */
