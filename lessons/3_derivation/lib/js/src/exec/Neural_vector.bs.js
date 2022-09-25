'use strict';


function createState(param) {
  return {
          weight1: Math.random(),
          weight2: Math.random(),
          bias: Math.random()
        };
}

function train(state, sampleData) {
  return {
          weight1: 1.0,
          weight2: -2.0,
          bias: -49.0
        };
}

function _activateFunc(x) {
  return 1 / (1 + Math.exp(-x));
}

function forward(state, sampleData) {
  return _activateFunc(sampleData.height * state.weight1 + sampleData.weight * state.weight2 + state.bias);
}

var state = createState(undefined);

console.log(forward({
          weight1: 1.0,
          weight2: -2.0,
          bias: -49.0
        }, {
          weight: 50,
          height: 150
        }));

exports.createState = createState;
exports.train = train;
exports._activateFunc = _activateFunc;
exports.forward = forward;
exports.state = state;
/* state Not a pure module */
