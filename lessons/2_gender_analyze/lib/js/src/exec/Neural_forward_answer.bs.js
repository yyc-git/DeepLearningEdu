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
  return x;
}

function _convert(x) {
  if (x === 0) {
    return /* Male */0;
  }
  if (x === 1) {
    return /* Female */1;
  }
  throw {
        RE_EXN_ID: "Match_failure",
        _1: [
          "Neural_forward_answer.res",
          33,
          2
        ],
        Error: new Error()
      };
}

function forward(state, sampleData) {
  return sampleData.height * state.weight1 + sampleData.weight * state.weight2 + state.bias;
}

function inference(state, sampleData) {
  return _convert(forward(state, sampleData));
}

var state = createState(undefined);

var gender = _convert(forward({
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
exports._convert = _convert;
exports.forward = forward;
exports.inference = inference;
exports.state = state;
exports.gender = gender;
/* state Not a pure module */
