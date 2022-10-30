'use strict';

var Neural_forward_answer$Gender_analyze = require("./Neural_forward_answer.bs.js");

function createState(param) {
  return {
          weight31: Math.random(),
          weight41: Math.random(),
          weight32: Math.random(),
          weight42: Math.random(),
          weight53: Math.random(),
          weight54: Math.random(),
          bias3: Math.random(),
          bias4: Math.random(),
          bias5: Math.random()
        };
}

function train(state, allSampleData) {
  return state;
}

function _activateFunc(x) {
  return x;
}

function _convert(x) {
  if (x !== 0) {
    if (x !== 1) {
      return /* InValid */2;
    } else {
      return /* Female */1;
    }
  } else {
    return /* Male */0;
  }
}

function forward(state, sampleData) {
  var y3 = Neural_forward_answer$Gender_analyze.forward({
        weight1: state.weight31,
        weight2: state.weight32,
        bias: state.bias3
      }, sampleData);
  var y4 = Neural_forward_answer$Gender_analyze.forward({
        weight1: state.weight41,
        weight2: state.weight42,
        bias: state.bias4
      }, sampleData);
  return Neural_forward_answer$Gender_analyze.forward({
              weight1: state.weight53,
              weight2: state.weight54,
              bias: state.bias5
            }, {
              weight: y3,
              height: y4
            });
}

function inference(state, sampleData) {
  console.log(forward(state, sampleData));
  return _convert(forward(state, sampleData));
}

var state = createState(undefined);

var allSampleData = [
  {
    weight: 50,
    height: 150
  },
  {
    weight: 51,
    height: 149
  },
  {
    weight: 60,
    height: 172
  },
  {
    weight: 90,
    height: 188
  }
];

allSampleData.forEach(function (sampleData) {
      console.log(inference(state, sampleData));
      
    });

exports.createState = createState;
exports.train = train;
exports._activateFunc = _activateFunc;
exports._convert = _convert;
exports.forward = forward;
exports.inference = inference;
exports.allSampleData = allSampleData;
exports.state = state;
/* state Not a pure module */
