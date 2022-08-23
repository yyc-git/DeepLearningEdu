

import * as Neural_forward_answer$Gender_analyze from "./Neural_forward_answer.bs.js";

function createState(param) {
  return {
          weight13: Math.random(),
          weight14: Math.random(),
          weight23: Math.random(),
          weight24: Math.random(),
          weight35: Math.random(),
          weight45: Math.random(),
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
        weight1: state.weight13,
        weight2: state.weight23,
        bias: state.bias3
      }, sampleData);
  var y4 = Neural_forward_answer$Gender_analyze.forward({
        weight1: state.weight14,
        weight2: state.weight24,
        bias: state.bias4
      }, sampleData);
  return Neural_forward_answer$Gender_analyze.forward({
              weight1: state.weight35,
              weight2: state.weight45,
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

export {
  createState ,
  train ,
  _activateFunc ,
  _convert ,
  forward ,
  inference ,
  allSampleData ,
  state ,
  
}
/* state Not a pure module */
