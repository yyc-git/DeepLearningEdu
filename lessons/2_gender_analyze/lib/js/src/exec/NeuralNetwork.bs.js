'use strict';


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
  return 1;
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
  return 1;
}

function inference(state, sampleData) {
  return 1;
}

createState(undefined);

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
      console.log(1);
      
    });

var state = 1;

exports.createState = createState;
exports.train = train;
exports._activateFunc = _activateFunc;
exports._convert = _convert;
exports.forward = forward;
exports.inference = inference;
exports.allSampleData = allSampleData;
exports.state = state;
/*  Not a pure module */
