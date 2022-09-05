

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Belt_Array from "../../../../../../node_modules/rescript/lib/es6/belt_Array.js";
import * as Neural_forward_answer$Gender_analyze from "./Neural_forward_answer.bs.js";

function length(prim) {
  return prim.length;
}

function map(arr, func) {
  return arr.map(Curry.__1(func));
}

function range(a, b) {
  var result = [];
  for(var i = a; i <= b; ++i){
    result.push(i);
  }
  return result;
}

function reduceOneParam(arr, func, param) {
  return Belt_Array.reduceU(arr, param, func);
}

function reduceOneParami(arr, func, param) {
  var mutableParam = param;
  for(var i = 0 ,i_finish = arr.length; i < i_finish; ++i){
    mutableParam = func(mutableParam, arr[i], i);
  }
  return mutableParam;
}

var ArraySt = {
  length: length,
  map: map,
  range: range,
  reduceOneParam: reduceOneParam,
  reduceOneParami: reduceOneParami
};

function forward(state, feature) {
  var net = feature.height * state.weight1 + feature.weight * state.weight2 + state.bias;
  return [
          net,
          Neural_forward_answer$Gender_analyze._activateFunc(net)
        ];
}

var Neural_forward = {
  forward: forward
};

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

function _activateFunc(x) {
  return x;
}

function _deriv_Linear(x) {
  return 1;
}

function forward$1(state, feature) {
  var match = forward({
        weight1: state.weight13,
        weight2: state.weight23,
        bias: state.bias3
      }, feature);
  var y3 = match[1];
  var match$1 = forward({
        weight1: state.weight14,
        weight2: state.weight24,
        bias: state.bias4
      }, feature);
  var y4 = match$1[1];
  var match$2 = forward({
        weight1: state.weight35,
        weight2: state.weight45,
        bias: state.bias5
      }, {
        weight: y3,
        height: y4
      });
  return [
          [
            match[0],
            match$1[0],
            match$2[0]
          ],
          [
            y3,
            y4,
            match$2[1]
          ]
        ];
}

function _convertLabelToFloat(label) {
  if (label) {
    return 1;
  } else {
    return 0;
  }
}

function train(state, features, labels) {
  return Belt_Array.reduceU(range(0, 999), state, (function (state, epoch) {
                return state;
              }));
}

function inference(state, feature) {
  var match = forward$1(state, feature);
  return match[1][2];
}

var state = createState(undefined);

var features = [
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

var labels = [
  /* Female */1,
  /* Female */1,
  /* Male */0,
  /* Male */0
];

var state$1 = train(state, features, labels);

var featuresForInference = [
  {
    weight: 89,
    height: 190
  },
  {
    weight: 60,
    height: 155
  }
];

featuresForInference.forEach(function (feature) {
      console.log(inference(state$1, feature));
      
    });

export {
  ArraySt ,
  Neural_forward ,
  createState ,
  _activateFunc ,
  _deriv_Linear ,
  forward$1 as forward,
  _convertLabelToFloat ,
  train ,
  inference ,
  features ,
  labels ,
  state$1 as state,
  featuresForInference ,
  
}
/* state Not a pure module */
