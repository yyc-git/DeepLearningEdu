

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Js_math from "../../../../../../node_modules/rescript/lib/es6/js_math.js";
import * as Belt_Array from "../../../../../../node_modules/rescript/lib/es6/belt_Array.js";
import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
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
  return 1 / (1 + Math.exp(-x));
}

function _deriv_Sigmoid(x) {
  var fx = _activateFunc(x);
  return fx * (1 - fx);
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

function _computeLoss(labels, outputs) {
  return reduceOneParami(labels, (function (result, label, i) {
                return result + Math.pow(label - Caml_array.get(outputs, i), 2.0);
              }), 0) / labels.length;
}

function train(state, features, labels) {
  var n = features.length;
  return Belt_Array.reduceU(range(0, 999), state, (function (state, epoch) {
                var state$1 = reduceOneParami(features, (function (state, feature, i) {
                        var label = Caml_array.get(labels, i) ? 1 : 0;
                        var x1 = feature.weight;
                        var x2 = feature.height;
                        var match = forward$1(state, feature);
                        var match$1 = match[1];
                        var match$2 = match[0];
                        var net5 = match$2[2];
                        var net4 = match$2[1];
                        var net3 = match$2[0];
                        var d_E_d_y5 = -2 / n * (label - match$1[2]);
                        var d_y5_d_w35 = match$1[0] * _deriv_Sigmoid(net5);
                        var d_y5_d_w45 = match$1[1] * _deriv_Sigmoid(net5);
                        var d_y5_d_b5 = _deriv_Sigmoid(net5);
                        var d_y5_d_y3 = state.weight35 * _deriv_Sigmoid(net5);
                        var d_y5_d_y4 = state.weight45 * _deriv_Sigmoid(net5);
                        var d_y3_d_w13 = x1 * _deriv_Sigmoid(net3);
                        var d_y3_d_w23 = x2 * _deriv_Sigmoid(net3);
                        var d_y3_d_b3 = _deriv_Sigmoid(net3);
                        var d_y4_d_w14 = x1 * _deriv_Sigmoid(net4);
                        var d_y4_d_w24 = x2 * _deriv_Sigmoid(net4);
                        var d_y4_d_b4 = _deriv_Sigmoid(net4);
                        return {
                                weight13: state.weight13 - 0.1 * d_E_d_y5 * d_y5_d_y3 * d_y3_d_w13,
                                weight14: state.weight14 - 0.1 * d_E_d_y5 * d_y5_d_y4 * d_y4_d_w14,
                                weight23: state.weight14 - 0.1 * d_E_d_y5 * d_y5_d_y3 * d_y3_d_w23,
                                weight24: state.weight24 - 0.1 * d_E_d_y5 * d_y5_d_y4 * d_y4_d_w24,
                                weight35: state.weight35 - 0.1 * d_E_d_y5 * d_y5_d_w35,
                                weight45: state.weight45 - 0.1 * d_E_d_y5 * d_y5_d_w45,
                                bias3: state.bias3 - 0.1 * d_E_d_y5 * d_y5_d_y3 * d_y3_d_b3,
                                bias4: state.bias4 - 0.1 * d_E_d_y5 * d_y5_d_y4 * d_y4_d_b4,
                                bias5: state.bias5 - 0.1 * d_E_d_y5 * d_y5_d_b5
                              };
                      }), state);
                if (epoch % 10 === 0) {
                  console.log([
                        "loss: ",
                        _computeLoss(labels.map(_convertLabelToFloat), features.map(function (feature) {
                                  var match = forward$1(state$1, feature);
                                  return match[1][2];
                                }))
                      ]);
                  return state$1;
                } else {
                  return state$1;
                }
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

function _mean(values) {
  return Belt_Array.reduceU(values, 0, (function (sum, value) {
                return sum + value;
              })) / values.length;
}

function _zeroMean(features) {
  var weightMean = Js_math.floor(_mean(features.map(function (feature) {
                return feature.weight;
              })));
  var heightMean = Js_math.floor(_mean(features.map(function (feature) {
                return feature.height;
              })));
  return features.map(function (feature) {
              return {
                      weight: feature.weight - weightMean,
                      height: feature.height - heightMean
                    };
            });
}

var features$1 = _zeroMean(features);

var state$1 = train(state, features$1, labels);

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

var __x = _zeroMean(featuresForInference);

__x.forEach(function (feature) {
      console.log(inference(state$1, feature));
      
    });

export {
  ArraySt ,
  Neural_forward ,
  createState ,
  _activateFunc ,
  _deriv_Sigmoid ,
  forward$1 as forward,
  _convertLabelToFloat ,
  _computeLoss ,
  train ,
  inference ,
  labels ,
  _mean ,
  _zeroMean ,
  features$1 as features,
  state$1 as state,
  featuresForInference ,
  
}
/* state Not a pure module */
