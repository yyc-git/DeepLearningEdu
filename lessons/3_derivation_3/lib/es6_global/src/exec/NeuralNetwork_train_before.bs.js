

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Belt_Array from "../../../../../../node_modules/rescript/lib/es6/belt_Array.js";
import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";

function createState(param) {
  return {
          weight1: Math.random(),
          weight2: Math.random(),
          bias: Math.random()
        };
}

function _activateFunc(x) {
  return 1 / (1 + Math.exp(-x));
}

function forward(state, sampleData) {
  return _activateFunc(sampleData.height * state.weight1 + sampleData.weight * state.weight2 + state.bias);
}

var Neural_forward_answer = {
  createState: createState,
  _activateFunc: _activateFunc,
  forward: forward
};

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

function forward$1(state, feature) {
  var net = feature.height * state.weight1 + feature.weight * state.weight2 + state.bias;
  return [
          net,
          _activateFunc(net)
        ];
}

var Neural_forward = {
  forward: forward$1
};

function createState$1(param) {
  return {
          weight31: 0.1,
          weight41: 0.1,
          weight32: 0.1,
          weight42: 0.1,
          weight53: 0.1,
          weight54: 0.1,
          bias3: 0.1,
          bias4: 0.1,
          bias5: 0.1
        };
}

function _activateFunc$1(x) {
  return 1 / (1 + Math.exp(-x));
}

function _deriv_Sigmoid(x) {
  var fx = _activateFunc$1(x);
  return fx * (1 - fx);
}

function forward$2(state, feature) {
  var match = forward$1({
        weight1: state.weight31,
        weight2: state.weight32,
        bias: state.bias3
      }, feature);
  var y3 = match[1];
  var match$1 = forward$1({
        weight1: state.weight41,
        weight2: state.weight42,
        bias: state.bias4
      }, feature);
  var y4 = match$1[1];
  var match$2 = forward$1({
        weight1: state.weight53,
        weight2: state.weight54,
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
                        var match = forward$2(state, feature);
                        var match$1 = match[1];
                        var match$2 = match[0];
                        var net5 = match$2[2];
                        var net4 = match$2[1];
                        var net3 = match$2[0];
                        var d_E_d_y5 = -2 / n * (label - match$1[2]);
                        var d_y5_d_w53 = match$1[0] * _deriv_Sigmoid(net5);
                        var d_y5_d_w54 = match$1[1] * _deriv_Sigmoid(net5);
                        var d_y5_d_b5 = _deriv_Sigmoid(net5);
                        var d_y5_d_y3 = state.weight53 * _deriv_Sigmoid(net5);
                        var d_y5_d_y4 = state.weight54 * _deriv_Sigmoid(net5);
                        var d_y3_d_w31 = x1 * _deriv_Sigmoid(net3);
                        var d_y3_d_w32 = x2 * _deriv_Sigmoid(net3);
                        var d_y3_d_b3 = _deriv_Sigmoid(net3);
                        var d_y4_d_w41 = x1 * _deriv_Sigmoid(net4);
                        var d_y4_d_w42 = x2 * _deriv_Sigmoid(net4);
                        var d_y4_d_b4 = _deriv_Sigmoid(net4);
                        return {
                                weight31: state.weight31 - 0.1 * d_E_d_y5 * d_y5_d_y3 * d_y3_d_w31,
                                weight41: state.weight41 - 0.1 * d_E_d_y5 * d_y5_d_y4 * d_y4_d_w41,
                                weight32: state.weight41 - 0.1 * d_E_d_y5 * d_y5_d_y3 * d_y3_d_w32,
                                weight42: state.weight42 - 0.1 * d_E_d_y5 * d_y5_d_y4 * d_y4_d_w42,
                                weight53: state.weight53 - 0.1 * d_E_d_y5 * d_y5_d_w53,
                                weight54: state.weight54 - 0.1 * d_E_d_y5 * d_y5_d_w54,
                                bias3: state.bias3 - 0.1 * d_E_d_y5 * d_y5_d_y3 * d_y3_d_b3,
                                bias4: state.bias4 - 0.1 * d_E_d_y5 * d_y5_d_y4 * d_y4_d_b4,
                                bias5: state.bias5 - 0.1 * d_E_d_y5 * d_y5_d_b5
                              };
                      }), state);
                if (epoch % 10 === 0) {
                  console.log([
                        "loss: ",
                        _computeLoss(labels.map(_convertLabelToFloat), features.map(function (feature) {
                                  var match = forward$2(state$1, feature);
                                  return match[1][2];
                                }))
                      ]);
                  return state$1;
                } else {
                  return state$1;
                }
              }));
}

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

var state = train({
      weight31: 0.1,
      weight41: 0.1,
      weight32: 0.1,
      weight42: 0.1,
      weight53: 0.1,
      weight54: 0.1,
      bias3: 0.1,
      bias4: 0.1,
      bias5: 0.1
    }, features, labels);

export {
  Neural_forward_answer ,
  ArraySt ,
  Neural_forward ,
  createState$1 as createState,
  _activateFunc$1 as _activateFunc,
  _deriv_Sigmoid ,
  forward$2 as forward,
  _convertLabelToFloat ,
  _computeLoss ,
  train ,
  features ,
  labels ,
  state ,
  
}
/* state Not a pure module */
