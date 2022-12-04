'use strict';

var Curry = require("rescript/lib/js/curry.js");
var NP$Cnn = require("../NP.bs.js");
var InitValue$Cnn = require("../InitValue.bs.js");
var ImmutableSparseMap$Cnn = require("../sparse_map/ImmutableSparseMap.bs.js");

function _createAdamData(depth, height, width) {
  return {
          vWeight: NP$Cnn.zeroMatrixMap(depth, height, width),
          vBias: 0.0,
          sWeight: NP$Cnn.zeroMatrixMap(depth, height, width),
          sBias: 0.0
        };
}

function create(param, width, height, depth) {
  var fanOut = param[3];
  var fanIn = param[2];
  var randomFunc = param[1];
  var initValueMethodFunc = param[0];
  return {
          weights: NP$Cnn.createMatrixMap((function (param) {
                  return Curry._3(initValueMethodFunc, randomFunc, fanIn, fanOut);
                }), depth, height, width),
          bias: InitValue$Cnn.constant(0.0),
          adamData: _createAdamData(depth, height, width)
        };
}

function getWeights(state) {
  return state.weights;
}

function getBias(state) {
  return state.bias;
}

function update(state, param, param$1) {
  var match = param$1[2];
  var epsion = match.epsion;
  var beta2 = match.beta2;
  var beta1 = match.beta1;
  var learnRate = match.learnRate;
  var t = param$1[1].t;
  var match$1 = param$1[0];
  var updateWeightFunc = match$1.updateWeightFunc;
  var gradientDataSumForWeight = param[1];
  var miniBatchSize = param[0];
  var match$2 = state.adamData;
  var sWeight = match$2.sWeight;
  var vWeight = match$2.vWeight;
  var match$3 = ImmutableSparseMap$Cnn.reducei(state.weights, (function (param, weight, i) {
          var gradientDataSumForWeight$1 = ImmutableSparseMap$Cnn.getExn(gradientDataSumForWeight, i);
          var vWeight$1 = ImmutableSparseMap$Cnn.getExn(vWeight, i);
          var sWeight$1 = ImmutableSparseMap$Cnn.getExn(sWeight, i);
          var match = Curry._5(updateWeightFunc, weight, [
                learnRate,
                t,
                [
                  beta1,
                  beta2,
                  epsion
                ]
              ], gradientDataSumForWeight$1, miniBatchSize, [
                vWeight$1,
                sWeight$1
              ]);
          var match$1 = match[1];
          return [
                  ImmutableSparseMap$Cnn.set(param[0], i, match[0]),
                  ImmutableSparseMap$Cnn.set(param[1], i, match$1[0]),
                  ImmutableSparseMap$Cnn.set(param[2], i, match$1[1])
                ];
        }), [
        ImmutableSparseMap$Cnn.createEmpty(undefined, undefined),
        ImmutableSparseMap$Cnn.createEmpty(undefined, undefined),
        ImmutableSparseMap$Cnn.createEmpty(undefined, undefined)
      ]);
  var match$4 = Curry._5(match$1.updateValueFunc, state.bias, [
        learnRate,
        t,
        [
          beta1,
          beta2,
          epsion
        ]
      ], match$2.vBias, match$2.sBias, param[2] / miniBatchSize);
  var match$5 = match$4[1];
  return {
          weights: match$3[0],
          bias: match$4[0],
          adamData: {
            vWeight: match$3[1],
            vBias: match$5[0],
            sWeight: match$3[2],
            sBias: match$5[1]
          }
        };
}

exports._createAdamData = _createAdamData;
exports.create = create;
exports.getWeights = getWeights;
exports.getBias = getBias;
exports.update = update;
/* No side effect */
