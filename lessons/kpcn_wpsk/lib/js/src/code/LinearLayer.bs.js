'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Matrix$Cnn = require("./Matrix.bs.js");
var Random$Cnn = require("./Random.bs.js");
var Vector$Cnn = require("./Vector.bs.js");
var ArraySt$Cnn = require("./ArraySt.bs.js");
var Caml_option = require("rescript/lib/js/caml_option.js");
var OptionSt$Cnn = require("./OptionSt.bs.js");
var InitValue$Cnn = require("./more/InitValue.bs.js");
var DebugUtils$Cnn = require("./DebugUtils.bs.js");
var MatrixUtils$Cnn = require("./MatrixUtils.bs.js");

function getOutputNumber(outputVector) {
  return Vector$Cnn.reducei(outputVector, (function (param, value, index) {
                    var maxValue = param[0];
                    if (value >= maxValue) {
                      return [
                              value,
                              index
                            ];
                    } else {
                      return [
                              maxValue,
                              param[1]
                            ];
                    }
                  }))([
                0,
                undefined
              ])[1];
}

function _createWeight(initValueMethodFunc, randomFunc, inLinearLayerNodeCount, outLinearLayerNodeCount) {
  return Matrix$Cnn.create(outLinearLayerNodeCount, inLinearLayerNodeCount, ArraySt$Cnn.map(ArraySt$Cnn.range(0, Math.imul(outLinearLayerNodeCount, inLinearLayerNodeCount) - 1 | 0), (function (param) {
                    return Curry._3(initValueMethodFunc, randomFunc, inLinearLayerNodeCount, outLinearLayerNodeCount);
                  })));
}

function _createBias(outLinearLayerNodeCount) {
  return Vector$Cnn.create(ArraySt$Cnn.map(ArraySt$Cnn.range(0, outLinearLayerNodeCount - 1 | 0), (function (param) {
                    return InitValue$Cnn.constant(0.0);
                  })));
}

function _createZeroWeight(inLinearLayerNodeCount, outLinearLayerNodeCount) {
  return Matrix$Cnn.create(outLinearLayerNodeCount, inLinearLayerNodeCount, ArraySt$Cnn.map(ArraySt$Cnn.range(0, Math.imul(outLinearLayerNodeCount, inLinearLayerNodeCount) - 1 | 0), (function (param) {
                    return 0.0;
                  })));
}

function _createZeroBias(outLinearLayerNodeCount) {
  return Vector$Cnn.create(ArraySt$Cnn.map(ArraySt$Cnn.range(0, outLinearLayerNodeCount - 1 | 0), (function (param) {
                    return 0.0;
                  })));
}

function getNodeCount(param) {
  var weight = param.weight;
  return [
          Matrix$Cnn.getColCount(weight),
          Matrix$Cnn.getRowCount(weight)
        ];
}

function forward(state, activatorData, input, param) {
  var match = OptionSt$Cnn.getExn(activatorData);
  var net = Vector$Cnn.add(Vector$Cnn.transformMatrix(state.weight, input), state.bias);
  var output = Curry._1(match.forwardNet, net);
  DebugUtils$Cnn.checkOutputVectorExplosion(output);
  return [
          state,
          net,
          output
        ];
}

function createGradientDataSum(state) {
  var match = getNodeCount(state);
  var outLinearLayerNodeCount = match[1];
  return [
          _createZeroWeight(match[0], outLinearLayerNodeCount),
          _createZeroBias(outLinearLayerNodeCount)
        ];
}

function bpDelta(previousLayerActivatorData, param, nextLayerDelta, state) {
  var match = OptionSt$Cnn.getExn(previousLayerActivatorData);
  var backward = match.backward;
  var previousLayerNet = OptionSt$Cnn.getExn(param[1]);
  var weight = state.weight;
  return Vector$Cnn.mapi(previousLayerNet, (function (previousLayerNetValue, j) {
                return Curry._1(backward, previousLayerNetValue) * Vector$Cnn.dot(nextLayerDelta, MatrixUtils$Cnn.getCol(Matrix$Cnn.getRowCount(weight), Matrix$Cnn.getColCount(weight), j, Matrix$Cnn.getData(weight)));
              }));
}

function computeGradient(input, delta, param) {
  return [
          Matrix$Cnn.multiply(Matrix$Cnn.create(Vector$Cnn.length(delta), 1, delta), Matrix$Cnn.create(1, Vector$Cnn.length(input), input)),
          delta
        ];
}

function backward(previousLayerActivatorData, previousLayerData, nextLayerDelta, state) {
  var currentLayerDelta = bpDelta(previousLayerActivatorData, previousLayerData, nextLayerDelta, state);
  var previousLayerOutput = OptionSt$Cnn.getExn(previousLayerData[0]);
  return [
          currentLayerDelta,
          computeGradient(previousLayerOutput, nextLayerDelta, state)
        ];
}

function addToGradientDataSum(gradientDataSum, gradientData) {
  var match = OptionSt$Cnn.getExn(gradientDataSum);
  var match$1 = OptionSt$Cnn.getExn(gradientData);
  return [
          Matrix$Cnn.add(match[0], match$1[0]),
          Vector$Cnn.add(match[1], match$1[1])
        ];
}

function update(state, param, param$1) {
  var miniBatchSize = param$1[0];
  var match = param[2];
  var epsion = match.epsion;
  var beta2 = match.beta2;
  var beta1 = match.beta1;
  var learnRate = match.learnRate;
  var t = param[1].t;
  var match$1 = param[0];
  var updateValueFunc = match$1.updateValueFunc;
  var match$2 = OptionSt$Cnn.getExn(param$1[1]);
  var gradientDataSumForBias = match$2[1];
  var match$3 = state.adamData;
  var sBias = match$3.sBias;
  var vBias = match$3.vBias;
  var match$4 = Curry._5(match$1.updateWeightFunc, state.weight, [
        learnRate,
        t,
        [
          beta1,
          beta2,
          epsion
        ]
      ], match$2[0], miniBatchSize, [
        match$3.vWeight,
        match$3.sWeight
      ]);
  var match$5 = match$4[1];
  var match$6 = Vector$Cnn.reducei(state.bias, (function (param, biasValue, index) {
            var match = param[1];
            var match$1 = Curry._5(updateValueFunc, biasValue, [
                  learnRate,
                  t,
                  [
                    beta1,
                    beta2,
                    epsion
                  ]
                ], Vector$Cnn.getExn(vBias, index), Vector$Cnn.getExn(sBias, index), Vector$Cnn.getExn(gradientDataSumForBias, index) / miniBatchSize);
            var match$2 = match$1[1];
            return [
                    ArraySt$Cnn.push(param[0], match$1[0]),
                    [
                      ArraySt$Cnn.push(match[0], match$2[0]),
                      ArraySt$Cnn.push(match[1], match$2[1])
                    ]
                  ];
          }))([
        [],
        [
          [],
          []
        ]
      ]);
  var match$7 = match$6[1];
  getNodeCount(state);
  return {
          weight: match$4[0],
          bias: Vector$Cnn.create(match$6[0]),
          adamData: {
            vWeight: match$5[0],
            vBias: Vector$Cnn.create(match$7[0]),
            sWeight: match$5[1],
            sBias: Vector$Cnn.create(match$7[1])
          }
        };
}

function _createAdamData(inLinearLayerNodeCount, outLinearLayerNodeCount) {
  return {
          vWeight: _createZeroWeight(inLinearLayerNodeCount, outLinearLayerNodeCount),
          vBias: _createZeroBias(outLinearLayerNodeCount),
          sWeight: _createZeroWeight(inLinearLayerNodeCount, outLinearLayerNodeCount),
          sBias: _createZeroBias(outLinearLayerNodeCount)
        };
}

function create(inLinearLayerNodeCount, outLinearLayerNodeCount, initValueMethodOpt, randomFuncOpt, param) {
  var initValueMethod = initValueMethodOpt !== undefined ? initValueMethodOpt : Random$Cnn.random;
  var randomFunc = randomFuncOpt !== undefined ? randomFuncOpt : (function (prim) {
        return Math.random();
      });
  return {
          weight: _createWeight(initValueMethod, randomFunc, inLinearLayerNodeCount, outLinearLayerNodeCount),
          bias: _createBias(outLinearLayerNodeCount),
          adamData: _createAdamData(inLinearLayerNodeCount, outLinearLayerNodeCount)
        };
}

function createLayerData(state, activatorData) {
  return {
          layerName: "linear",
          state: state,
          forward: forward,
          backward: backward,
          update: update,
          createGradientDataSum: createGradientDataSum,
          addToGradientDataSum: addToGradientDataSum,
          activatorData: activatorData,
          getWeight: (function (state) {
              return Caml_option.some(state.weight);
            }),
          getBias: (function (state) {
              return Caml_option.some(state.bias);
            })
        };
}

exports.getOutputNumber = getOutputNumber;
exports._createWeight = _createWeight;
exports._createBias = _createBias;
exports._createZeroWeight = _createZeroWeight;
exports._createZeroBias = _createZeroBias;
exports.getNodeCount = getNodeCount;
exports.forward = forward;
exports.createGradientDataSum = createGradientDataSum;
exports.bpDelta = bpDelta;
exports.computeGradient = computeGradient;
exports.backward = backward;
exports.addToGradientDataSum = addToGradientDataSum;
exports.update = update;
exports._createAdamData = _createAdamData;
exports.create = create;
exports.createLayerData = createLayerData;
/* No side effect */
