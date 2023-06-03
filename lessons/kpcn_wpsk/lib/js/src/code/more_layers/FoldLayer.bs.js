'use strict';

var Matrix$Cnn = require("../Matrix.bs.js");
var Vector$Cnn = require("../Vector.bs.js");
var ImmutableSparseMap$Cnn = require("../sparse_map/ImmutableSparseMap.bs.js");

function forward(state, param, input, param$1) {
  var output = ImmutableSparseMap$Cnn.set(ImmutableSparseMap$Cnn.createEmpty(undefined, undefined), 0, Matrix$Cnn.create(state.outputHeight, state.outputWidth, Vector$Cnn.toArray(input)));
  return [
          state,
          output,
          output
        ];
}

function bpDelta(param, param$1, param$2, param$3) {
  
}

function backward(param, previousLayerData, nextLayerDelta, state) {
  return [
          undefined,
          undefined
        ];
}

function createGradientDataSum(state) {
  
}

function addToGradientDataSum(param, param$1) {
  
}

function update(state, param, param$1) {
  return state;
}

function create(outputWidth, outputHeight) {
  return {
          outputWidth: outputWidth,
          outputHeight: outputHeight
        };
}

function createLayerData(state, activatorData) {
  return {
          layerName: "fold",
          state: state,
          forward: forward,
          backward: backward,
          update: update,
          createGradientDataSum: createGradientDataSum,
          addToGradientDataSum: addToGradientDataSum,
          activatorData: activatorData,
          getWeight: (function (state) {
              
            }),
          getBias: (function (state) {
              
            })
        };
}

exports.forward = forward;
exports.bpDelta = bpDelta;
exports.backward = backward;
exports.createGradientDataSum = createGradientDataSum;
exports.addToGradientDataSum = addToGradientDataSum;
exports.update = update;
exports.create = create;
exports.createLayerData = createLayerData;
/* No side effect */
