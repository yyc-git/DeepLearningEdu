'use strict';

var Caml_int32 = require("rescript/lib/js/caml_int32.js");
var Matrix$Cnn = require("../Matrix.bs.js");
var Vector$Cnn = require("../Vector.bs.js");
var ArraySt$Cnn = require("../ArraySt.bs.js");
var Exception$Cnn = require("../Exception.bs.js");
var ImmutableSparseMap$Cnn = require("../sparse_map/ImmutableSparseMap.bs.js");

var _flattenMatrix = Matrix$Cnn.getData;

function _flatten(inputMap) {
  return Vector$Cnn.create(ImmutableSparseMap$Cnn.reducei(inputMap, (function (vectorData, output, param) {
                    return vectorData.concat(Matrix$Cnn.getData(output));
                  }), []));
}

function forward(state, param, inputMap, param$1) {
  var output = _flatten(inputMap);
  return [
          state,
          output,
          output
        ];
}

function bpDelta(param, param$1, nextLayerDelta, state) {
  var inputHeight = state.inputHeight;
  var inputWidth = state.inputWidth;
  if (Vector$Cnn.length(nextLayerDelta) === Math.imul(Math.imul(state.depthNumber, inputWidth), inputHeight)) {
    
  } else {
    Exception$Cnn.throwErr("error");
  }
  var oneDeltaMapCount = Math.imul(inputWidth, inputHeight);
  var match = Vector$Cnn.reducei(nextLayerDelta, (function (param, value, index) {
            var depthDelta = param[1];
            var totalDepthDelta = param[0];
            if (index !== 0 && Caml_int32.mod_(index, oneDeltaMapCount) === 0) {
              return [
                      ArraySt$Cnn.push(totalDepthDelta, depthDelta),
                      [value]
                    ];
            } else {
              return [
                      totalDepthDelta,
                      ArraySt$Cnn.push(depthDelta, value)
                    ];
            }
          }))([
        [],
        []
      ]);
  var totalDepthDelta = ArraySt$Cnn.push(match[0], match[1]);
  return ImmutableSparseMap$Cnn.createFromArr(ArraySt$Cnn.map(totalDepthDelta, (function (depthDelta) {
                    return Matrix$Cnn.create(inputHeight, inputWidth, depthDelta);
                  })));
}

function backward(param, previousLayerData, nextLayerDelta, state) {
  return [
          bpDelta(undefined, previousLayerData, nextLayerDelta, state),
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

function create(inputWidth, inputHeight, depthNumberOpt, param) {
  var depthNumber = depthNumberOpt !== undefined ? depthNumberOpt : 1;
  return {
          inputWidth: inputWidth,
          inputHeight: inputHeight,
          depthNumber: depthNumber
        };
}

function createLayerData(state, activatorData) {
  return {
          layerName: "flatten",
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

exports._flattenMatrix = _flattenMatrix;
exports._flatten = _flatten;
exports.forward = forward;
exports.bpDelta = bpDelta;
exports.backward = backward;
exports.createGradientDataSum = createGradientDataSum;
exports.addToGradientDataSum = addToGradientDataSum;
exports.update = update;
exports.create = create;
exports.createLayerData = createLayerData;
/* No side effect */
