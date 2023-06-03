

import * as Matrix$Cnn from "../Matrix.bs.js";
import * as Vector$Cnn from "../Vector.bs.js";
import * as ImmutableSparseMap$Cnn from "../sparse_map/ImmutableSparseMap.bs.js";

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

export {
  forward ,
  bpDelta ,
  backward ,
  createGradientDataSum ,
  addToGradientDataSum ,
  update ,
  create ,
  createLayerData ,
  
}
/* No side effect */
