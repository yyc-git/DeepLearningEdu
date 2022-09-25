

import * as Vector$Gender_analyze from "./Vector.bs.js";
import * as ArraySt$Gender_analyze from "./ArraySt.bs.js";

function createState(inputCount) {
  return {
          wVector: Vector$Gender_analyze.create(ArraySt$Gender_analyze.map(ArraySt$Gender_analyze.range(0, (inputCount + 1 | 0) - 1 | 0), (function (param) {
                      return Math.random();
                    })))
        };
}

function train(state, sampleData) {
  return {
          wVector: Vector$Gender_analyze.create([
                1.0,
                -2.0,
                -49.0
              ])
        };
}

function _activateFunc(x) {
  return 1 / (1 + Math.exp(-x));
}

function forward(state, sampleData) {
  var inputVector = Vector$Gender_analyze.create([
        sampleData.height,
        sampleData.weight,
        1.0
      ]);
  return _activateFunc(Vector$Gender_analyze.dot(state.wVector, inputVector));
}

var state = createState(2);

console.log(forward(train(state, {
              weight: 50,
              height: 150
            }), {
          weight: 50,
          height: 150
        }));

export {
  createState ,
  train ,
  _activateFunc ,
  forward ,
  state ,
  
}
/* state Not a pure module */
