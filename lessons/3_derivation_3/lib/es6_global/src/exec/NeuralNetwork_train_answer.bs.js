

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
import * as Matrix$Gender_analyze from "./Matrix.bs.js";
import * as Vector$Gender_analyze from "./Vector.bs.js";
import * as ArraySt$Gender_analyze from "./ArraySt.bs.js";
import * as MatrixUtils$Gender_analyze from "./MatrixUtils.bs.js";

function _createWMatrix(getValueFunc, firstLayerNodeCount, secondLayerNodeCount) {
  var col = firstLayerNodeCount + 1 | 0;
  return Matrix$Gender_analyze.create(secondLayerNodeCount, col, ArraySt$Gender_analyze.map(ArraySt$Gender_analyze.range(0, Math.imul(secondLayerNodeCount, col) - 1 | 0), (function (param) {
                    return Curry._1(getValueFunc, undefined);
                  })));
}

function createState(layer1NodeCount, layer2NodeCount, layer3NodeCount) {
  return {
          wMatrixBetweenLayer1Layer2: _createWMatrix((function (param) {
                  return 0.1;
                }), layer1NodeCount, layer2NodeCount),
          wMatrixBetweenLayer2Layer3: _createWMatrix((function (param) {
                  return 0.1;
                }), layer2NodeCount, layer3NodeCount)
        };
}

function _activateFunc(x) {
  return 1 / (1 + Math.exp(-x));
}

function _deriv_Sigmoid(x) {
  var fx = _activateFunc(x);
  return fx * (1 - fx);
}

function forward(inputVector, state) {
  var layer2Net = Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector);
  var layer2OutputVector = Vector$Gender_analyze.map(layer2Net, _activateFunc);
  var layer3Net = Vector$Gender_analyze.transformMatrix(state.wMatrixBetweenLayer2Layer3, Vector$Gender_analyze.push(layer2OutputVector, 1.0));
  var layer3OutputVector = Vector$Gender_analyze.map(layer3Net, _activateFunc);
  return [
          [
            layer2Net,
            layer2OutputVector
          ],
          [
            layer3Net,
            layer3OutputVector
          ]
        ];
}

function backward(param, n, label, inputVector, state) {
  var match = param[1];
  var layer3Net = match[0];
  var match$1 = param[0];
  var labelVector = Vector$Gender_analyze.create([label]);
  var layer3Delta = Vector$Gender_analyze.mapi(match[1], (function (layer3OutputValue, i) {
          var d_E_d_value = -2 / n * (Vector$Gender_analyze.getExn(labelVector, i) - layer3OutputValue);
          var d_y_net_value = _deriv_Sigmoid(Vector$Gender_analyze.getExn(layer3Net, i));
          return d_E_d_value * d_y_net_value;
        }));
  var layer2Delta = Vector$Gender_analyze.mapi(match$1[0], (function (layer2NetValue, i) {
          return Vector$Gender_analyze.dot(layer3Delta, MatrixUtils$Gender_analyze.getCol(Matrix$Gender_analyze.getRowCount(state.wMatrixBetweenLayer2Layer3), Matrix$Gender_analyze.getColCount(state.wMatrixBetweenLayer2Layer3), i, Matrix$Gender_analyze.getData(state.wMatrixBetweenLayer2Layer3))) * _deriv_Sigmoid(layer2NetValue);
        }));
  var layer2Gradient = Matrix$Gender_analyze.multiply(Matrix$Gender_analyze.create(Vector$Gender_analyze.length(layer2Delta), 1, layer2Delta), Matrix$Gender_analyze.create(1, Vector$Gender_analyze.length(inputVector), inputVector));
  var layer2OutputVector = Vector$Gender_analyze.push(match$1[1], 1.0);
  var layer3Gradient = Matrix$Gender_analyze.multiply(Matrix$Gender_analyze.create(Vector$Gender_analyze.length(layer3Delta), 1, layer3Delta), Matrix$Gender_analyze.create(1, Vector$Gender_analyze.length(layer2OutputVector), layer2OutputVector));
  return [
          layer2Gradient,
          layer3Gradient
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
  return ArraySt$Gender_analyze.reduceOneParami(labels, (function (result, label, i) {
                return result + Math.pow(label - Caml_array.get(outputs, i), 2.0);
              }), 0) / ArraySt$Gender_analyze.length(labels);
}

function _createInputVector(feature) {
  return Vector$Gender_analyze.create([
              feature.height,
              feature.weight,
              1.0
            ]);
}

function train(state, features, labels) {
  var n = ArraySt$Gender_analyze.length(features);
  return ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, 999), (function (state, epoch) {
                var state$1 = ArraySt$Gender_analyze.reduceOneParami(features, (function (state, feature, i) {
                        var label = Caml_array.get(labels, i) ? 1 : 0;
                        var inputVector = _createInputVector(feature);
                        var match = backward(forward(inputVector, state), n, label, inputVector, state);
                        return {
                                wMatrixBetweenLayer1Layer2: Matrix$Gender_analyze.sub(state.wMatrixBetweenLayer1Layer2, Matrix$Gender_analyze.multiplyScalar(0.1, match[0])),
                                wMatrixBetweenLayer2Layer3: Matrix$Gender_analyze.sub(state.wMatrixBetweenLayer2Layer3, Matrix$Gender_analyze.multiplyScalar(0.1, match[1]))
                              };
                      }), state);
                if (epoch % 10 === 0) {
                  console.log([
                        "loss: ",
                        _computeLoss(ArraySt$Gender_analyze.map(labels, _convertLabelToFloat), ArraySt$Gender_analyze.map(features, (function (feature) {
                                    var inputVector = _createInputVector(feature);
                                    var match = forward(inputVector, state$1);
                                    return Vector$Gender_analyze.getExn(match[1][1], 0);
                                  })))
                      ]);
                  return state$1;
                } else {
                  return state$1;
                }
              }), state);
}

var state = createState(2, 2, 1);

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

export {
  _createWMatrix ,
  createState ,
  _activateFunc ,
  _deriv_Sigmoid ,
  forward ,
  backward ,
  _convertLabelToFloat ,
  _computeLoss ,
  _createInputVector ,
  train ,
  features ,
  labels ,
  state$1 as state,
  
}
/* state Not a pure module */
