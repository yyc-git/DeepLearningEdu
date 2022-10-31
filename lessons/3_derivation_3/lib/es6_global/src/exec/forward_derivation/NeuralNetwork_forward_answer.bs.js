

import * as Curry from "../../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Belt_Array from "../../../../../../../node_modules/rescript/lib/es6/belt_Array.js";
import * as Caml_array from "../../../../../../../node_modules/rescript/lib/es6/caml_array.js";

function length(prim) {
  return prim.length;
}

function sliceFrom(arr, index) {
  return arr.slice(index);
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

function range(a, b) {
  var result = [];
  for(var i = a; i <= b; ++i){
    result.push(i);
  }
  return result;
}

function map(arr, func) {
  return arr.map(Curry.__1(func));
}

function mapi(arr, func) {
  return arr.map(Curry.__2(func));
}

function push(arr, value) {
  arr.push(value);
  return arr;
}

function forEach(arr, func) {
  arr.forEach(Curry.__1(func));
  
}

function forEachi(arr, func) {
  arr.forEach(Curry.__2(func));
  
}

var ArraySt = {
  length: length,
  sliceFrom: sliceFrom,
  reduceOneParam: reduceOneParam,
  reduceOneParami: reduceOneParami,
  range: range,
  map: map,
  mapi: mapi,
  push: push,
  forEach: forEach,
  forEachi: forEachi
};

function computeIndex(colCount, rowIndex, colIndex) {
  return Math.imul(rowIndex, colCount) + colIndex | 0;
}

function getRow(colCount, rowIndex, data) {
  return data.slice(Math.imul(rowIndex, colCount), Math.imul(rowIndex + 1 | 0, colCount));
}

function getCol(rowCount, colCount, colIndex, data) {
  return Belt_Array.reduceU(range(0, rowCount - 1 | 0), [], (function (arr, rowIndex) {
                return push(arr, Caml_array.get(data, computeIndex(colCount, rowIndex, colIndex)));
              }));
}

function getValue(rowIndex, colIndex, param) {
  return Caml_array.get(param[2], computeIndex(param[1], rowIndex, colIndex));
}

var MatrixUtils = {
  computeIndex: computeIndex,
  getRow: getRow,
  getCol: getCol,
  getValue: getValue
};

function create(arr) {
  return arr;
}

function push$1(vec, value) {
  return push(vec.slice(0), value);
}

function dot(vec1, vec2) {
  return reduceOneParami(vec1, (function (sum, data1, i) {
                return sum + data1 * Caml_array.get(vec2, i);
              }), 0);
}

function multiply(vec1, vec2) {
  return reduceOneParami(vec1, (function (arr, data1, i) {
                return push(arr, data1 * Caml_array.get(vec2, i));
              }), []);
}

function multiplyScalar(vec, scalar) {
  return Belt_Array.reduceU(vec, [], (function (arr, data) {
                return push(arr, data * scalar);
              }));
}

function add(vec1, vec2) {
  return reduceOneParami(vec1, (function (arr, data1, i) {
                return push(arr, data1 + Caml_array.get(vec2, i));
              }), []);
}

function sub(vec1, vec2) {
  return reduceOneParami(vec1, (function (arr, data1, i) {
                return push(arr, data1 - Caml_array.get(vec2, i));
              }), []);
}

function scalarSub(scalar, vec) {
  return Belt_Array.reduceU(vec, [], (function (arr, data) {
                return push(arr, scalar - data);
              }));
}

function length$1(vec) {
  return vec.length;
}

function transformMatrix(param, vec) {
  var matrixData = param[2];
  var col = param[1];
  return Belt_Array.reduceU(range(0, param[0] - 1 | 0), [], (function (arr, rowIndex) {
                return push(arr, dot(getRow(col, rowIndex, matrixData), vec));
              }));
}

function map$1(vec, func) {
  return vec.map(Curry.__1(func));
}

function mapi$1(vec, func) {
  return vec.map(Curry.__2(func));
}

var forEachi$1 = forEachi;

function reducei(vec, func) {
  return function (param) {
    return reduceOneParami(vec, func, param);
  };
}

function toArray(vec) {
  return vec;
}

var Vector = {
  create: create,
  push: push$1,
  dot: dot,
  multiply: multiply,
  multiplyScalar: multiplyScalar,
  add: add,
  sub: sub,
  scalarSub: scalarSub,
  length: length$1,
  transformMatrix: transformMatrix,
  map: map$1,
  mapi: mapi$1,
  forEachi: forEachi$1,
  reducei: reducei,
  toArray: toArray
};

function getData(param) {
  return param[2];
}

function getColCount(param) {
  return param[1];
}

function getRowCount(param) {
  return param[0];
}

function forEachRow(param, func) {
  return forEach(range(0, param[0] - 1 | 0), Curry.__1(func));
}

function forEachCol(param, func) {
  return forEach(range(0, param[1] - 1 | 0), Curry.__1(func));
}

function create$1(row, col, data) {
  return [
          row,
          col,
          data
        ];
}

function transpose(param) {
  var data = param[2];
  var col = param[1];
  var row = param[0];
  return [
          col,
          row,
          Belt_Array.reduceU(range(0, row - 1 | 0), [], (function (arr, rowIndex) {
                  return Belt_Array.reduceU(range(0, col - 1 | 0), arr, (function (arr, colIndex) {
                                arr[computeIndex(row, colIndex, rowIndex)] = Caml_array.get(data, computeIndex(col, rowIndex, colIndex));
                                return arr;
                              }));
                }))
        ];
}

function multiplyScalar$1(scalar, param) {
  return [
          param[0],
          param[1],
          param[2].map(function (v) {
                return v * scalar;
              })
        ];
}

function map$2(matrix, func) {
  var arr = getData(matrix);
  return [
          getRowCount(matrix),
          getColCount(matrix),
          arr.map(Curry.__1(func))
        ];
}

function mapi$2(matrix, func) {
  var arr = getData(matrix);
  return [
          getRowCount(matrix),
          getColCount(matrix),
          arr.map(Curry.__2(func))
        ];
}

var Matrix = {
  getData: getData,
  getColCount: getColCount,
  getRowCount: getRowCount,
  forEachRow: forEachRow,
  forEachCol: forEachCol,
  create: create$1,
  transpose: transpose,
  multiplyScalar: multiplyScalar$1,
  map: map$2,
  mapi: mapi$2
};

function _createWMatrix(getValueFunc, firstLayerNodeCount, secondLayerNodeCount) {
  var col = firstLayerNodeCount + 1 | 0;
  var arr = range(0, Math.imul(secondLayerNodeCount, col) - 1 | 0);
  return [
          secondLayerNodeCount,
          col,
          arr.map(function (param) {
                return Curry._1(getValueFunc, undefined);
              })
        ];
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

function forward(state, feature) {
  var inputVector = [
    feature.height,
    feature.weight,
    1.0
  ];
  var arr = transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector);
  var layer2OutputVector = arr.map(_activateFunc);
  var arr$1 = transformMatrix(state.wMatrixBetweenLayer2Layer3, push$1(layer2OutputVector, 1.0));
  var layer3OutputVector = arr$1.map(_activateFunc);
  return [
          layer2OutputVector,
          layer3OutputVector
        ];
}

var state = createState(2, 2, 1);

var feature = {
  weight: 50,
  height: 150
};

console.log(forward(state, feature));

export {
  ArraySt ,
  MatrixUtils ,
  Vector ,
  Matrix ,
  _createWMatrix ,
  createState ,
  _activateFunc ,
  forward ,
  state ,
  feature ,
  
}
/* state Not a pure module */
