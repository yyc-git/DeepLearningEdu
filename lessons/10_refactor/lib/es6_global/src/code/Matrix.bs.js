

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
import * as Vector$Cnn from "./Vector.bs.js";
import * as ArraySt$Cnn from "./ArraySt.bs.js";
import * as Exception$Cnn from "./Exception.bs.js";
import * as MatrixUtils$Cnn from "./MatrixUtils.bs.js";

function getData(param) {
  return param[2];
}

function getColCount(param) {
  return param[1];
}

function getRowCount(param) {
  return param[0];
}

function map(matrix, func) {
  return [
          getRowCount(matrix),
          getColCount(matrix),
          ArraySt$Cnn.map(getData(matrix), func)
        ];
}

function mapi(matrix, func) {
  return [
          getRowCount(matrix),
          getColCount(matrix),
          ArraySt$Cnn.mapi(getData(matrix), func)
        ];
}

function forEachRow(param, func) {
  return ArraySt$Cnn.forEach(ArraySt$Cnn.range(0, param[0] - 1 | 0), Curry.__1(func));
}

function forEachCol(param, func) {
  return ArraySt$Cnn.forEach(ArraySt$Cnn.range(0, param[1] - 1 | 0), Curry.__1(func));
}

function create(row, col, data) {
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
          ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, row - 1 | 0), (function (arr, rowIndex) {
                  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, col - 1 | 0), (function (arr, colIndex) {
                                arr[MatrixUtils$Cnn.computeIndex(row, colIndex, rowIndex)] = Caml_array.get(data, MatrixUtils$Cnn.computeIndex(col, rowIndex, colIndex));
                                return arr;
                              }), arr);
                }), [])
        ];
}

function multiply(param, param$1) {
  var data2 = param$1[2];
  var col2 = param$1[1];
  var row2 = param$1[0];
  var data1 = param[2];
  var col1 = param[1];
  var row1 = param[0];
  if (col1 !== row2) {
    return Exception$Cnn.throwErr("error");
  } else {
    return [
            row1,
            col2,
            ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, row1 - 1 | 0), (function (arr, rowIndex) {
                    var row = MatrixUtils$Cnn.getRow(col1, rowIndex, data1);
                    return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, col2 - 1 | 0), (function (arr, colIndex) {
                                  return ArraySt$Cnn.push(arr, Vector$Cnn.dot(row, MatrixUtils$Cnn.getCol(row2, col2, colIndex, data2)));
                                }), arr);
                  }), [])
          ];
  }
}

function multiplyScalar(scalar, param) {
  return [
          param[0],
          param[1],
          ArraySt$Cnn.map(param[2], (function (v) {
                  return v * scalar;
                }))
        ];
}

function add(param, param$1) {
  var data2 = param$1[2];
  var col1 = param[1];
  var row1 = param[0];
  if (col1 !== param$1[1] || row1 !== param$1[0]) {
    return Exception$Cnn.throwErr("error");
  } else {
    return [
            row1,
            col1,
            ArraySt$Cnn.mapi(param[2], (function (v, i) {
                    return v + Caml_array.get(data2, i);
                  }))
          ];
  }
}

function sub(param, param$1) {
  var data2 = param$1[2];
  var col1 = param[1];
  var row1 = param[0];
  if (col1 !== param$1[1] || row1 !== param$1[0]) {
    return Exception$Cnn.throwErr("error");
  } else {
    return [
            row1,
            col1,
            ArraySt$Cnn.mapi(param[2], (function (v, i) {
                    return v - Caml_array.get(data2, i);
                  }))
          ];
  }
}

export {
  getData ,
  getColCount ,
  getRowCount ,
  map ,
  mapi ,
  forEachRow ,
  forEachCol ,
  create ,
  transpose ,
  multiply ,
  multiplyScalar ,
  add ,
  sub ,
  
}
/* No side effect */
