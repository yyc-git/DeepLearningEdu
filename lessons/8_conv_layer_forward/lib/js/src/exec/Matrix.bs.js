'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var Vector$8_cnn = require("./Vector.bs.js");
var ArraySt$8_cnn = require("./ArraySt.bs.js");
var Exception$8_cnn = require("./Exception.bs.js");
var MatrixUtils$8_cnn = require("./MatrixUtils.bs.js");

function getData(param) {
  return param[2];
}

function getRowCount(param) {
  return param[0];
}

function getColCount(param) {
  return param[1];
}

function forEachRow(param, func) {
  return ArraySt$8_cnn.forEach(ArraySt$8_cnn.range(0, param[0] - 1 | 0), Curry.__1(func));
}

function forEachCol(param, func) {
  return ArraySt$8_cnn.forEach(ArraySt$8_cnn.range(0, param[1] - 1 | 0), Curry.__1(func));
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
          ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, row - 1 | 0), (function (arr, rowIndex) {
                  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, col - 1 | 0), (function (arr, colIndex) {
                                arr[MatrixUtils$8_cnn.computeIndex(row, colIndex, rowIndex)] = Caml_array.get(data, MatrixUtils$8_cnn.computeIndex(col, rowIndex, colIndex));
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
    return Exception$8_cnn.throwErr("error");
  } else {
    return [
            row1,
            col2,
            ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, row1 - 1 | 0), (function (arr, rowIndex) {
                    var row = MatrixUtils$8_cnn.getRow(col1, rowIndex, data1);
                    return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, col2 - 1 | 0), (function (arr, colIndex) {
                                  return ArraySt$8_cnn.push(arr, Vector$8_cnn.dot(row, MatrixUtils$8_cnn.getCol(row2, col2, colIndex, data2)));
                                }), arr);
                  }), [])
          ];
  }
}

function multiplyScalar(scalar, param) {
  return [
          param[0],
          param[1],
          ArraySt$8_cnn.map(param[2], (function (v) {
                  return v * scalar;
                }))
        ];
}

function add(param, param$1) {
  var data2 = param$1[2];
  var col1 = param[1];
  var row1 = param[0];
  if (col1 !== param$1[1] || row1 !== param$1[0]) {
    return Exception$8_cnn.throwErr("error");
  } else {
    return [
            row1,
            col1,
            ArraySt$8_cnn.mapi(param[2], (function (v, i) {
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
    return Exception$8_cnn.throwErr("error");
  } else {
    return [
            row1,
            col1,
            ArraySt$8_cnn.mapi(param[2], (function (v, i) {
                    return v - Caml_array.get(data2, i);
                  }))
          ];
  }
}

function map(matrix, func) {
  return [
          getRowCount(matrix),
          getColCount(matrix),
          ArraySt$8_cnn.map(getData(matrix), func)
        ];
}

function mapi(matrix, func) {
  return [
          getRowCount(matrix),
          getColCount(matrix),
          ArraySt$8_cnn.mapi(getData(matrix), func)
        ];
}

exports.getData = getData;
exports.getRowCount = getRowCount;
exports.getColCount = getColCount;
exports.forEachRow = forEachRow;
exports.forEachCol = forEachCol;
exports.create = create;
exports.transpose = transpose;
exports.multiply = multiply;
exports.multiplyScalar = multiplyScalar;
exports.add = add;
exports.sub = sub;
exports.map = map;
exports.mapi = mapi;
/* No side effect */
