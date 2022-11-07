'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var Vector$Gender_analyze = require("./Vector.bs.js");
var ArraySt$Gender_analyze = require("./ArraySt.bs.js");
var Exception$Gender_analyze = require("./Exception.bs.js");
var MatrixUtils$Gender_analyze = require("./MatrixUtils.bs.js");

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
  return ArraySt$Gender_analyze.forEach(ArraySt$Gender_analyze.range(0, param[0] - 1 | 0), Curry.__1(func));
}

function forEachCol(param, func) {
  return ArraySt$Gender_analyze.forEach(ArraySt$Gender_analyze.range(0, param[1] - 1 | 0), Curry.__1(func));
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
          ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, row - 1 | 0), (function (arr, rowIndex) {
                  return ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, col - 1 | 0), (function (arr, colIndex) {
                                arr[MatrixUtils$Gender_analyze.computeIndex(row, colIndex, rowIndex)] = Caml_array.get(data, MatrixUtils$Gender_analyze.computeIndex(col, rowIndex, colIndex));
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
    return Exception$Gender_analyze.throwErr("error");
  } else {
    return [
            row1,
            col2,
            ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, row1 - 1 | 0), (function (arr, rowIndex) {
                    var row = MatrixUtils$Gender_analyze.getRow(col1, rowIndex, data1);
                    return ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, col2 - 1 | 0), (function (arr, colIndex) {
                                  return ArraySt$Gender_analyze.push(arr, Vector$Gender_analyze.dot(row, MatrixUtils$Gender_analyze.getCol(row2, col2, colIndex, data2)));
                                }), arr);
                  }), [])
          ];
  }
}

function multiplyScalar(scalar, param) {
  return [
          param[0],
          param[1],
          ArraySt$Gender_analyze.map(param[2], (function (v) {
                  return v * scalar;
                }))
        ];
}

function add(param, param$1) {
  var data2 = param$1[2];
  var col1 = param[1];
  var row1 = param[0];
  if (col1 !== param$1[1] || row1 !== param$1[0]) {
    return Exception$Gender_analyze.throwErr("error");
  } else {
    return [
            row1,
            col1,
            ArraySt$Gender_analyze.mapi(param[2], (function (v, i) {
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
    return Exception$Gender_analyze.throwErr("error");
  } else {
    return [
            row1,
            col1,
            ArraySt$Gender_analyze.mapi(param[2], (function (v, i) {
                    return v - Caml_array.get(data2, i);
                  }))
          ];
  }
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
/* No side effect */
