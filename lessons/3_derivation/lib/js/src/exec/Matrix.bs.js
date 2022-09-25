'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var ArraySt$Gender_analyze = require("./ArraySt.bs.js");
var MatrixUtils$Gender_analyze = require("./MatrixUtils.bs.js");

function getData(param) {
  return param[2];
}

function getRowCount(param) {
  return param[0];
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

function multiplyScalar(scalar, param) {
  return [
          param[0],
          param[1],
          ArraySt$Gender_analyze.map(param[2], (function (v) {
                  return v * scalar;
                }))
        ];
}

exports.getData = getData;
exports.getRowCount = getRowCount;
exports.forEachRow = forEachRow;
exports.forEachCol = forEachCol;
exports.create = create;
exports.transpose = transpose;
exports.multiplyScalar = multiplyScalar;
/* No side effect */
