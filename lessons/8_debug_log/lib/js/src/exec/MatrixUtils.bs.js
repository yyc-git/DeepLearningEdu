'use strict';

var Caml_array = require("rescript/lib/js/caml_array.js");
var ArraySt$Gender_analyze = require("./ArraySt.bs.js");

function computeIndex(colCount, rowIndex, colIndex) {
  return Math.imul(rowIndex, colCount) + colIndex | 0;
}

function getRow(colCount, rowIndex, data) {
  return data.slice(Math.imul(rowIndex, colCount), Math.imul(rowIndex + 1 | 0, colCount));
}

function getCol(rowCount, colCount, colIndex, data) {
  return ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, rowCount - 1 | 0), (function (arr, rowIndex) {
                return ArraySt$Gender_analyze.push(arr, Caml_array.get(data, computeIndex(colCount, rowIndex, colIndex)));
              }), []);
}

function getValue(rowIndex, colIndex, param) {
  return Caml_array.get(param[2], computeIndex(param[1], rowIndex, colIndex));
}

exports.computeIndex = computeIndex;
exports.getRow = getRow;
exports.getCol = getCol;
exports.getValue = getValue;
/* No side effect */
