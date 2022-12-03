'use strict';

var Caml_array = require("rescript/lib/js/caml_array.js");
var ArraySt$8_cnn = require("./ArraySt.bs.js");

function computeIndex(colCount, rowIndex, colIndex) {
  return Math.imul(rowIndex, colCount) + colIndex | 0;
}

function getRow(colCount, rowIndex, data) {
  return data.slice(Math.imul(rowIndex, colCount), Math.imul(rowIndex + 1 | 0, colCount));
}

function getCol(rowCount, colCount, colIndex, data) {
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, rowCount - 1 | 0), (function (arr, rowIndex) {
                return ArraySt$8_cnn.push(arr, Caml_array.get(data, computeIndex(colCount, rowIndex, colIndex)));
              }), []);
}

function getValue(rowIndex, colIndex, param) {
  return Caml_array.get(param[2], computeIndex(param[1], rowIndex, colIndex));
}

function setValue(param, value, rowIndex, colIndex) {
  var data = param[2];
  var col = param[1];
  data[computeIndex(col, rowIndex, colIndex)] = value;
  return [
          param[0],
          col,
          data
        ];
}

exports.computeIndex = computeIndex;
exports.getRow = getRow;
exports.getCol = getCol;
exports.getValue = getValue;
exports.setValue = setValue;
/* No side effect */
