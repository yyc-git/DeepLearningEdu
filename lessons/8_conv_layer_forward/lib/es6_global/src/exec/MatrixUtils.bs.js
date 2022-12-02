

import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
import * as ArraySt$8_cnn from "./ArraySt.bs.js";

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

export {
  computeIndex ,
  getRow ,
  getCol ,
  getValue ,
  setValue ,
  
}
/* No side effect */
