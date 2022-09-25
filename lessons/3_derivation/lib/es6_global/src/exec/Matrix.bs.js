

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
import * as ArraySt$Gender_analyze from "./ArraySt.bs.js";
import * as MatrixUtils$Gender_analyze from "./MatrixUtils.bs.js";

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

export {
  getData ,
  getRowCount ,
  forEachRow ,
  forEachCol ,
  create ,
  transpose ,
  multiplyScalar ,
  
}
/* No side effect */
