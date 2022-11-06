

import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
import * as ArraySt$Gender_analyze from "./ArraySt.bs.js";
import * as Exception$Gender_analyze from "./Exception.bs.js";
import * as MatrixUtils$Gender_analyze from "./MatrixUtils.bs.js";

function create(arr) {
  return arr;
}

function dot(vec1, vec2) {
  return ArraySt$Gender_analyze.reduceOneParami(vec1, (function (sum, data1, i) {
                return sum + data1 * Caml_array.get(vec2, i);
              }), 0);
}

function multiply(vec1, vec2) {
  return ArraySt$Gender_analyze.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$Gender_analyze.push(arr, data1 * Caml_array.get(vec2, i));
              }), []);
}

function multiplyScalar(vec, scalar) {
  return ArraySt$Gender_analyze.reduceOneParam(vec, (function (arr, data) {
                return ArraySt$Gender_analyze.push(arr, data * scalar);
              }), []);
}

function add(vec1, vec2) {
  return ArraySt$Gender_analyze.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$Gender_analyze.push(arr, data1 + Caml_array.get(vec2, i));
              }), []);
}

function sub(vec1, vec2) {
  return ArraySt$Gender_analyze.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$Gender_analyze.push(arr, data1 - Caml_array.get(vec2, i));
              }), []);
}

function scalarSub(scalar, vec) {
  return ArraySt$Gender_analyze.reduceOneParam(vec, (function (arr, data) {
                return ArraySt$Gender_analyze.push(arr, scalar - data);
              }), []);
}

function length(vec) {
  return vec.length;
}

function transformMatrix(param, vec) {
  var matrixData = param[2];
  var col = param[1];
  return ArraySt$Gender_analyze.reduceOneParam(ArraySt$Gender_analyze.range(0, param[0] - 1 | 0), (function (arr, rowIndex) {
                return ArraySt$Gender_analyze.push(arr, dot(MatrixUtils$Gender_analyze.getRow(col, rowIndex, matrixData), vec));
              }), []);
}

var map = ArraySt$Gender_analyze.map;

var mapi = ArraySt$Gender_analyze.mapi;

var forEachi = ArraySt$Gender_analyze.forEachi;

function getExn(vec, index) {
  if (index > vec.length) {
    return Exception$Gender_analyze.throwErr("error");
  } else {
    return vec[index];
  }
}

function push(vec, value) {
  return ArraySt$Gender_analyze.push(ArraySt$Gender_analyze.sliceFrom(vec, 0), value);
}

export {
  create ,
  dot ,
  multiply ,
  multiplyScalar ,
  add ,
  sub ,
  scalarSub ,
  length ,
  transformMatrix ,
  map ,
  mapi ,
  forEachi ,
  getExn ,
  push ,
  
}
/* No side effect */
