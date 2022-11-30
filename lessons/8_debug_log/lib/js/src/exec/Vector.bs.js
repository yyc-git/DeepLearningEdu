'use strict';

var Caml_array = require("rescript/lib/js/caml_array.js");
var ArraySt$Gender_analyze = require("./ArraySt.bs.js");
var Exception$Gender_analyze = require("./Exception.bs.js");
var MatrixUtils$Gender_analyze = require("./MatrixUtils.bs.js");

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

function reducei(vec, func) {
  return function (param) {
    return ArraySt$Gender_analyze.reduceOneParami(vec, func, param);
  };
}

function sum(vec) {
  var func = function (result, value, param) {
    return result + value;
  };
  return ArraySt$Gender_analyze.reduceOneParami(vec, func, 0);
}

function toArray(vec) {
  return vec;
}

exports.create = create;
exports.dot = dot;
exports.multiply = multiply;
exports.multiplyScalar = multiplyScalar;
exports.add = add;
exports.sub = sub;
exports.scalarSub = scalarSub;
exports.length = length;
exports.transformMatrix = transformMatrix;
exports.map = map;
exports.mapi = mapi;
exports.forEachi = forEachi;
exports.getExn = getExn;
exports.push = push;
exports.reducei = reducei;
exports.sum = sum;
exports.toArray = toArray;
/* No side effect */
