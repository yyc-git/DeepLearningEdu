'use strict';

var Caml_array = require("rescript/lib/js/caml_array.js");
var ArraySt$8_cnn = require("./ArraySt.bs.js");
var Exception$8_cnn = require("./Exception.bs.js");
var MatrixUtils$8_cnn = require("./MatrixUtils.bs.js");

function create(arr) {
  return arr;
}

function dot(vec1, vec2) {
  return ArraySt$8_cnn.reduceOneParami(vec1, (function (sum, data1, i) {
                return sum + data1 * Caml_array.get(vec2, i);
              }), 0);
}

function multiply(vec1, vec2) {
  return ArraySt$8_cnn.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$8_cnn.push(arr, data1 * Caml_array.get(vec2, i));
              }), []);
}

function multiplyScalar(vec, scalar) {
  return ArraySt$8_cnn.reduceOneParam(vec, (function (arr, data) {
                return ArraySt$8_cnn.push(arr, data * scalar);
              }), []);
}

function add(vec1, vec2) {
  return ArraySt$8_cnn.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$8_cnn.push(arr, data1 + Caml_array.get(vec2, i));
              }), []);
}

function sub(vec1, vec2) {
  return ArraySt$8_cnn.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$8_cnn.push(arr, data1 - Caml_array.get(vec2, i));
              }), []);
}

function scalarSub(scalar, vec) {
  return ArraySt$8_cnn.reduceOneParam(vec, (function (arr, data) {
                return ArraySt$8_cnn.push(arr, scalar - data);
              }), []);
}

function length(vec) {
  return vec.length;
}

function transformMatrix(param, vec) {
  var matrixData = param[2];
  var col = param[1];
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, param[0] - 1 | 0), (function (arr, rowIndex) {
                return ArraySt$8_cnn.push(arr, dot(MatrixUtils$8_cnn.getRow(col, rowIndex, matrixData), vec));
              }), []);
}

var map = ArraySt$8_cnn.map;

var mapi = ArraySt$8_cnn.mapi;

var forEachi = ArraySt$8_cnn.forEachi;

function getExn(vec, index) {
  if (index > vec.length) {
    return Exception$8_cnn.throwErr("error");
  } else {
    return vec[index];
  }
}

function push(vec, value) {
  return ArraySt$8_cnn.push(ArraySt$8_cnn.sliceFrom(vec, 0), value);
}

function reducei(vec, func) {
  return function (param) {
    return ArraySt$8_cnn.reduceOneParami(vec, func, param);
  };
}

function sum(vec) {
  var func = function (result, value, param) {
    return result + value;
  };
  return ArraySt$8_cnn.reduceOneParami(vec, func, 0);
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
