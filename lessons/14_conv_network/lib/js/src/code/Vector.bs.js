'use strict';

var Caml_array = require("rescript/lib/js/caml_array.js");
var ArraySt$Cnn = require("./ArraySt.bs.js");
var Exception$Cnn = require("./Exception.bs.js");
var MatrixUtils$Cnn = require("./MatrixUtils.bs.js");
var NumberUtils$Cnn = require("./NumberUtils.bs.js");

function create(arr) {
  return arr;
}

function dot(vec1, vec2) {
  return ArraySt$Cnn.reduceOneParami(vec1, (function (sum, data1, i) {
                return sum + data1 * Caml_array.get(vec2, i);
              }), 0);
}

function multiply(vec1, vec2) {
  return ArraySt$Cnn.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$Cnn.push(arr, data1 * Caml_array.get(vec2, i));
              }), []);
}

function multiplyScalar(vec, scalar) {
  return ArraySt$Cnn.reduceOneParam(vec, (function (arr, data) {
                return ArraySt$Cnn.push(arr, data * scalar);
              }), []);
}

function add(vec1, vec2) {
  return ArraySt$Cnn.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$Cnn.push(arr, data1 + Caml_array.get(vec2, i));
              }), []);
}

function addScalar(vec1, scalar) {
  return ArraySt$Cnn.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$Cnn.push(arr, data1 + scalar);
              }), []);
}

function sub(vec1, vec2) {
  return ArraySt$Cnn.reduceOneParami(vec1, (function (arr, data1, i) {
                return ArraySt$Cnn.push(arr, data1 - Caml_array.get(vec2, i));
              }), []);
}

function scalarSub(scalar, vec) {
  return ArraySt$Cnn.reduceOneParam(vec, (function (arr, data) {
                return ArraySt$Cnn.push(arr, scalar - data);
              }), []);
}

function length(vec) {
  return vec.length;
}

function transformMatrix(param, vec) {
  var matrixData = param[2];
  var col = param[1];
  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, param[0] - 1 | 0), (function (arr, rowIndex) {
                return ArraySt$Cnn.push(arr, dot(MatrixUtils$Cnn.getRow(col, rowIndex, matrixData), vec));
              }), []);
}

var map = ArraySt$Cnn.map;

var mapi = ArraySt$Cnn.mapi;

var forEachi = ArraySt$Cnn.forEachi;

function reducei(vec, func) {
  return function (param) {
    return ArraySt$Cnn.reduceOneParami(vec, func, param);
  };
}

function toArray(vec) {
  return vec;
}

function max(vec) {
  var func = function (result, value, param) {
    return Math.max(result, value);
  };
  return ArraySt$Cnn.reduceOneParami(vec, func, NumberUtils$Cnn.getMinNumber(undefined));
}

function min(vec) {
  var func = function (result, value, param) {
    return Math.min(result, value);
  };
  return ArraySt$Cnn.reduceOneParami(vec, func, NumberUtils$Cnn.getMaxNumber(undefined));
}

function sum(vec) {
  var func = function (result, value, param) {
    return result + value;
  };
  return ArraySt$Cnn.reduceOneParami(vec, func, 0);
}

function getExn(vec, index) {
  if (index > vec.length) {
    return Exception$Cnn.throwErr("error");
  } else {
    return vec[index];
  }
}

exports.create = create;
exports.dot = dot;
exports.multiply = multiply;
exports.multiplyScalar = multiplyScalar;
exports.add = add;
exports.addScalar = addScalar;
exports.sub = sub;
exports.scalarSub = scalarSub;
exports.length = length;
exports.transformMatrix = transformMatrix;
exports.map = map;
exports.mapi = mapi;
exports.forEachi = forEachi;
exports.reducei = reducei;
exports.toArray = toArray;
exports.max = max;
exports.min = min;
exports.sum = sum;
exports.getExn = getExn;
/* No side effect */
