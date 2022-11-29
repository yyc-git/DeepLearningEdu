'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Belt_Array = require("rescript/lib/js/belt_Array.js");
var Exception$Cnn = require("./Exception.bs.js");

function length(prim) {
  return prim.length;
}

function getExn(arr, index) {
  if (index <= (arr.length - 1 | 0) && index >= 0) {
    return arr[index];
  } else {
    return Exception$Cnn.throwErr("Not_found");
  }
}

function getFirstExn(arr) {
  return getExn(arr, 0);
}

function sliceFrom(arr, index) {
  return arr.slice(index);
}

function reduceOneParam(arr, func, param) {
  return Belt_Array.reduceU(arr, param, func);
}

function reduceOneParami(arr, func, param) {
  var mutableParam = param;
  for(var i = 0 ,i_finish = arr.length; i < i_finish; ++i){
    mutableParam = func(mutableParam, arr[i], i);
  }
  return mutableParam;
}

function range(a, b) {
  var result = [];
  for(var i = a; i <= b; ++i){
    result.push(i);
  }
  return result;
}

function map(arr, func) {
  return arr.map(Curry.__1(func));
}

function mapi(arr, func) {
  return arr.map(Curry.__2(func));
}

function push(arr, value) {
  arr.push(value);
  return arr;
}

function forEach(arr, func) {
  arr.forEach(Curry.__1(func));
  
}

function forEachi(arr, func) {
  arr.forEach(Curry.__2(func));
  
}

function copy(arr) {
  return arr.slice(0);
}

var reverse = ((arr) => { return arr.reverse()});

exports.length = length;
exports.getExn = getExn;
exports.getFirstExn = getFirstExn;
exports.sliceFrom = sliceFrom;
exports.reduceOneParam = reduceOneParam;
exports.reduceOneParami = reduceOneParami;
exports.range = range;
exports.map = map;
exports.mapi = mapi;
exports.push = push;
exports.forEach = forEach;
exports.forEachi = forEachi;
exports.copy = copy;
exports.reverse = reverse;
/* No side effect */
