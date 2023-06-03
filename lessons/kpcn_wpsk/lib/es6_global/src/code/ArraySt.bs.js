

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Belt_Array from "../../../../../../node_modules/rescript/lib/es6/belt_Array.js";
import * as Exception$Cnn from "./Exception.bs.js";

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

export {
  length ,
  getExn ,
  getFirstExn ,
  sliceFrom ,
  reduceOneParam ,
  reduceOneParami ,
  range ,
  map ,
  mapi ,
  push ,
  forEach ,
  forEachi ,
  copy ,
  reverse ,
  
}
/* No side effect */
