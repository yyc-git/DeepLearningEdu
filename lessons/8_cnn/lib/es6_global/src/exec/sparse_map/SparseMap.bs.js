

import * as Curry from "../../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Caml_option from "../../../../../../../node_modules/rescript/lib/es6/caml_option.js";
import * as ArraySt$8_cnn from "../ArraySt.bs.js";
import * as NullUtils$8_cnn from "../NullUtils.bs.js";

function createEmpty(hintSizeOpt, param) {
  return [];
}

function length(map) {
  return map.length;
}

function forEachi(map, func) {
  map.forEach(Curry.__2(func));
  
}

function copy(prim) {
  return prim.slice();
}

function unsafeGet(map, key) {
  return map[key];
}

var _getExn = ((nullableData) => {































































































































  if (nullableData !== undefined) {































































































































    return nullableData;































































































































  }































































































































































































































































  throw new Error("Not_found")































































































































});

function getExn(map, key) {
  return _getExn(map[key]);
}

function get(map, key) {
  var value = map[key];
  if (NullUtils$8_cnn.isEmpty(value)) {
    return ;
  } else {
    return Caml_option.some(value);
  }
}

function getNullable(map, key) {
  return map[key];
}

function has(map, key) {
  return !NullUtils$8_cnn.isEmpty(map[key]);
}

function map(map$1, func) {
  return map$1.map(function (value) {
              if (NullUtils$8_cnn.isNotInMap(value)) {
                return ;
              } else {
                return func(value);
              }
            });
}

function mapi(map, func) {
  return map.map(function (value, i) {
              if (NullUtils$8_cnn.isNotInMap(value)) {
                return ;
              } else {
                return func(value, i);
              }
            });
}

function reducei(map, func, initValue) {
  return ArraySt$8_cnn.reduceOneParami(map, (function (previousValue, value, index) {
                if (NullUtils$8_cnn.isNotInMap(value)) {
                  return previousValue;
                } else {
                  return func(previousValue, value, index);
                }
              }), initValue);
}

function getValues(map) {
  return map.filter(NullUtils$8_cnn.isInMap);
}

function getKeys(map) {
  return ArraySt$8_cnn.reduceOneParami(map, (function (arr, value, key) {
                if (NullUtils$8_cnn.isNotInMap(value)) {
                  return arr;
                } else {
                  arr.push(key);
                  return arr;
                }
              }), []);
}

export {
  createEmpty ,
  length ,
  forEachi ,
  copy ,
  unsafeGet ,
  _getExn ,
  getExn ,
  get ,
  getNullable ,
  has ,
  map ,
  mapi ,
  reducei ,
  getValues ,
  getKeys ,
  
}
/* No side effect */
