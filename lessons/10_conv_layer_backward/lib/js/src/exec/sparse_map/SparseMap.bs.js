'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Caml_option = require("rescript/lib/js/caml_option.js");
var ArraySt$8_cnn = require("../ArraySt.bs.js");
var NullUtils$8_cnn = require("../NullUtils.bs.js");

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

exports.createEmpty = createEmpty;
exports.length = length;
exports.forEachi = forEachi;
exports.copy = copy;
exports.unsafeGet = unsafeGet;
exports._getExn = _getExn;
exports.getExn = getExn;
exports.get = get;
exports.getNullable = getNullable;
exports.has = has;
exports.map = map;
exports.mapi = mapi;
exports.reducei = reducei;
exports.getValues = getValues;
exports.getKeys = getKeys;
/* No side effect */
