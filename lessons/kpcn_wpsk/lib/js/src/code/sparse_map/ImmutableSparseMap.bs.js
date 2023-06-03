'use strict';

var SparseMap$Cnn = require("./SparseMap.bs.js");

function createFromArr(arr) {
  return arr;
}

function set(map, key, value) {
  var newMap = SparseMap$Cnn.copy(map);
  newMap[key] = value;
  return newMap;
}

function remove(map, key) {
  var newMap = SparseMap$Cnn.copy(map);
  newMap[key] = undefined;
  return newMap;
}

function deleteVal(map, key) {
  var newMap = SparseMap$Cnn.copy(map);
  newMap[key] = undefined;
  return newMap;
}

var createEmpty = SparseMap$Cnn.createEmpty;

var length = SparseMap$Cnn.length;

var copy = SparseMap$Cnn.copy;

var unsafeGet = SparseMap$Cnn.unsafeGet;

var get = SparseMap$Cnn.get;

var getNullable = SparseMap$Cnn.getNullable;

var getExn = SparseMap$Cnn.getExn;

var has = SparseMap$Cnn.has;

var map = SparseMap$Cnn.map;

var mapi = SparseMap$Cnn.mapi;

var reducei = SparseMap$Cnn.reducei;

var getValues = SparseMap$Cnn.getValues;

var getKeys = SparseMap$Cnn.getKeys;

var forEachi = SparseMap$Cnn.forEachi;

exports.createEmpty = createEmpty;
exports.createFromArr = createFromArr;
exports.length = length;
exports.copy = copy;
exports.unsafeGet = unsafeGet;
exports.get = get;
exports.getNullable = getNullable;
exports.getExn = getExn;
exports.has = has;
exports.set = set;
exports.remove = remove;
exports.map = map;
exports.mapi = mapi;
exports.reducei = reducei;
exports.getValues = getValues;
exports.getKeys = getKeys;
exports.deleteVal = deleteVal;
exports.forEachi = forEachi;
/* No side effect */
