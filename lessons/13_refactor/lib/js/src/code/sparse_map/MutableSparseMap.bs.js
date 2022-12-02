'use strict';

var NullUtils$Cnn = require("../NullUtils.bs.js");
var SparseMap$Cnn = require("./SparseMap.bs.js");

function fastGet(map, key) {
  var value = SparseMap$Cnn.unsafeGet(map, key);
  return [
          NullUtils$Cnn.isInMap(value),
          value
        ];
}

function set(map, key, value) {
  map[key] = value;
  return map;
}

function remove(map, key) {
  map[key] = undefined;
  return map;
}

function deleteVal(map, key) {
  map[key] = undefined;
  return map;
}

var createEmpty = SparseMap$Cnn.createEmpty;

var copy = SparseMap$Cnn.copy;

var unsafeGet = SparseMap$Cnn.unsafeGet;

var get = SparseMap$Cnn.get;

var getExn = SparseMap$Cnn.getExn;

var getNullable = SparseMap$Cnn.getNullable;

var has = SparseMap$Cnn.has;

var map = SparseMap$Cnn.map;

var reducei = SparseMap$Cnn.reducei;

var getValues = SparseMap$Cnn.getValues;

var getKeys = SparseMap$Cnn.getKeys;

exports.createEmpty = createEmpty;
exports.copy = copy;
exports.unsafeGet = unsafeGet;
exports.get = get;
exports.getExn = getExn;
exports.fastGet = fastGet;
exports.getNullable = getNullable;
exports.has = has;
exports.set = set;
exports.remove = remove;
exports.map = map;
exports.reducei = reducei;
exports.getValues = getValues;
exports.getKeys = getKeys;
exports.deleteVal = deleteVal;
/* No side effect */
