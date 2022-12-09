'use strict';

var NullUtils$8_cnn = require("../NullUtils.bs.js");
var SparseMap$8_cnn = require("./SparseMap.bs.js");

function fastGet(map, key) {
  var value = SparseMap$8_cnn.unsafeGet(map, key);
  return [
          NullUtils$8_cnn.isInMap(value),
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

var createEmpty = SparseMap$8_cnn.createEmpty;

var copy = SparseMap$8_cnn.copy;

var unsafeGet = SparseMap$8_cnn.unsafeGet;

var get = SparseMap$8_cnn.get;

var getExn = SparseMap$8_cnn.getExn;

var getNullable = SparseMap$8_cnn.getNullable;

var has = SparseMap$8_cnn.has;

var map = SparseMap$8_cnn.map;

var reducei = SparseMap$8_cnn.reducei;

var getValues = SparseMap$8_cnn.getValues;

var getKeys = SparseMap$8_cnn.getKeys;

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
