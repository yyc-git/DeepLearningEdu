

import * as SparseMap$8_cnn from "./SparseMap.bs.js";

function set(map, key, value) {
  var newMap = SparseMap$8_cnn.copy(map);
  newMap[key] = value;
  return newMap;
}

function remove(map, key) {
  var newMap = SparseMap$8_cnn.copy(map);
  newMap[key] = undefined;
  return newMap;
}

function deleteVal(map, key) {
  var newMap = SparseMap$8_cnn.copy(map);
  newMap[key] = undefined;
  return newMap;
}

var createEmpty = SparseMap$8_cnn.createEmpty;

var length = SparseMap$8_cnn.length;

var copy = SparseMap$8_cnn.copy;

var unsafeGet = SparseMap$8_cnn.unsafeGet;

var get = SparseMap$8_cnn.get;

var getNullable = SparseMap$8_cnn.getNullable;

var getExn = SparseMap$8_cnn.getExn;

var has = SparseMap$8_cnn.has;

var map = SparseMap$8_cnn.map;

var mapi = SparseMap$8_cnn.mapi;

var reducei = SparseMap$8_cnn.reducei;

var getValues = SparseMap$8_cnn.getValues;

var getKeys = SparseMap$8_cnn.getKeys;

var forEachi = SparseMap$8_cnn.forEachi;

export {
  createEmpty ,
  length ,
  copy ,
  unsafeGet ,
  get ,
  getNullable ,
  getExn ,
  has ,
  set ,
  remove ,
  map ,
  mapi ,
  reducei ,
  getValues ,
  getKeys ,
  deleteVal ,
  forEachi ,
  
}
/* No side effect */
