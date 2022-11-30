

import * as SparseMap$Cnn from "./SparseMap.bs.js";

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

export {
  createEmpty ,
  createFromArr ,
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
