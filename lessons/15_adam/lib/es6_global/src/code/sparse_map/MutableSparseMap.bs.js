

import * as NullUtils$Cnn from "../NullUtils.bs.js";
import * as SparseMap$Cnn from "./SparseMap.bs.js";

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

export {
  createEmpty ,
  copy ,
  unsafeGet ,
  get ,
  getExn ,
  fastGet ,
  getNullable ,
  has ,
  set ,
  remove ,
  map ,
  reducei ,
  getValues ,
  getKeys ,
  deleteVal ,
  
}
/* No side effect */
