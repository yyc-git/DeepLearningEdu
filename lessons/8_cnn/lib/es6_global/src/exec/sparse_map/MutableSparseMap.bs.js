

import * as NullUtils$8_cnn from "../NullUtils.bs.js";
import * as SparseMap$8_cnn from "./SparseMap.bs.js";

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
