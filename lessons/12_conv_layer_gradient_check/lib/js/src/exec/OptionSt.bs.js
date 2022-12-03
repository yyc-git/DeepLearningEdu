'use strict';

var Belt_Option = require("rescript/lib/js/belt_Option.js");

function unsafeGet(prim) {
  return prim;
}

var _getExn = ((nullableData) => {



  if (nullableData !== undefined) {



    return nullableData;



  }







  throw new Error("Not_found")



});

var getExn = _getExn;

var getWithDefault = Belt_Option.getWithDefault;

var map = Belt_Option.map;

var bind = Belt_Option.flatMap;

exports.unsafeGet = unsafeGet;
exports._getExn = _getExn;
exports.getExn = getExn;
exports.getWithDefault = getWithDefault;
exports.map = map;
exports.bind = bind;
/* No side effect */
