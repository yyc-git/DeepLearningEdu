

import * as Belt_Option from "../../../../../../node_modules/rescript/lib/es6/belt_Option.js";

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

export {
  unsafeGet ,
  _getExn ,
  getExn ,
  getWithDefault ,
  map ,
  bind ,
  
}
/* No side effect */
