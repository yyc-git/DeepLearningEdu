

import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
import * as ArraySt$Gender_analyze from "./ArraySt.bs.js";

function create(arr) {
  return arr;
}

function dot(vec1, vec2) {
  return ArraySt$Gender_analyze.reduceOneParami(vec1, (function (sum, data1, i) {
                return sum + data1 * Caml_array.get(vec2, i);
              }), 0);
}

export {
  create ,
  dot ,
  
}
/* No side effect */
