'use strict';

var Caml_array = require("rescript/lib/js/caml_array.js");
var ArraySt$Gender_analyze = require("./ArraySt.bs.js");

function create(arr) {
  return arr;
}

function dot(vec1, vec2) {
  return ArraySt$Gender_analyze.reduceOneParami(vec1, (function (sum, data1, i) {
                return sum + data1 * Caml_array.get(vec2, i);
              }), 0);
}

exports.create = create;
exports.dot = dot;
/* No side effect */
