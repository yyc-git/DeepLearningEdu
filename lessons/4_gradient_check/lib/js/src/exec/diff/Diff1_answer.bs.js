'use strict';

var Curry = require("rescript/lib/js/curry.js");

function diff(f, x) {
  return (Curry._1(f, x + 1e-4) - Curry._1(f, x)) / 1e-4;
}

exports.diff = diff;
/* No side effect */
