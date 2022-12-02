'use strict';

var Curry = require("rescript/lib/js/curry.js");

function random(randomFunc, _fanIn, _fanOut) {
  return Curry._1(randomFunc, undefined);
}

exports.random = random;
/* No side effect */
