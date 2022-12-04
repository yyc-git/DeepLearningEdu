'use strict';

var InitValue$Cnn = require("./InitValue.bs.js");

function normal(randomFunc, fanIn, _fanOut) {
  var std = Math.sqrt(2 / fanIn);
  return InitValue$Cnn.normal(randomFunc, 0.0, std);
}

exports.normal = normal;
/* No side effect */
