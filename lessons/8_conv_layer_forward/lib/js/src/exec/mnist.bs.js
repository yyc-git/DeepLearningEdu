'use strict';

var ArraySt$8_cnn = require("./ArraySt.bs.js");

function getMnistData(data) {
  return ArraySt$8_cnn.map(data, (function (param) {
                return param.input;
              }));
}

function getMnistLabels(data) {
  return ArraySt$8_cnn.map(data, (function (param) {
                return param.output;
              }));
}

exports.getMnistData = getMnistData;
exports.getMnistLabels = getMnistLabels;
/* No side effect */
