'use strict';

var ArraySt$Cnn = require("../ArraySt.bs.js");

function getMnistData(data) {
  return ArraySt$Cnn.map(data, (function (param) {
                return param.input;
              }));
}

function getMnistLabels(data) {
  return ArraySt$Cnn.map(data, (function (param) {
                return param.output;
              }));
}

exports.getMnistData = getMnistData;
exports.getMnistLabels = getMnistLabels;
/* No side effect */
