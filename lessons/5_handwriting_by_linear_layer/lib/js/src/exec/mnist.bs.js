'use strict';

var ArraySt$Gender_analyze = require("./ArraySt.bs.js");

function getMnistData(data) {
  return ArraySt$Gender_analyze.map(data, (function (param) {
                return param.input;
              }));
}

function getMnistLabels(data) {
  return ArraySt$Gender_analyze.map(data, (function (param) {
                return param.output;
              }));
}

exports.getMnistData = getMnistData;
exports.getMnistLabels = getMnistLabels;
/* No side effect */
