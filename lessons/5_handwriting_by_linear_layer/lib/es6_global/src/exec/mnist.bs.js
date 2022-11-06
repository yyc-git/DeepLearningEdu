

import * as ArraySt$Gender_analyze from "./ArraySt.bs.js";

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

export {
  getMnistData ,
  getMnistLabels ,
  
}
/* No side effect */
