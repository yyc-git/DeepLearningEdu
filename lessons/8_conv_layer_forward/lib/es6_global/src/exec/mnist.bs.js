

import * as ArraySt$8_cnn from "./ArraySt.bs.js";

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

export {
  getMnistData ,
  getMnistLabels ,
  
}
/* No side effect */
