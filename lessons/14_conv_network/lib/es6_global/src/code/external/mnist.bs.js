

import * as ArraySt$Cnn from "../ArraySt.bs.js";

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

export {
  getMnistData ,
  getMnistLabels ,
  
}
/* No side effect */
