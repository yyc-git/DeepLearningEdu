

import * as InitValue$Cnn from "./InitValue.bs.js";

function normal(randomFunc, fanIn, fanOut) {
  var std = Math.sqrt(2 / (fanIn + fanOut | 0));
  return InitValue$Cnn.normal(randomFunc, 0.0, std);
}

export {
  normal ,
  
}
/* No side effect */
