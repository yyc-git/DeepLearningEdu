

import * as InitValue$Cnn from "./InitValue.bs.js";

function normal(randomFunc, fanIn, _fanOut) {
  var std = Math.sqrt(2 / fanIn);
  return InitValue$Cnn.normal(randomFunc, 0.0, std);
}

export {
  normal ,
  
}
/* No side effect */
