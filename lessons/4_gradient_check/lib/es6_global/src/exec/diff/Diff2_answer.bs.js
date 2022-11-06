

import * as Curry from "../../../../../../../node_modules/rescript/lib/es6/curry.js";

function diff(f, x) {
  return (Curry._1(f, x + 1e-4) - Curry._1(f, x - 1e-4)) / (2.0 * 1e-4);
}

export {
  diff ,
  
}
/* No side effect */
