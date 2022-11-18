

import * as Js_math from "../../../../../../node_modules/rescript/lib/es6/js_math.js";
import * as Matrix$Gender_analyze from "./Matrix.bs.js";
import * as Vector$Gender_analyze from "./Vector.bs.js";

var _isGradientExplosionOrDisappear = ((gradient) => { return Number.isNaN(gradient) || Math.abs(gradient) < 0.000000001 || !Number.isFinite(gradient) });

function checkGradientExplosionOrDisappear(gradient) {
  if (_isGradientExplosionOrDisappear(gradient)) {
    console.log("checkGradientExplosionOrDisappear fail: " + gradient);
    return ;
  }
  
}

function _checkWeightValueAndGradientValueRadio(weightValue, gradientValue) {
  if (gradientValue === 0) {
    return ;
  }
  var radio = Math.abs(weightValue) / Math.abs(gradientValue);
  if (Js_math.floor(radio) > 50 || Js_math.floor(radio) < 5) {
    console.log("checkWeightValueAndGradientValueRadio fail: " + weightValue + ",  " + gradientValue + ", radio: " + radio);
    return ;
  }
  
}

function checkWeightVectorAndGradientVectorRadio(weight, gradient) {
  Vector$Gender_analyze.mapi(weight, (function (weightValue, i) {
          return _checkWeightValueAndGradientValueRadio(weightValue, Vector$Gender_analyze.toArray(gradient)[i]);
        }));
  
}

function checkWeightMatrixAndGradientMatrixRadio(weight, gradient) {
  Matrix$Gender_analyze.mapi(weight, (function (weightValue, i) {
          return _checkWeightValueAndGradientValueRadio(weightValue, Matrix$Gender_analyze.getData(gradient)[i]);
        }));
  
}

function checkSigmoidInputTooLarge(input) {
  if (input > 7) {
    console.log("input for sigmoid is too large: " + input);
    return ;
  }
  
}

export {
  _isGradientExplosionOrDisappear ,
  checkGradientExplosionOrDisappear ,
  _checkWeightValueAndGradientValueRadio ,
  checkWeightVectorAndGradientVectorRadio ,
  checkWeightMatrixAndGradientMatrixRadio ,
  checkSigmoidInputTooLarge ,
  
}
/* No side effect */
