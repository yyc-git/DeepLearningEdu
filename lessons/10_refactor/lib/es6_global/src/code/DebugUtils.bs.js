

import * as Matrix$Cnn from "./Matrix.bs.js";
import * as Vector$Cnn from "./Vector.bs.js";

var _isGradientExplosionOrDisappear = ((gradient) => { return Number.isNaN(gradient) || (gradient !== 0.0 && Math.abs(gradient) < 0.0000001)|| !Number.isFinite(gradient) || Math.abs(gradient) > 1.0 });

function checkGradientExplosionOrDisappear(gradient) {
  return Matrix$Cnn.mapi(gradient, (function (value, i) {
                if (_isGradientExplosionOrDisappear(value)) {
                  console.log("checkGradientExplosionOrDisappear fail: " + value);
                  return ;
                }
                
              }));
}

function _checkWeightValueAndGradientValueRadio(weightValue, gradientValue) {
  if (gradientValue === 0) {
    return ;
  }
  var radio = Math.abs(weightValue) / Math.abs(gradientValue);
  if (radio > 5000 || Math.abs(weightValue) > 0.001 && radio < 0.1) {
    console.log("checkWeightValueAndGradientValueRadio fail: " + weightValue + ",  " + gradientValue + ", radio: " + radio);
    return ;
  }
  
}

function checkWeightVectorAndGradientVectorRadio(weight, gradient) {
  Vector$Cnn.mapi(weight, (function (weightValue, i) {
          return _checkWeightValueAndGradientValueRadio(weightValue, Vector$Cnn.toArray(gradient)[i]);
        }));
  
}

function checkWeightMatrixAndGradientMatrixRadio(weight, gradient) {
  Matrix$Cnn.mapi(weight, (function (weightValue, i) {
          return _checkWeightValueAndGradientValueRadio(weightValue, Matrix$Cnn.getData(gradient)[i]);
        }));
  
}

function checkSigmoidInputTooLarge(input) {
  if (Math.abs(input) > 7) {
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
