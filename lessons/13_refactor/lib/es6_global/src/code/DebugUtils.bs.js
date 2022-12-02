

import * as Matrix$Cnn from "./Matrix.bs.js";
import * as Vector$Cnn from "./Vector.bs.js";
import * as Exception$Cnn from "./Exception.bs.js";

var _isExplosion = ((value) => { return !Number.isFinite(value) || Math.abs(value) > 1e9 });

function _checkExplosion(value) {
  if (_isExplosion(value)) {
    return Exception$Cnn.throwErr("_checkExplosion fail: " + value);
  }
  
}

function checkOutputVectorExplosion(output) {
  Vector$Cnn.map(output, _checkExplosion);
  
}

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
  _isExplosion ,
  _checkExplosion ,
  checkOutputVectorExplosion ,
  _isGradientExplosionOrDisappear ,
  checkGradientExplosionOrDisappear ,
  _checkWeightValueAndGradientValueRadio ,
  checkWeightVectorAndGradientVectorRadio ,
  checkWeightMatrixAndGradientMatrixRadio ,
  checkSigmoidInputTooLarge ,
  
}
/* No side effect */
