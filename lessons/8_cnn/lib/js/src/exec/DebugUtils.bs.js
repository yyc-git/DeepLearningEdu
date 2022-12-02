'use strict';

var Matrix$8_cnn = require("./Matrix.bs.js");
var Vector$8_cnn = require("./Vector.bs.js");

var _isGradientExplosionOrDisappear = ((gradient) => { return Number.isNaN(gradient) || (gradient !== 0.0 && Math.abs(gradient) < 0.0000001)|| !Number.isFinite(gradient) || Math.abs(gradient) > 1.0 });

function checkGradientExplosionOrDisappear(gradient) {
  return Matrix$8_cnn.mapi(gradient, (function (value, i) {
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
  Vector$8_cnn.mapi(weight, (function (weightValue, i) {
          return _checkWeightValueAndGradientValueRadio(weightValue, Vector$8_cnn.toArray(gradient)[i]);
        }));
  
}

function checkWeightMatrixAndGradientMatrixRadio(weight, gradient) {
  Matrix$8_cnn.mapi(weight, (function (weightValue, i) {
          return _checkWeightValueAndGradientValueRadio(weightValue, Matrix$8_cnn.getData(gradient)[i]);
        }));
  
}

function checkSigmoidInputTooLarge(input) {
  if (Math.abs(input) > 7) {
    console.log("input for sigmoid is too large: " + input);
    return ;
  }
  
}

exports._isGradientExplosionOrDisappear = _isGradientExplosionOrDisappear;
exports.checkGradientExplosionOrDisappear = checkGradientExplosionOrDisappear;
exports._checkWeightValueAndGradientValueRadio = _checkWeightValueAndGradientValueRadio;
exports.checkWeightVectorAndGradientVectorRadio = checkWeightVectorAndGradientVectorRadio;
exports.checkWeightMatrixAndGradientMatrixRadio = checkWeightMatrixAndGradientMatrixRadio;
exports.checkSigmoidInputTooLarge = checkSigmoidInputTooLarge;
/* No side effect */
