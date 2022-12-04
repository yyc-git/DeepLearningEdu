

import * as Curry from "../../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as NP$Cnn from "../NP.bs.js";
import * as Matrix$Cnn from "../Matrix.bs.js";
import * as ArraySt$Cnn from "../ArraySt.bs.js";
import * as MatrixUtils$Cnn from "../MatrixUtils.bs.js";

function updateWeight(update, weight, param, gradientDataSum, miniBatchSize, param$1) {
  var sWeight = param$1[1];
  var vWeight = param$1[0];
  var match = param[2];
  var epsion = match[2];
  var beta2 = match[1];
  var beta1 = match[0];
  var t = param[1];
  var learnRate = param[0];
  var match$1 = NP$Cnn.reduceMatrix(weight, (function (param, weightValue, rowIndex, colIndex) {
          var match = Curry._5(update, weightValue, [
                learnRate,
                t,
                [
                  beta1,
                  beta2,
                  epsion
                ]
              ], MatrixUtils$Cnn.getValue(vWeight, rowIndex, colIndex), MatrixUtils$Cnn.getValue(sWeight, rowIndex, colIndex), MatrixUtils$Cnn.getValue(gradientDataSum, rowIndex, colIndex) / miniBatchSize);
          var match$1 = match[1];
          return [
                  ArraySt$Cnn.push(param[0], match[0]),
                  ArraySt$Cnn.push(param[1], match$1[0]),
                  ArraySt$Cnn.push(param[2], match$1[1])
                ];
        }), [
        [],
        [],
        []
      ]);
  var rowCount = Matrix$Cnn.getRowCount(weight);
  var colCount = Matrix$Cnn.getColCount(weight);
  return [
          Matrix$Cnn.create(rowCount, colCount, match$1[0]),
          [
            Matrix$Cnn.create(rowCount, colCount, match$1[1]),
            Matrix$Cnn.create(rowCount, colCount, match$1[2])
          ]
        ];
}

export {
  updateWeight ,
  
}
/* No side effect */
