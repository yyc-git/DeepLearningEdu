

import * as ArraySt$Cnn from "./ArraySt.bs.js";

function computeIndex(colCount, rowIndex, colIndex) {
  return Math.imul(rowIndex, colCount) + colIndex | 0;
}

function getRow(colCount, rowIndex, data) {
  return data.slice(Math.imul(rowIndex, colCount), Math.imul(rowIndex + 1 | 0, colCount));
}

function getCol(rowCount, colCount, colIndex, data) {
  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, rowCount - 1 | 0), (function (arr, rowIndex) {
                return ArraySt$Cnn.push(arr, ArraySt$Cnn.getExn(data, computeIndex(colCount, rowIndex, colIndex)));
              }), []);
}

function getValueByIndex(param, index) {
  return ArraySt$Cnn.getExn(param[2], index);
}

function getValue(matrix, rowIndex, colIndex) {
  return getValueByIndex(matrix, computeIndex(matrix[1], rowIndex, colIndex));
}

function setValue(param, value, rowIndex, colIndex) {
  var data = param[2];
  var col = param[1];
  data[computeIndex(col, rowIndex, colIndex)] = value;
  return [
          param[0],
          col,
          data
        ];
}

export {
  computeIndex ,
  getRow ,
  getCol ,
  getValueByIndex ,
  getValue ,
  setValue ,
  
}
/* No side effect */
