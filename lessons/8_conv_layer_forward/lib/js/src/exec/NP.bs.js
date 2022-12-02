'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Caml_obj = require("rescript/lib/js/caml_obj.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var Matrix$8_cnn = require("./Matrix.bs.js");
var ArraySt$8_cnn = require("./ArraySt.bs.js");
var OptionSt$8_cnn = require("./OptionSt.bs.js");
var Exception$8_cnn = require("./Exception.bs.js");
var MatrixUtils$8_cnn = require("./MatrixUtils.bs.js");
var ImmutableSparseMap$8_cnn = require("./sparse_map/ImmutableSparseMap.bs.js");

function createMatrixMap(getValueFunc, length, row, col) {
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, length - 1 | 0), (function (map, index) {
                return ImmutableSparseMap$8_cnn.set(map, index, Matrix$8_cnn.create(row, col, ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, row - 1 | 0), (function (arr, param) {
                                      return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, col - 1 | 0), (function (arr, param) {
                                                    return ArraySt$8_cnn.push(arr, Curry._1(getValueFunc, undefined));
                                                  }), arr);
                                    }), [])));
              }), ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined));
}

function getMatrixMapSize(matrixMap) {
  var mat = ImmutableSparseMap$8_cnn.getExn(matrixMap, 0);
  return [
          Matrix$8_cnn.getColCount(mat),
          Matrix$8_cnn.getRowCount(mat),
          ImmutableSparseMap$8_cnn.length(matrixMap)
        ];
}

function zeroMatrixMap(length, row, col) {
  return createMatrixMap((function (param) {
                return 0;
              }), length, row, col);
}

function zeroMatrix(row, col) {
  return ImmutableSparseMap$8_cnn.getExn(createMatrixMap((function (param) {
                    return 0;
                  }), 1, row, col), 0);
}

function forEachMatrix(matrix, func) {
  var col = matrix[1];
  return ArraySt$8_cnn.forEach(ArraySt$8_cnn.range(0, matrix[0] - 1 | 0), (function (rowIndex) {
                return ArraySt$8_cnn.forEach(ArraySt$8_cnn.range(0, col - 1 | 0), (function (colIndex) {
                              return Curry._3(func, MatrixUtils$8_cnn.getValue(rowIndex, colIndex, matrix), rowIndex, colIndex);
                            }));
              }));
}

function copyMatrix(param) {
  return [
          param[0],
          param[1],
          param[2].slice()
        ];
}

function reduceMatrix(matrix, func, initialValue) {
  var col = matrix[1];
  var value = {
    contents: initialValue
  };
  ArraySt$8_cnn.forEach(ArraySt$8_cnn.range(0, matrix[0] - 1 | 0), (function (rowIndex) {
          return ArraySt$8_cnn.forEach(ArraySt$8_cnn.range(0, col - 1 | 0), (function (colIndex) {
                        value.contents = Curry._4(func, value.contents, MatrixUtils$8_cnn.getValue(rowIndex, colIndex, matrix), rowIndex, colIndex);
                        
                      }));
        }));
  return value.contents;
}

function fillMatrix(offsetLeft, offsetTop, sourceMatrix, fillMatrix$1) {
  return reduceMatrix(fillMatrix$1, (function (sourceMatrix, value, rowIndex, colIndex) {
                return MatrixUtils$8_cnn.setValue(sourceMatrix, value, rowIndex + offsetTop | 0, colIndex + offsetLeft | 0);
              }), sourceMatrix);
}

function getMatrixRegion(offsetLeft, offsetWidth, offsetTop, offsetHeight, matrix) {
  return reduceMatrix(matrix, (function (regionMatrix, value, rowIndex, colIndex) {
                if (rowIndex >= offsetTop && rowIndex < (offsetTop + offsetHeight | 0) && colIndex >= offsetLeft && colIndex < (offsetLeft + offsetWidth | 0)) {
                  return MatrixUtils$8_cnn.setValue(regionMatrix, value, rowIndex - offsetTop | 0, colIndex - offsetLeft | 0);
                } else {
                  return regionMatrix;
                }
              }), Matrix$8_cnn.create(offsetHeight, offsetWidth, []));
}

function dot(param, param$1) {
  var data2 = param$1[2];
  var col1 = param[1];
  var row1 = param[0];
  if (col1 !== col1 || row1 !== row1) {
    return Exception$8_cnn.throwErr("error");
  } else {
    return [
            row1,
            col1,
            ArraySt$8_cnn.mapi(param[2], (function (v, i) {
                    return v * Caml_array.get(data2, i);
                  }))
          ];
  }
}

var _getMinNumber = (() => { return Number.MIN_VALUE });

function max(matrix) {
  return reduceMatrix(matrix, (function (result, value, param, param$1) {
                return Math.max(result, value);
              }), Curry._1(_getMinNumber, undefined));
}

function getMaxIndex(matrix) {
  var match = reduceMatrix(matrix, (function (result, value, rowIndex, colIndex) {
          if (Caml_obj.caml_greaterequal(value, result[0])) {
            return [
                    value,
                    rowIndex,
                    colIndex
                  ];
          } else {
            return result;
          }
        }), [
        Curry._1(_getMinNumber, undefined),
        undefined,
        undefined
      ]);
  return [
          match[0],
          OptionSt$8_cnn.getExn(match[1]),
          OptionSt$8_cnn.getExn(match[2])
        ];
}

function sum(matrix) {
  return reduceMatrix(matrix, (function (result, value, param, param$1) {
                return result + value;
              }), 0);
}

function sumMatrixMap(matrixMap) {
  return ImmutableSparseMap$8_cnn.reducei(ImmutableSparseMap$8_cnn.map(matrixMap, sum), (function (result, value, param) {
                return result + value;
              }), 0);
}

function rotate180(matrix) {
  var row = Matrix$8_cnn.getRowCount(matrix);
  var col = Matrix$8_cnn.getColCount(matrix);
  return reduceMatrix(matrix, (function (result, value, rowIndex, colIndex) {
                return MatrixUtils$8_cnn.setValue(result, value, (row - rowIndex | 0) - 1 | 0, (col - colIndex | 0) - 1 | 0);
              }), Matrix$8_cnn.create(Matrix$8_cnn.getRowCount(matrix), Matrix$8_cnn.getColCount(matrix), []));
}

function addMatrixMap(matrixMap1, matrixMap2) {
  return ImmutableSparseMap$8_cnn.mapi(matrixMap1, (function (matrix1, i) {
                return Matrix$8_cnn.add(matrix1, ImmutableSparseMap$8_cnn.getExn(matrixMap2, i));
              }));
}

function mapMatrixMap(matrixMap, func) {
  return ImmutableSparseMap$8_cnn.map(matrixMap, Curry.__1(func));
}

function createMatrix(dataArr) {
  var rowCount = ArraySt$8_cnn.length(dataArr);
  var colCount = ArraySt$8_cnn.length(Caml_array.get(dataArr, 0));
  return Matrix$8_cnn.create(rowCount, colCount, ArraySt$8_cnn.reduceOneParam(dataArr, (function (data, rowData) {
                    return data.concat(rowData);
                  }), []));
}

function createMatrixMapByDataArr(dataArr) {
  return ArraySt$8_cnn.map(dataArr, createMatrix);
}

function getMatrixMapValue(matrixMap, lengthIndex, rowIndex, colIndex) {
  return MatrixUtils$8_cnn.getValue(rowIndex, colIndex, ImmutableSparseMap$8_cnn.getExn(matrixMap, lengthIndex));
}

exports.createMatrixMap = createMatrixMap;
exports.getMatrixMapSize = getMatrixMapSize;
exports.zeroMatrixMap = zeroMatrixMap;
exports.zeroMatrix = zeroMatrix;
exports.forEachMatrix = forEachMatrix;
exports.copyMatrix = copyMatrix;
exports.reduceMatrix = reduceMatrix;
exports.fillMatrix = fillMatrix;
exports.getMatrixRegion = getMatrixRegion;
exports.dot = dot;
exports._getMinNumber = _getMinNumber;
exports.max = max;
exports.getMaxIndex = getMaxIndex;
exports.sum = sum;
exports.sumMatrixMap = sumMatrixMap;
exports.rotate180 = rotate180;
exports.addMatrixMap = addMatrixMap;
exports.mapMatrixMap = mapMatrixMap;
exports.createMatrix = createMatrix;
exports.createMatrixMapByDataArr = createMatrixMapByDataArr;
exports.getMatrixMapValue = getMatrixMapValue;
/* No side effect */
