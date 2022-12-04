

import * as Curry from "../../../../../../node_modules/rescript/lib/es6/curry.js";
import * as Caml_obj from "../../../../../../node_modules/rescript/lib/es6/caml_obj.js";
import * as Caml_array from "../../../../../../node_modules/rescript/lib/es6/caml_array.js";
import * as Matrix$Cnn from "./Matrix.bs.js";
import * as ArraySt$Cnn from "./ArraySt.bs.js";
import * as OptionSt$Cnn from "./OptionSt.bs.js";
import * as Exception$Cnn from "./Exception.bs.js";
import * as MatrixUtils$Cnn from "./MatrixUtils.bs.js";
import * as ImmutableSparseMap$Cnn from "./sparse_map/ImmutableSparseMap.bs.js";

function createMatrixMap(getValueFunc, length, row, col) {
  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, length - 1 | 0), (function (map, index) {
                return ImmutableSparseMap$Cnn.set(map, index, Matrix$Cnn.create(row, col, ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, row - 1 | 0), (function (arr, param) {
                                      return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, col - 1 | 0), (function (arr, param) {
                                                    return ArraySt$Cnn.push(arr, Curry._1(getValueFunc, undefined));
                                                  }), arr);
                                    }), [])));
              }), ImmutableSparseMap$Cnn.createEmpty(undefined, undefined));
}

function getMatrixMapSize(matrixMap) {
  var mat = ImmutableSparseMap$Cnn.getExn(matrixMap, 0);
  return [
          Matrix$Cnn.getColCount(mat),
          Matrix$Cnn.getRowCount(mat),
          ImmutableSparseMap$Cnn.length(matrixMap)
        ];
}

function zeroMatrixMap(length, row, col) {
  return createMatrixMap((function (param) {
                return 0;
              }), length, row, col);
}

function zeroMatrix(row, col) {
  return ImmutableSparseMap$Cnn.getExn(createMatrixMap((function (param) {
                    return 0;
                  }), 1, row, col), 0);
}

function forEachMatrix(matrix, func) {
  var col = matrix[1];
  return ArraySt$Cnn.forEach(ArraySt$Cnn.range(0, matrix[0] - 1 | 0), (function (rowIndex) {
                return ArraySt$Cnn.forEach(ArraySt$Cnn.range(0, col - 1 | 0), (function (colIndex) {
                              return Curry._3(func, MatrixUtils$Cnn.getValue(matrix, rowIndex, colIndex), rowIndex, colIndex);
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
  ArraySt$Cnn.forEach(ArraySt$Cnn.range(0, matrix[0] - 1 | 0), (function (rowIndex) {
          return ArraySt$Cnn.forEach(ArraySt$Cnn.range(0, col - 1 | 0), (function (colIndex) {
                        value.contents = Curry._4(func, value.contents, MatrixUtils$Cnn.getValue(matrix, rowIndex, colIndex), rowIndex, colIndex);
                        
                      }));
        }));
  return value.contents;
}

function fillMatrix(offsetLeft, offsetTop, sourceMatrix, fillMatrix$1) {
  return reduceMatrix(fillMatrix$1, (function (sourceMatrix, value, rowIndex, colIndex) {
                return MatrixUtils$Cnn.setValue(sourceMatrix, value, rowIndex + offsetTop | 0, colIndex + offsetLeft | 0);
              }), sourceMatrix);
}

function getMatrixRegion(offsetLeft, offsetWidth, offsetTop, offsetHeight, matrix) {
  return reduceMatrix(matrix, (function (regionMatrix, value, rowIndex, colIndex) {
                if (rowIndex >= offsetTop && rowIndex < (offsetTop + offsetHeight | 0) && colIndex >= offsetLeft && colIndex < (offsetLeft + offsetWidth | 0)) {
                  return MatrixUtils$Cnn.setValue(regionMatrix, value, rowIndex - offsetTop | 0, colIndex - offsetLeft | 0);
                } else {
                  return regionMatrix;
                }
              }), Matrix$Cnn.create(offsetHeight, offsetWidth, []));
}

function dot(param, param$1) {
  var data2 = param$1[2];
  var col1 = param[1];
  var row1 = param[0];
  if (col1 !== col1 || row1 !== row1) {
    return Exception$Cnn.throwErr("error");
  } else {
    return [
            row1,
            col1,
            ArraySt$Cnn.mapi(param[2], (function (v, i) {
                    return v * Caml_array.get(data2, i);
                  }))
          ];
  }
}

var _getMinNumber = (() => { return -Number.MAX_VALUE });

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
          OptionSt$Cnn.getExn(match[1]),
          OptionSt$Cnn.getExn(match[2])
        ];
}

function sum(matrix) {
  return reduceMatrix(matrix, (function (result, value, param, param$1) {
                return result + value;
              }), 0);
}

function sumMatrixMap(matrixMap) {
  return ImmutableSparseMap$Cnn.reducei(ImmutableSparseMap$Cnn.map(matrixMap, sum), (function (result, value, param) {
                return result + value;
              }), 0);
}

function rotate180(matrix) {
  var row = Matrix$Cnn.getRowCount(matrix);
  var col = Matrix$Cnn.getColCount(matrix);
  return reduceMatrix(matrix, (function (result, value, rowIndex, colIndex) {
                return MatrixUtils$Cnn.setValue(result, value, (row - rowIndex | 0) - 1 | 0, (col - colIndex | 0) - 1 | 0);
              }), Matrix$Cnn.create(Matrix$Cnn.getRowCount(matrix), Matrix$Cnn.getColCount(matrix), []));
}

function addMatrixMap(matrixMap1, matrixMap2) {
  return ImmutableSparseMap$Cnn.mapi(matrixMap1, (function (matrix1, i) {
                return Matrix$Cnn.add(matrix1, ImmutableSparseMap$Cnn.getExn(matrixMap2, i));
              }));
}

function mapMatrixMap(matrixMap, func) {
  return ImmutableSparseMap$Cnn.map(matrixMap, Curry.__1(func));
}

function createMatrix(dataArr) {
  var rowCount = ArraySt$Cnn.length(dataArr);
  var colCount = ArraySt$Cnn.length(Caml_array.get(dataArr, 0));
  return Matrix$Cnn.create(rowCount, colCount, ArraySt$Cnn.reduceOneParam(dataArr, (function (data, rowData) {
                    return data.concat(rowData);
                  }), []));
}

function createMatrixMapByDataArr(dataArr) {
  return ArraySt$Cnn.map(dataArr, createMatrix);
}

function getMatrixMapValue(matrixMap, lengthIndex, rowIndex, colIndex) {
  return MatrixUtils$Cnn.getValue(ImmutableSparseMap$Cnn.getExn(matrixMap, lengthIndex), rowIndex, colIndex);
}

export {
  createMatrixMap ,
  getMatrixMapSize ,
  zeroMatrixMap ,
  zeroMatrix ,
  forEachMatrix ,
  copyMatrix ,
  reduceMatrix ,
  fillMatrix ,
  getMatrixRegion ,
  dot ,
  _getMinNumber ,
  max ,
  getMaxIndex ,
  sum ,
  sumMatrixMap ,
  rotate180 ,
  addMatrixMap ,
  mapMatrixMap ,
  createMatrix ,
  createMatrixMapByDataArr ,
  getMatrixMapValue ,
  
}
/* No side effect */
