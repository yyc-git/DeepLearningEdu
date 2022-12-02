let createMatrixMap = (getValueFunc, length, row, col) => {
  ArraySt.range(0, length - 1)->ArraySt.reduceOneParam((. map, index) => {
    map->ImmutableSparseMap.set(
      index,
      Matrix.create(row, col, ArraySt.range(0, row - 1)->ArraySt.reduceOneParam((. arr, _) => {
          ArraySt.range(0, col - 1)->ArraySt.reduceOneParam((. arr, _) => {
            arr->ArraySt.push(getValueFunc())
          }, arr)
        }, [])),
    )
  }, ImmutableSparseMap.createEmpty())
}

let getMatrixMapSize = matrixMap => {
  let mat = matrixMap->ImmutableSparseMap.getExn(0)

  (Matrix.getColCount(mat), Matrix.getRowCount(mat), matrixMap->ImmutableSparseMap.length)
}

let zeroMatrixMap = (length, row, col) => {
  createMatrixMap(() => 0., length, row, col)
}

let zeroMatrix = (row, col) => {
  createMatrixMap(() => 0., 1, row, col)->ImmutableSparseMap.getExn(0)
}

let forEachMatrix = ((row, col, data) as matrix, func) => {
  ArraySt.range(0, row - 1)->ArraySt.forEach(rowIndex => {
    ArraySt.range(0, col - 1)->ArraySt.forEach(colIndex => {
      func(MatrixUtils.getValue(rowIndex, colIndex, matrix), rowIndex, colIndex)
    })
  })
}

let copyMatrix = ((row, col, data)) => {
  (row, col, data->Js.Array.copy)
}

let reduceMatrix = ((row, col, data) as matrix, func, initialValue) => {
  let value = ref(initialValue)
  ArraySt.range(0, row - 1)->ArraySt.forEach(rowIndex => {
    ArraySt.range(0, col - 1)->ArraySt.forEach(colIndex => {
      value :=
        func(value.contents, MatrixUtils.getValue(rowIndex, colIndex, matrix), rowIndex, colIndex)
    })
  })

  value.contents
}

let fillMatrix = (offsetLeft, offsetTop, sourceMatrix, fillMatrix) => {
  fillMatrix->reduceMatrix((sourceMatrix, value, rowIndex, colIndex) => {
    sourceMatrix->MatrixUtils.setValue(value, rowIndex + offsetTop, colIndex + offsetLeft)
  }, sourceMatrix)
}

let getMatrixRegion = (offsetLeft, offsetWidth, offsetTop, offsetHeight, matrix) => {
  matrix->reduceMatrix((regionMatrix, value, rowIndex, colIndex) => {
    rowIndex >= offsetTop &&
    rowIndex < offsetTop + offsetHeight &&
    colIndex >= offsetLeft &&
    colIndex < offsetLeft + offsetWidth
      ? {
          regionMatrix->MatrixUtils.setValue(value, rowIndex - offsetTop, colIndex - offsetLeft)
        }
      : {
          regionMatrix
        }
  }, Matrix.create(offsetHeight, offsetWidth, []))
}

let dot = ((row1, col1, data1), (row2, col2, data2)) => {
  col1 !== col1 || row1 !== row1
    ? Exception.throwErr("error")
    : {
        (row1, col1, data1->ArraySt.mapi((v, i) => v *. data2[i]))
      }
}

let _getMinNumber = %raw(` () => { return Number.MIN_VALUE } `)

let max = matrix => {
  matrix->reduceMatrix((result, value, _, _) => {
    Js.Math.max_float(result, value)
  }, _getMinNumber())
}

let getMaxIndex = matrix => {
  let (maxValue, rowIndexOpt, colIndexOpt) =
    matrix->reduceMatrix(
      ((maxValue, rowIndexOpt, colIndexOpt) as result, value, rowIndex, colIndex) => {
        value >= maxValue ? (value, Some(rowIndex), Some(colIndex)) : result
      },
      (_getMinNumber(), None, None),
    )

  (maxValue, rowIndexOpt->OptionSt.getExn, colIndexOpt->OptionSt.getExn)
}

let sum = matrix => {
  matrix->reduceMatrix((result, value, _, _) => {
    result +. value
  }, 0.)
}

let sumMatrixMap = matrixMap => {
  matrixMap
  ->ImmutableSparseMap.map((. matrix) => matrix->sum)
  ->ImmutableSparseMap.reducei((. result, value, _) => {
    result +. value
  }, 0.)
}

let rotate180 = matrix => {
  let row = matrix->Matrix.getRowCount
  let col = matrix->Matrix.getColCount

  matrix->reduceMatrix((result, value, rowIndex, colIndex) => {
    result->MatrixUtils.setValue(value, row - rowIndex - 1, col - colIndex - 1)
  }, Matrix.create(matrix->Matrix.getRowCount, matrix->Matrix.getColCount, []))
}

let addMatrixMap = (matrixMap1, matrixMap2) => {
  matrixMap1->ImmutableSparseMap.mapi((. matrix1, i) => {
    Matrix.add(matrix1, matrixMap2->ImmutableSparseMap.getExn(i))
  })
}

let mapMatrixMap = (matrixMap, func) => {
  matrixMap->ImmutableSparseMap.map((. matrix) => {
    func(matrix)
  })
}

let createMatrix = dataArr => {
  let rowCount = dataArr->ArraySt.length
  let colCount = dataArr[0]->ArraySt.length

  Matrix.create(rowCount, colCount, dataArr->ArraySt.reduceOneParam((. data, rowData) => {
      Js.Array.concat(rowData, data)
    }, []))
}

let createMatrixMapByDataArr = (dataArr): ImmutableSparseMapType.t<int, Matrix.t> => {
  dataArr->ArraySt.map(createMatrix)->Obj.magic
}

let getMatrixMapValue = (matrixMap, lengthIndex, rowIndex, colIndex) => {
  matrixMap->ImmutableSparseMap.getExn(lengthIndex)->MatrixUtils.getValue(rowIndex, colIndex, _)
}
