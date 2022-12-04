open MatrixUtils

// row first
type t = (int, int, array<float>)

let getData = ((_, _, data)) => {
  data
}

let getColCount = ((_, col, _)) => {
  col
}

let getRowCount = ((row, _, _)) => {
  row
}

let map = (matrix, func) => {
  (getRowCount(matrix), getColCount(matrix), getData(matrix)->ArraySt.map(func))
}

let mapi = (matrix, func) => {
  (getRowCount(matrix), getColCount(matrix), getData(matrix)->ArraySt.mapi(func))
}

let forEachRow = ((row, _, data), func) => {
  ArraySt.range(0, row - 1)->ArraySt.forEach(rowIndex => {
    func(rowIndex)
  })
}

let forEachCol = ((_, col, data), func) => {
  ArraySt.range(0, col - 1)->ArraySt.forEach(colIndex => {
    func(colIndex)
  })
}

let create = (row, col, data) => (row, col, data)

let transpose = ((row, col, data)) => {
  (col, row, ArraySt.range(0, row - 1)->ArraySt.reduceOneParam((. arr, rowIndex) => {
      ArraySt.range(0, col - 1)->ArraySt.reduceOneParam((. arr, colIndex) => {
        Array.unsafe_set(
          arr,
          MatrixUtils.computeIndex(row, colIndex, rowIndex),
          data[MatrixUtils.computeIndex(col, rowIndex, colIndex)],
        )

        arr
      }, arr)
    }, []))
}

let multiply = ((row1, col1, data1), (row2, col2, data2)) => {
  col1 !== row2
    ? Exception.throwErr("error")
    : {
        (row1, col2, ArraySt.range(0, row1 - 1)->ArraySt.reduceOneParam((. arr, rowIndex) => {
            let row = getRow(col1, rowIndex, data1)
            ArraySt.range(0, col2 - 1)->ArraySt.reduceOneParam((. arr, colIndex) => {
              arr->ArraySt.push(Vector.dot(row, getCol(row2, col2, colIndex, data2)))
            }, arr)
          }, []))
      }
}

let multiplyScalar = (scalar, (row, col, data)) => {
  (row, col, data->ArraySt.map(v => v *. scalar))
}

let add = ((row1, col1, data1), (row2, col2, data2)) => {
  col1 !== col2 || row1 !== row2
    ? Exception.throwErr("error")
    : {
        (row1, col1, data1->ArraySt.mapi((v, i) => v +. data2[i]))
      }
}

let sub = ((row1, col1, data1), (row2, col2, data2)) => {
  col1 !== col2 || row1 !== row2
    ? Exception.throwErr("error")
    : {
        (row1, col1, data1->ArraySt.mapi((v, i) => v -. data2[i]))
      }
}
