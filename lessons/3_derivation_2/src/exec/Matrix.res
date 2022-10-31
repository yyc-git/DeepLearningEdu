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

  let multiplyScalar = (scalar, (row, col, data)) => {
    (row, col, data->ArraySt.map(v => v *. scalar))
  }

  let map = (matrix, func) => {
    (getRowCount(matrix), getColCount(matrix), getData(matrix)->ArraySt.map(func))
  }

  let mapi = (matrix, func) => {
    (getRowCount(matrix), getColCount(matrix), getData(matrix)->ArraySt.mapi(func))
  }
