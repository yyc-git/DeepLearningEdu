let computeIndex = (colCount, rowIndex, colIndex) => {
  rowIndex * colCount + colIndex
}

let getRow = (colCount, rowIndex, data) => {
  data->Js.Array.slice(~start=rowIndex * colCount, ~end_=(rowIndex + 1) * colCount)
}

let getCol = (rowCount, colCount, colIndex, data) => {
  ArraySt.range(0, rowCount - 1)->ArraySt.reduceOneParam((. arr, rowIndex) => {
    arr->ArraySt.push(data[computeIndex(colCount, rowIndex, colIndex)])
  }, [])
}

let getValue = (rowIndex, colIndex, (row, col, data)) => {
  data[computeIndex(col, rowIndex, colIndex)]
}

let setValue = ((row, col, data), value, rowIndex, colIndex) => {
  Array.unsafe_set(data, computeIndex(col, rowIndex, colIndex), value)

  (row, col, data)
}
