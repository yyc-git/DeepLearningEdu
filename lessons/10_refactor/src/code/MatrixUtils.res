let computeIndex = (colCount, rowIndex, colIndex) => {
  rowIndex * colCount + colIndex
}

let getRow = (colCount, rowIndex, data) => {
  data->Js.Array.slice(~start=rowIndex * colCount, ~end_=(rowIndex + 1) * colCount)
}

let getCol = (rowCount, colCount, colIndex, data) => {
  ArraySt.range(0, rowCount - 1)->ArraySt.reduceOneParam((. arr, rowIndex) => {
    arr->ArraySt.push(ArraySt.getExn(data, computeIndex(colCount, rowIndex, colIndex)))
  }, [])
}

let getValueByIndex = ((row, col, data), index) => {
  ArraySt.getExn(data, index)
}

let getValue = ((row, col, data) as matrix, rowIndex, colIndex) => {
  getValueByIndex(matrix, computeIndex(col, rowIndex, colIndex))
}

let setValue = ((row, col, data), value, rowIndex, colIndex) => {
  Array.unsafe_set(data, computeIndex(col, rowIndex, colIndex), value)

  (row, col, data)
}
