module ArraySt = {
  let length = Js.Array.length

  let sliceFrom = (arr, index) => Js.Array.sliceFrom(index, arr)

  let reduceOneParam = (arr, func, param) => Belt.Array.reduceU(arr, param, func)

  let reduceOneParami = (arr, func, param) => {
    let mutableParam = ref(param)
    for i in 0 to Js.Array.length(arr) - 1 {
      mutableParam := func(. mutableParam.contents, Array.unsafe_get(arr, i), i)
    }
    mutableParam.contents
  }

  let range = (a: int, b: int) => {
    let result = []

    for i in a to b {
      Js.Array.push(i, result)->ignore
    }

    result
  }

  let map = (arr, func) => Js.Array.map(func, arr)

  let mapi = (arr, func) => Js.Array.mapi(func, arr)

  let push = (arr, value) => {
    Js.Array.push(value, arr)->ignore

    arr
  }

  let forEach = (arr, func) => Js.Array.forEach(func, arr)

  let forEachi = (arr, func) => Js.Array.forEachi(func, arr)
}

module MatrixUtils = {
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
}

module Vector = {
  type t = array<float>

  let create = arr => arr

  let push = (vec, value) => {
    vec->ArraySt.sliceFrom(0)->ArraySt.push(value)
  }

  let dot = (vec1, vec2) => {
    vec1->ArraySt.reduceOneParami((. sum, data1, i) => {
      sum +. data1 *. vec2[i]
    }, 0.)
  }

  let multiply = (vec1, vec2) => {
    vec1->ArraySt.reduceOneParami((. arr, data1, i) => {
      arr->ArraySt.push(data1 *. vec2[i])
    }, [])
  }

  let multiplyScalar = (vec, scalar) => {
    vec->ArraySt.reduceOneParam((. arr, data) => {
      arr->ArraySt.push(data *. scalar)
    }, [])
  }

  let add = (vec1, vec2) => {
    vec1->ArraySt.reduceOneParami((. arr, data1, i) => {
      arr->ArraySt.push(data1 +. vec2[i])
    }, [])
  }

  let sub = (vec1, vec2) => {
    vec1->ArraySt.reduceOneParami((. arr, data1, i) => {
      arr->ArraySt.push(data1 -. vec2[i])
    }, [])
  }

  let scalarSub = (scalar, vec) => {
    vec->ArraySt.reduceOneParam((. arr, data) => {
      arr->ArraySt.push(scalar -. data)
    }, [])
  }

  let length = vec => {
    vec->Js.Array.length
  }

  let transformMatrix = ((row, col, matrixData), vec) => {
    ArraySt.range(0, row - 1)->ArraySt.reduceOneParam((. arr, rowIndex) => {
      arr->ArraySt.push(dot(MatrixUtils.getRow(col, rowIndex, matrixData), vec))
    }, [])
  }

  let map = (vec, func) => {
    vec->ArraySt.map(func)
  }

  let mapi = (vec, func) => {
    vec->ArraySt.mapi(func)
  }

  let forEachi = (vec, func) => {
    vec->ArraySt.forEachi(func)
  }

  let reducei = (vec, func) => {
    vec->ArraySt.reduceOneParami(func)
  }

  let toArray = vec => {
    vec
  }
}

module Matrix = {
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
}

type state = {
  wMatrixBetweenLayer1Layer2: Matrix.t,
  wMatrixBetweenLayer2Layer3: Matrix.t,
}

type feature = {
  weight: float,
  height: float,
}

let _createWMatrix = (getValueFunc, firstLayerNodeCount, secondLayerNodeCount) => {
  Matrix.create(
    secondLayerNodeCount,
    firstLayerNodeCount,
    ArraySt.range(0, secondLayerNodeCount * firstLayerNodeCount - 1)->ArraySt.map(_ =>
      getValueFunc()
    ),
  )
}

let createState = (layer1NodeCount, layer2NodeCount, layer3NodeCount): state => {
  wMatrixBetweenLayer1Layer2: _createWMatrix(() => 0.1, layer1NodeCount + 1, layer2NodeCount),
  wMatrixBetweenLayer2Layer3: _createWMatrix(() => 0.1, layer2NodeCount + 1, layer3NodeCount),
}

let _activateFunc = x => {
  1. /. (1. +. Js.Math.exp(-.x))
}

let forward = (state: state, feature: feature) => {
  let inputVector = Vector.create([feature.height, feature.weight, 1.0])

  let layer2OutputVector =
    Vector.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector)->Vector.map(_activateFunc)

  let layer3OutputVector =
    Vector.transformMatrix(
      state.wMatrixBetweenLayer2Layer3,
      /*! 注意：此处push 1.0 */
      layer2OutputVector->Vector.push(1.0),
    )->Vector.map(_activateFunc)

  (layer2OutputVector, layer3OutputVector)
}

let state = createState(2, 2, 1)

let feature = {
  weight: 50.,
  height: 150.,
}

forward(state, feature)->Js.log
