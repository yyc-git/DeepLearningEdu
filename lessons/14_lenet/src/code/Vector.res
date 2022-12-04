type t = array<float>

let create = arr => arr

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

let addScalar = (vec1, scalar) => {
  vec1->ArraySt.reduceOneParami((. arr, data1, i) => {
    arr->ArraySt.push(data1 +. scalar)
  }, [])
}

let sub = (vec1, vec2) => {
  vec1->ArraySt.reduceOneParami((. arr, data1, i) => {
    arr->ArraySt.push(data1 -. vec2[i])
  }, [])
}

// let build = (data, length) => {
//   ArraySt.range(0, length)->ArraySt.map(_ => data)
// }

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
    // MatrixUtils.getRow(col, rowIndex, matrixData)[0]->Log.printForDebug->ignore

    //     dot(MatrixUtils.getRow(col, rowIndex, matrixData), vec)->Log.printForDebug-> NullUtils.isEmpty->Log.printForDebug? {
    // //  MatrixUtils.getRow(col, rowIndex, matrixData)     ->Log.printForDebug->ignore
    // ()
    //     } : ()

    //     arr->ArraySt.push(dot(matrixData->Js.Array.slice(~start=1 * rowIndex, ~end_=col), vec))
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


let max = vec => {
  vec->reducei((. result, value, _) => {
    Js.Math.max_float(result, value)
  }, NumberUtils. getMinNumber())
}

let min = vec => {
  vec->reducei((. result, value, _) => {
    Js.Math.min_float(result, value)
  }, NumberUtils.getMaxNumber())
}

// let mean = vec => {
//   vec->reducei((. result, value, _) => {
//     Js.Math.min_float(result, value)
//   }, _getMaxNumber())
// }

let sum = vec => {
  vec->reducei((. result, value, _) => {
    result +. value
  }, 0.)
}

let getExn = (vec, index) => {
  index > vec->length ? Exception.throwErr("error") : Array.unsafe_get(vec, index)
}
