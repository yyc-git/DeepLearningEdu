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
