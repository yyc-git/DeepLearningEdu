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
