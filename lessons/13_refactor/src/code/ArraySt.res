let length = Js.Array.length

// let getExn = %raw(` (arr, index) => { let result = arr[index]; if (result !== undefined) { return result } else{throw new Error("Not_found") }} `)

let getExn = (arr, index) => {
  index <= arr->length - 1 && index >= 0
    ? Array.unsafe_get(arr, index)
    : Exception.throwErr("Not_found")
}

let getFirstExn = arr => getExn(arr, 0)

// let get = (arr, index) => {
//   index <= arr->length - 1 ? getExn(arr, index)->Some : None
// }

// let find = (arr, func) => Js.Array.find(func, arr)

// let includes = (arr, value) => Js.Array.includes(value, arr)

// // let includesByFunc = (arr, func) => {
// //   arr->find(func)->OptionSt.isSome
// // }

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

let copy = arr => arr->sliceFrom(0)

let reverse = %raw(` (arr) => { return arr.reverse()} `)
