let createEmpty = SparseMap.createEmpty

let createFromArr = arr => arr->SparseMapType.arrayNotNullableToArrayNullable

let length = SparseMap.length

let copy = SparseMap.copy

let unsafeGet = SparseMap.unsafeGet

let get = SparseMap.get

let getNullable = SparseMap.getNullable

let getExn = SparseMap.getExn

let has = SparseMap.has

let set = (map, key: int, value) => {
  let newMap = map->copy

  Array.unsafe_set(newMap, key, value->SparseMapType.notNullableToNullable)

  newMap
}

let remove = (map, key: int) => {
  let newMap = map->copy

  Array.unsafe_set(newMap, key, Js.Nullable.undefined)

  newMap
}

let map = SparseMap.map

let mapi = SparseMap.mapi

let reducei = SparseMap.reducei

let getValues = SparseMap.getValues

let getKeys = SparseMap.getKeys

let deleteVal = (map, key: int) => {
  let newMap = map->copy

  Array.unsafe_set(newMap, key, Js.Nullable.undefined)

  newMap
}

let forEachi = SparseMap.forEachi
