let unsafeGet = Belt.Option.getUnsafe

let _getExn = %raw(`



(nullableData) => {



  if (nullableData !== undefined) {



    return nullableData;



  }







  throw new Error("Not_found")



}



`)

// let getExn = Belt.Option.getExn
let getExn = (optionData: option<'a>): 'a => {
  optionData->unsafeGet->_getExn
}

let getWithDefault = Belt.Option.getWithDefault

let map = Belt.Option.map

let bind = Belt.Option.flatMap