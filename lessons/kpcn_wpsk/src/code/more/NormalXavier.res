let normal = (randomFunc, fanIn: int, fanOut: int) => {
  let std = Js.Math.sqrt(2. /. (fanIn + fanOut)->Obj.magic)

  InitValue.normal(randomFunc, 0.0, std)
}
