let normal = (randomFunc, fanIn: int, _fanOut: int) => {
  let std = Js.Math.sqrt(2. /. fanIn->Obj.magic)

  InitValue.normal(randomFunc, 0.0, std)
}
