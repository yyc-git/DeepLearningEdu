let forward = x => {
  Js.Math.max_float(0., x)
}

let backward = x => {
  x > 0. ? 1. : 0.
}
