let forward = x => {
  Js.Math.max_float(0., x)
}

let backward = x => {
  x > 0. ? 1. : 0.
}

/* ! get x => x = f-1(y) */
let invert = y => {
  y
}
