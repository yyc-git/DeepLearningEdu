type t = array<float>

let create = arr => arr

let dot = (vec1, vec2) => {
  vec1->ArraySt.reduceOneParami((. sum, data1, i) => {
    sum +. data1 *. vec2[i]
  }, 0.)
}