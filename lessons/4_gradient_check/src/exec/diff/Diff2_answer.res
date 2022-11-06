let diff = (f: float => float, x: float): float => {
  let h = 1e-4

  (f(x +. h) -. f(x -. h)) /. (2.0 *. h)
}
