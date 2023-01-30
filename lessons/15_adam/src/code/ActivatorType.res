type data = {
  forwardNet: Vector.t => Vector.t,
  forwardMatrix: Matrix.t => Matrix.t,
  backward: float => float,
  invert: float => float,
}
