type element = {
  input: array<float>, // a 784-length array of floats representing each pixel of the 28 x 28 image, normalized between 0 and 1
  output: array<float>, // a 10-length binary array that tells which digits (from 0 to 9) is in that image
}

type data = {
  training: array<element>,
  test: array<element>,
}

@module("mnist")
external set: (int, int) => data = ""

// @module("mnist")
// external draw: (array<float>, Canvas.context) => unit = ""

let getMnistData = data => {
  data->ArraySt.map(({input}) => {
    input
  })
}

let getMnistLabels = data => {
  data->ArraySt.map(({output}) => {
    output
  })
}
