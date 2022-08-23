type state = {
  weight1: float,
  weight2: float,
  bias: float,
}

type sampleData = {
  weight: float,
  height: float,
}

type gender =
  | Male
  | Female

let createState = (): state => {
  weight1: Js.Math.random(),
  weight2: Js.Math.random(),
  bias: Js.Math.random(),
}

let train = (state: state, sampleData: sampleData): state => {
  {
    weight1: 1.0,
    weight2: -2.0,
    bias: -49.0,
  }
}

let _activateFunc = x => x

let _convert = x =>
  switch x {
  | 0. => Male
  | 1. => Female
  }

let forward = (state: state, sampleData: sampleData): float => {
  sampleData.height *. state.weight1 +. sampleData.weight *. state.weight2 +. state.bias
}

let inference = (state: state, sampleData: sampleData): gender => {
  forward(state, sampleData)->_activateFunc->_convert
}

let state = createState()

let gender =
  state
  ->train({
    weight: 50.,
    height: 150.,
  })
  ->inference({
    weight: 50.,
    height: 150.,
  })
