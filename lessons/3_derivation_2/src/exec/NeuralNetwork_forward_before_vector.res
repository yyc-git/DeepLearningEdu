type state = {
  weight31: float,
  weight41: float,
  weight32: float,
  weight42: float,
  weight53: float,
  weight54: float,
  bias3: float,
  bias4: float,
  bias5: float,
}

type feature = {
  weight: float,
  height: float,
}

let createState = (): state => {
  // weight31: Js.Math.random(),
  // weight41: Js.Math.random(),
  // weight32: Js.Math.random(),
  // weight42: Js.Math.random(),
  // weight53: Js.Math.random(),
  // weight54: Js.Math.random(),
  // bias3: Js.Math.random(),
  // bias4: Js.Math.random(),
  // bias5: Js.Math.random(),
  weight31: 0.1,
  weight41: 0.1,
  weight32: 0.1,
  weight42: 0.1,
  weight53: 0.1,
  weight54: 0.1,
  bias3: 0.1,
  bias4: 0.1,
  bias5: 0.1,
}

let _activateFunc = x => {
  1. /. (1. +. Js.Math.exp(-.x))
}

module Neural_forward_answer = {
  type state = {
    weight1: float,
    weight2: float,
    bias: float,
  }
}

module Neural_forward = {
  let forward = (state: Neural_forward_answer.state, feature: feature) => {
    let net = feature.height *. state.weight1 +. feature.weight *. state.weight2 +. state.bias

    // (net, net->Neural_forward_answer._activateFunc)
    (net, net->_activateFunc)
  }
}

let forward = (state: state, feature: feature) => {
  let (net3, y3) = Neural_forward.forward(
    (
      {
        weight1: state.weight31,
        weight2: state.weight32,
        bias: state.bias3,
      }: Neural_forward_answer.state
    ),
    feature->Obj.magic,
  )

  let (net4, y4) = Neural_forward.forward(
    (
      {
        weight1: state.weight41,
        weight2: state.weight42,
        bias: state.bias4,
      }: Neural_forward_answer.state
    ),
    feature->Obj.magic,
  )

  let (net5, y5) = Neural_forward.forward(
    (
      {
        weight1: state.weight53,
        weight2: state.weight54,
        bias: state.bias5,
      }: Neural_forward_answer.state
    ),
    (
      {
        weight: y3,
        height: y4,
      }: feature
    ),
  )

  // ((net3, net4, net5), (y3, y4, y5))
  (y3, y4, y5)
}

let state = createState()

let feature = {
  weight: 50.,
  height: 150.,
}

forward(state, feature)->Js.log
