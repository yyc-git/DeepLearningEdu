type state = {
  weight13: float,
  weight14: float,
  weight23: float,
  weight24: float,
  weight35: float,
  weight45: float,
  bias3: float,
  bias4: float,
  bias5: float,
}

type feature = {
  weight: float,
  height: float,
}

type label =
  | Male
  | Female

module ArraySt = {
  let length = Js.Array.length

  let map = (arr, func) => Js.Array.map(func, arr)

  let range = (a: int, b: int) => {
    let result = []

    for i in a to b {
      Js.Array.push(i, result)->ignore
    }

    result
  }

  let reduceOneParam = (arr, func, param) => Belt.Array.reduceU(arr, param, func)

  let reduceOneParami = (arr, func, param) => {
    let mutableParam = ref(param)
    for i in 0 to Js.Array.length(arr) - 1 {
      mutableParam := func(. mutableParam.contents, Array.unsafe_get(arr, i), i)
    }
    mutableParam.contents
  }
}

module Neural_forward = {
  let forward = (state: Neural_forward_answer.state, feature: feature) => {
    let net = feature.height *. state.weight1 +. feature.weight *. state.weight2 +. state.bias

    (net, net->Neural_forward_answer._activateFunc)
  }
}

let createState = (): state => {
  weight13: Js.Math.random(),
  weight14: Js.Math.random(),
  weight23: Js.Math.random(),
  weight24: Js.Math.random(),
  weight35: Js.Math.random(),
  weight45: Js.Math.random(),
  bias3: Js.Math.random(),
  bias4: Js.Math.random(),
  bias5: Js.Math.random(),
}

let _activateFunc = x => x

let _deriv_linear = x => {
  1.
}

let forward = (state: state, feature: feature) => {
  let (net3, y3) = Neural_forward.forward(
    (
      {
        weight1: state.weight13,
        weight2: state.weight23,
        bias: state.bias3,
      }: Neural_forward_answer.state
    ),
    feature->Obj.magic,
  )

  let (net4, y4) = Neural_forward.forward(
    (
      {
        weight1: state.weight14,
        weight2: state.weight24,
        bias: state.bias4,
      }: Neural_forward_answer.state
    ),
    feature->Obj.magic,
  )

  let (net5, y5) = Neural_forward.forward(
    (
      {
        weight1: state.weight35,
        weight2: state.weight45,
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

  ((net3, net4, net5), (y3, y4, y5))
}

let _convertLabelToFloat = label =>
  switch label {
  | Male => 0.
  | Female => 1.
  }

let train = (state: state, features: array<feature>, labels: array<label>): state => {
  let learnRate = 0.1
  let epochs = 1000
  let n = features->ArraySt.length->Obj.magic

  ArraySt.range(0, epochs - 1)->ArraySt.reduceOneParam((. state, epoch) => {
    // TODO implement
    state
  }, state)
}

let inference = (state: state, feature: feature) => {
  let (_, (_, _, y5)) = forward(state, feature)

  y5
}

let state = createState()

let features = [
  {
    weight: 50.,
    height: 150.,
  },
  {
    weight: 51.,
    height: 149.,
  },
  {
    weight: 60.,
    height: 172.,
  },
  {
    weight: 90.,
    height: 188.,
  },
]

let labels = [Female, Female, Male, Male]

let state = state->train(features, labels)

let featuresForInference = [
  {
    weight: 89.,
    height: 190.,
  },
  {
    weight: 60.,
    height: 155.,
  },
]

featuresForInference->Js.Array.forEach(feature => {
  inference(state, feature)->Js.log
}, _)
