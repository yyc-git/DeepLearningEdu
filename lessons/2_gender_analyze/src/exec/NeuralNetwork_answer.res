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

type sampleData = {
  weight: float,
  height: float,
}

type gender =
  | Male
  | Female
  | InValid

let createState = (): state => {
  weight31: Js.Math.random(),
  weight41: Js.Math.random(),
  weight32: Js.Math.random(),
  weight42: Js.Math.random(),
  weight53: Js.Math.random(),
  weight54: Js.Math.random(),
  bias3: Js.Math.random(),
  bias4: Js.Math.random(),
  bias5: Js.Math.random(),
}

// not implement
let train = (state: state, allSampleData: array<sampleData>): state => {
  // Obj.magic(1)
  state
}

let _activateFunc = x => x

let _convert = x =>
  switch x {
  | 0. => Male
  | 1. => Female
  | _ => InValid
  }

let forward = (state: state, sampleData: sampleData): float => {
  let y3 = Neural_forward_answer.forward(
    (
      {
        weight1: state.weight31,
        weight2: state.weight32,
        bias: state.bias3,
      }: Neural_forward_answer.state
    ),
    sampleData->Obj.magic,
  )

  let y4 = Neural_forward_answer.forward(
    (
      {
        weight1: state.weight41,
        weight2: state.weight42,
        bias: state.bias4,
      }: Neural_forward_answer.state
    ),
    sampleData->Obj.magic,
  )

  Neural_forward_answer.forward(
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
      }: Neural_forward_answer.sampleData
    ),
  )
}

let inference = (state: state, sampleData: sampleData): gender => {
  Js.log(forward(state, sampleData))

  forward(state, sampleData)->_convert
}

let state = createState()

let allSampleData = [
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

let state = state->train(allSampleData)

allSampleData->Js.Array.forEach(sampleData => {
  inference(state, sampleData)->Js.log
}, _)
