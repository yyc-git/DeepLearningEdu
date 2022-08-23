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

type sampleData = {
  weight: float,
  height: float,
}

type gender =
  | Male
  | Female
  | InValid

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

// not implement
let train = (state: state, allSampleData: array<sampleData>): state => {
  Obj.magic(1)
}

let _activateFunc = x => x

let _convert = x =>
  switch x {
  | 0. => Male
  | 1. => Female
  | _ => InValid
  }

let forward = (state: state, sampleData: sampleData): float => {
  //TODO implement
  Obj.magic(1)
}

let inference = (state: state, sampleData: sampleData): gender => {
  forward(state, sampleData)->_activateFunc->_convert
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
