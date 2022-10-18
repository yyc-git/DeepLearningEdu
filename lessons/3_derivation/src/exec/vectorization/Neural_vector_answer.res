type state = {wVector: Vector.t}

type sampleData = {
  weight: float,
  height: float,
}

let createState = (inputCount): state => {
  {
    wVector: Vector.create(
      ArraySt.range(0, inputCount + 1 - 1)->ArraySt.map(_ => Js.Math.random()),
    ),
  }
}

let train = (state: state, sampleData: sampleData): state => {
  {wVector: Vector.create([1.0, -2.0, -49.0])}
}

let _activateFunc = x => {
  1. /. (1. +. Js.Math.exp(-.x))
}

let forward = (state: state, sampleData: sampleData): float => {
  let inputVector = Vector.create([sampleData.height, sampleData.weight, 1.0])

  Vector.dot(state.wVector, inputVector)->_activateFunc
}

let state = createState(2)

Js.log(
  state
  ->train({
    weight: 50.,
    height: 150.,
  })
  ->forward({
    weight: 50.,
    height: 150.,
  }),
)
