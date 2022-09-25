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
  //TODO implement
  Obj.magic(1)
}

let train = (state: state, sampleData: sampleData): state => {
  //TODO implement
  Obj.magic(1)
}

let inference = (state: state, sampleData: sampleData): gender => {
  //TODO implement
  Obj.magic(1)
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


//1
Js.log(gender)
