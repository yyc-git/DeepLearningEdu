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

// type gender =
//   | Male
//   | Female
//   | InValid

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

// let _activateFunc = x => x

// let _deriv_Linear = x => {
//   1.
// }

let _activateFunc = x => {
  1. /. (1. +. Js.Math.exp(-.x))
}

// TODO rename
let _deriv_Linear = x => {
  let fx = _activateFunc(x)

  fx *. (1. -. fx)
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

let _computeLoss = (labels, outputs) => {
  // Js.log((labels, outputs))
  labels->ArraySt.reduceOneParami((. result, label, i) => {
    result +. Js.Math.pow_float(~base=label -. outputs[i], ~exp=2.0)
  }, 0.)
   /. ArraySt.length(labels)->Obj.magic
}

let train = (state: state, features: array<feature>, labels: array<label>): state => {
  // let learnRate = 0.001
  // let epochs = 100000

  let learnRate = 0.1
  let epochs = 1000

  let n = features->ArraySt.length->Obj.magic

  ArraySt.range(0, epochs - 1)->ArraySt.reduceOneParam((. state, epoch) => {
    let state = features->ArraySt.reduceOneParami((. state, feature, i) => {
      let label = labels[i]->_convertLabelToFloat
      let x1 = feature.weight
      let x2 = feature.height

      let ((net3, net4, net5), (y3, y4, y5)) = forward(state, feature)

      let d_E_d_y5 = -2. /. n *. (label -. y5)
      // let d_E_d_y5 = -2.  *. (label -. y5)

      // Neuron o5
      let d_y5_d_w35 = y3 *. _deriv_Linear(net5)
      let d_y5_d_w45 = y4 *. _deriv_Linear(net5)
      let d_y5_d_b5 = _deriv_Linear(net5)

      let d_y5_d_y3 = state.weight35 *. _deriv_Linear(net5)
      let d_y5_d_y4 = state.weight45 *. _deriv_Linear(net5)

      // Neuron o3
      let d_y3_d_w13 = x1 *. _deriv_Linear(net3)
      let d_y3_d_w23 = x2 *. _deriv_Linear(net3)
      let d_y3_d_b3 = _deriv_Linear(net3)

      // Neuron o4
      let d_y4_d_w14 = x1 *. _deriv_Linear(net4)
      let d_y4_d_w24 = x2 *. _deriv_Linear(net4)
      let d_y4_d_b4 = _deriv_Linear(net4)

      // Update weights and biases

      {
        weight13: state.weight13 -. learnRate *. d_E_d_y5 *. d_y5_d_y3 *. d_y3_d_w13,
        weight14: state.weight14 -. learnRate *. d_E_d_y5 *. d_y5_d_y4 *. d_y4_d_w14,
        weight23: state.weight14 -. learnRate *. d_E_d_y5 *. d_y5_d_y3 *. d_y3_d_w23,
        weight24: state.weight24 -. learnRate *. d_E_d_y5 *. d_y5_d_y4 *. d_y4_d_w24,
        weight35: state.weight35 -. learnRate *. d_E_d_y5 *. d_y5_d_w35,
        weight45: state.weight45 -. learnRate *. d_E_d_y5 *. d_y5_d_w45,
        bias3: state.bias3 -. learnRate *. d_E_d_y5 *. d_y5_d_y3 *. d_y3_d_b3,
        bias4: state.bias4 -. learnRate *. d_E_d_y5 *. d_y5_d_y4 *. d_y4_d_b4,
        bias5: state.bias5 -. learnRate *. d_E_d_y5 *. d_y5_d_b5,
      }
    }, state)

    mod(epoch, 10) == 0
      ? {
          // Js.log(state)
          Js.log((
            "loss: ",
            _computeLoss(
              labels->ArraySt.map(_convertLabelToFloat),
              features->ArraySt.map(feature => {
                let (_, (_, _, y5)) = forward(state, feature)

                y5
              }),
            ),
          ))

          state
        }
      : state
  }, state)
}

let inference = (state: state, feature: feature) => {
  // Js.log(forward(state, feature))

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

let _mean = values => {
  values->ArraySt.reduceOneParam((. sum, value) => {
    sum +. value
  }, 0.) /. ArraySt.length(values)->Obj.magic
}

let _zeroMean = features => {
  // let weightMean = features->ArraySt.map(feature => feature.weight)->_mean
  // let heightMean = features->ArraySt.map(feature => feature.height)->_mean
  let weightMean = features->ArraySt.map(feature => feature.weight)->_mean->Js.Math.floor->Obj.magic
  let heightMean = features->ArraySt.map(feature => feature.height)->_mean->Js.Math.floor->Obj.magic

  features->ArraySt.map(feature => {
    weight: feature.weight -. weightMean,
    height: feature.height -. heightMean,
  })
}

let features = features->_zeroMean

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

featuresForInference->_zeroMean->Js.Array.forEach(feature => {
  inference(state, feature)->Js.log
}, _)