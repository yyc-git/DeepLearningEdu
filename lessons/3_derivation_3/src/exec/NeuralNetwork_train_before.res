module Neural_forward_answer = {
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
    (sampleData.height *. state.weight1 +. sampleData.weight *. state.weight2 +. state.bias)
      ->_activateFunc
  }

  let inference = (state: state, sampleData: sampleData): gender => {
    forward(state, sampleData)->_convert
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
}

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

let _deriv_Sigmoid = x => {
  let fx = _activateFunc(x)

  fx *. (1. -. fx)
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
  }, 0.) /. ArraySt.length(labels)->Obj.magic
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
      let d_y5_d_w53 = y3 *. _deriv_Sigmoid(net5)
      let d_y5_d_w54 = y4 *. _deriv_Sigmoid(net5)
      let d_y5_d_b5 = _deriv_Sigmoid(net5)

      let d_y5_d_y3 = state.weight53 *. _deriv_Sigmoid(net5)
      let d_y5_d_y4 = state.weight54 *. _deriv_Sigmoid(net5)

      // Neuron o3
      let d_y3_d_w31 = x1 *. _deriv_Sigmoid(net3)
      let d_y3_d_w32 = x2 *. _deriv_Sigmoid(net3)
      let d_y3_d_b3 = _deriv_Sigmoid(net3)

      // Neuron o4
      let d_y4_d_w41 = x1 *. _deriv_Sigmoid(net4)
      let d_y4_d_w42 = x2 *. _deriv_Sigmoid(net4)
      let d_y4_d_b4 = _deriv_Sigmoid(net4)

      // Update weights and biases

      {
        weight31: state.weight31 -. learnRate *. d_E_d_y5 *. d_y5_d_y3 *. d_y3_d_w31,
        weight41: state.weight41 -. learnRate *. d_E_d_y5 *. d_y5_d_y4 *. d_y4_d_w41,
        weight32: state.weight41 -. learnRate *. d_E_d_y5 *. d_y5_d_y3 *. d_y3_d_w32,
        weight42: state.weight42 -. learnRate *. d_E_d_y5 *. d_y5_d_y4 *. d_y4_d_w42,
        weight53: state.weight53 -. learnRate *. d_E_d_y5 *. d_y5_d_w53,
        weight54: state.weight54 -. learnRate *. d_E_d_y5 *. d_y5_d_w54,
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

// let inference = (state: state, feature: feature) => {
//   // Js.log(forward(state, feature))

//   let (_, (_, _, y5)) = forward(state, feature)

//   y5
// }

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

// let _mean = values => {
//   values->ArraySt.reduceOneParam((. sum, value) => {
//     sum +. value
//   }, 0.) /. ArraySt.length(values)->Obj.magic
// }

// let _zeroMean = features => {
//   // let weightMean = features->ArraySt.map(feature => feature.weight)->_mean
//   // let heightMean = features->ArraySt.map(feature => feature.height)->_mean
//   let weightMean = features->ArraySt.map(feature => feature.weight)->_mean->Js.Math.floor->Obj.magic
//   let heightMean = features->ArraySt.map(feature => feature.height)->_mean->Js.Math.floor->Obj.magic

//   features->ArraySt.map(feature => {
//     weight: feature.weight -. weightMean,
//     height: feature.height -. heightMean,
//   })
// }

// let features = features->_zeroMean

let state = state->train(features, labels)

Js.log(state)

// let featuresForInference = [
//   {
//     weight: 89.,
//     height: 190.,
//   },
//   {
//     weight: 60.,
//     height: 155.,
//   },
// ]

// featuresForInference->_zeroMean->Js.Array.forEach(feature => {
//   inference(state, feature)->Js.log
// }, _)
