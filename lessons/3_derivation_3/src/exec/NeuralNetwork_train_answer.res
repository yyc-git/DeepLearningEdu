type state = {
  wMatrixBetweenLayer1Layer2: Matrix.t,
  wMatrixBetweenLayer2Layer3: Matrix.t,
}

type feature = {
  weight: float,
  height: float,
}

type label =
  | Male
  | Female

type forwardOutput = ((Vector.t, Vector.t), (Vector.t, Vector.t))

let _createWMatrix = (getValueFunc, firstLayerNodeCount, secondLayerNodeCount) => {
  let row = secondLayerNodeCount
  let col = firstLayerNodeCount + 1

  Matrix.create(row, col, ArraySt.range(0, row * col - 1)->ArraySt.map(_ => getValueFunc()))
}

let createState = (layer1NodeCount, layer2NodeCount, layer3NodeCount): state => {
  wMatrixBetweenLayer1Layer2: _createWMatrix(() => 0.1, layer1NodeCount, layer2NodeCount),
  wMatrixBetweenLayer2Layer3: _createWMatrix(() => 0.1, layer2NodeCount, layer3NodeCount),
}

let _activateFunc = x => {
  1. /. (1. +. Js.Math.exp(-.x))
}

let _deriv_Sigmoid = x => {
  let fx = _activateFunc(x)

  fx *. (1. -. fx)
}

let forward = (inputVector: Vector.t, state: state): forwardOutput => {
  let layer2Net = Vector.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector)

  let layer2OutputVector = layer2Net->Vector.map(_activateFunc)

  let layer3Net = Vector.transformMatrix(
    state.wMatrixBetweenLayer2Layer3,
    /* ! 注意：此处push 1.0 */
    layer2OutputVector->Vector.push(1.0),
  )

  let layer3OutputVector = layer3Net->Vector.map(_activateFunc)

  ((layer2Net, layer2OutputVector), (layer3Net, layer3OutputVector))
}

let backward = (
  ((layer2Net, layer2OutputVector), (layer3Net, layer3OutputVector)): forwardOutput,
  n: float,
  label: float,
  inputVector: Vector.t,
  state: state,
): // ): ((Vector.t, Matrix.t), (Vector.t, Matrix.t)) => {
(Matrix.t, Matrix.t) => {
  // let d_E_d_y5 = -2. /. n *. (label -. y5)

  // let d_y_net5 = _deriv_Sigmoid(layer3Net->Vector.getExn(0))

  // let y5_delta = d_E_d_y5 *. d_y_net5

  // let layer3Delta = Vector.create([y5_delta])

  // let d_y_net3 = _deriv_Sigmoid(layer2Net->Vector.getExn(0))

  // let y3_delta =
  //   y5_delta *. MatrixUtils.getValue(0, 0, state.wMatrixBetweenLayer2Layer3) *. d_y_net3

  // let d_y_net4 = _deriv_Sigmoid(layer2Net->Vector.getExn(1))

  // let y4_delta =
  //   y5_delta *. MatrixUtils.getValue(0, 1, state.wMatrixBetweenLayer2Layer3) *. d_y_net4

  // let layer2Delta = Vector.create([y3_delta, y4_delta])

  let layer3Delta = layer3OutputVector->Vector.mapi((layer3OutputValue, i) => {
    let d_E_d_value = -2. /. n *. (label -. layer3OutputValue)

    let d_y_net_value = _deriv_Sigmoid(layer3Net->Vector.getExn(i))

    d_E_d_value *. d_y_net_value
  })

  let layer2Delta = layer2Net->Vector.mapi((layer2NetValue, i) => {
    Vector.dot(
      layer3Delta,
      MatrixUtils.getCol(
        Matrix.getRowCount(state.wMatrixBetweenLayer2Layer3),
        Matrix.getColCount(state.wMatrixBetweenLayer2Layer3),
        i,
        Matrix.getData(state.wMatrixBetweenLayer2Layer3),
      ),
    ) *.
    _deriv_Sigmoid(layer2NetValue)
  })


  let layer2Gradient = Matrix.multiply(
    Matrix.create(Vector.length(layer2Delta), 1, layer2Delta),
    Matrix.create(1, Vector.length(inputVector), inputVector),
  )

  /* ! 注意：此处push 1.0 */
  let layer2OutputVector = layer2OutputVector->Vector.push(1.0)

  let layer3Gradient = Matrix.multiply(
    Matrix.create(Vector.length(layer3Delta), 1, layer3Delta),
    Matrix.create(1, Vector.length(layer2OutputVector), layer2OutputVector),
  )


  // Js.log((layer2Delta, layer3Delta))
  // Js.log(layer3Gradient)
  // Js.log(layer2Gradient)

  // ((layer2Delta, layer2Gradient), (layer3Delta, layer3Gradient))
  (layer2Gradient, layer3Gradient)
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

let _createInputVector = (feature: feature) => {
  Vector.create([feature.height, feature.weight, 1.0])
}

let train = (state: state, features: array<feature>, labels: array<label>): state => {
  // let learnRate = 0.001
  // let epochs = 100000

  let learnRate = 0.1
  let epochs = 1000
  // let epochs = 1

  let n = features->ArraySt.length->Obj.magic

  ArraySt.range(0, epochs - 1)->ArraySt.reduceOneParam((. state, epoch) => {
    let state = features->ArraySt.reduceOneParami((. state, feature, i) => {
      let label = labels[i]->_convertLabelToFloat

      let inputVector = _createInputVector(feature)

      // Js.log(forward(inputVector, state))

      let (layer2Gradient, layer3Gradient) =
        forward(inputVector, state)->backward(n, label, inputVector, state)

      {
        wMatrixBetweenLayer1Layer2: Matrix.sub(
          state.wMatrixBetweenLayer1Layer2,
          layer2Gradient->Matrix.multiplyScalar(learnRate, _),
        ),
        wMatrixBetweenLayer2Layer3: Matrix.sub(
          state.wMatrixBetweenLayer2Layer3,
          layer3Gradient->Matrix.multiplyScalar(learnRate, _),
        ),
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
                let inputVector = _createInputVector(feature)

                let (_, (_, layer3OutputVector)) = forward(inputVector, state)

                // TODO fix
                let y5 = layer3OutputVector->Vector.getExn(0)

                y5
              }),
            ),
          ))

          state
        }
      : state
  }, state)
}

let state = createState(2, 2, 1)

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
// let labels = [Female]

let state = state->train(features, labels)

// Js.log(state)