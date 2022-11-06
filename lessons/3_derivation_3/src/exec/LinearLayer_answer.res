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

type layer2Gradient = Matrix.t
type layer3Gradient = Matrix.t

let _createWMatrix = (getValueFunc, firstLayerNodeCount, secondLayerNodeCount) => {
  let row = secondLayerNodeCount
  let col = firstLayerNodeCount + 1

  Matrix.create(row, col, ArraySt.range(0, row * col - 1)->ArraySt.map(_ => getValueFunc()))
}

let createState = (layer1NodeCount, layer2NodeCount, layer3NodeCount): state => {
  wMatrixBetweenLayer1Layer2: _createWMatrix(Js.Math.random, layer1NodeCount, layer2NodeCount),
  wMatrixBetweenLayer2Layer3: _createWMatrix(Js.Math.random, layer2NodeCount, layer3NodeCount),
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
): (layer2Gradient, layer3Gradient) => {
  let labelVector = Vector.create([label])

  let layer3Delta = layer3OutputVector->Vector.mapi((layer3OutputValue, i) => {
    let d_E_d_value = -2. /. n *. (labelVector -> Vector.getExn(i) -. layer3OutputValue)

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

  (layer2Gradient, layer3Gradient)


}

let _convertLabelToFloat = label =>
  switch label {
  | Male => 0.
  | Female => 1.
  }

let _computeLoss = (labels, outputs) => {
  labels->ArraySt.reduceOneParami((. result, label, i) => {
    result +. Js.Math.pow_float(~base=label -. outputs[i], ~exp=2.0)
  }, 0.) /. ArraySt.length(labels)->Obj.magic
}

let _createInputVector = (feature: feature) => {
  Vector.create([feature.height, feature.weight, 1.0])
}

let train = (state: state, features: array<feature>, labels: array<label>): state => {
  let learnRate = 0.1
  let epochs = 1000

  let n = features->ArraySt.length->Obj.magic

  ArraySt.range(0, epochs - 1)->ArraySt.reduceOneParam((. state, epoch) => {
    let state = features->ArraySt.reduceOneParami((. state, feature, i) => {
      let label = labels[i]->_convertLabelToFloat

      let inputVector = _createInputVector(feature)

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

let inference = (state: state, feature: feature) => {
  let inputVector = _createInputVector(feature)

  let (_, (_, layer3OutputVector)) = forward(inputVector, state)

  let y5 = layer3OutputVector->Vector.getExn(0)

  y5
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

let _mean = values => {
  values->ArraySt.reduceOneParam((. sum, value) => {
    sum +. value
  }, 0.) /. ArraySt.length(values)->Obj.magic
}

let _zeroMean = features => {
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
