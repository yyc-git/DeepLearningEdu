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

let _deriv_sigmoid = x => {
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

let _bpLayer3Delta = (layer3Net, layer3OutputVector, n, labelVector) => {
  layer3OutputVector->Vector.mapi((layer3OutputValue, i) => {
    let d_E_d_value = -2. /. n *. (labelVector->Vector.getExn(i) -. layer3OutputValue)

    let d_y_net_value = _deriv_sigmoid(layer3Net->Vector.getExn(i))

    d_E_d_value *. d_y_net_value
  })
}

let backward = (
  ((layer2Net, layer2OutputVector), (layer3Net, layer3OutputVector)): forwardOutput,
  n: float,
  label: float,
  inputVector: Vector.t,
  state: state,
): (layer2Gradient, layer3Gradient) => {
  let labelVector = Vector.create([label])

  // let layer3Delta = layer3OutputVector->Vector.mapi((layer3OutputValue, i) => {
  //   let d_E_d_value = -2. /. n *. (labelVector->Vector.getExn(i) -. layer3OutputValue)

  //   let d_y_net_value = _deriv_sigmoid(layer3Net->Vector.getExn(i))

  //   d_E_d_value *. d_y_net_value
  // })

  let layer3Delta = _bpLayer3Delta(layer3Net, layer3OutputVector, n, labelVector)

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
    _deriv_sigmoid(layer2NetValue)
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

let checkGradient = (inputVector: Vector.t, labelVector: Vector.t): unit => {
  let _checkWeight = (
    (computeError, updateWMatrixByAddEpsilon, updateWMatrixBySubEpsilon),
    delta,
    (inputVector, previousLayerNodeOutput),
    state,
  ) => {
    let actualGradient = delta *. previousLayerNodeOutput

    /* ! update weight by add */

    let epsilon = 10e-4

    let newState1 = updateWMatrixByAddEpsilon(state, epsilon)

    /* ! compute error1 */

    let (_, (_, layer3OutputVector)) = forward(inputVector, newState1)

    let error1 = computeError(layer3OutputVector)

    /* ! update weight by sub */

    let newState2 = updateWMatrixBySubEpsilon(state, epsilon)

    let (_, (_, layer3OutputVector)) = forward(inputVector, newState2)

    /* ! compute error2 */

    let error2 = computeError(layer3OutputVector)

    /* ! compare */

    let expectedGradient = (error1 -. error2) /. (2. *. epsilon)

    Js.log((
      "check gradient: ",
      // expectedGradient,
      // actualGradient,
      // delta,
      // previousLayerNodeOutput,
      FloatUtils.truncateFloatValue(expectedGradient, 4) ==
        FloatUtils.truncateFloatValue(actualGradient, 4),
    ))
  }

  let _check = (
    (updateWMatrix, computeError, checkWeight),
    wMatrix,
    deltaVector,
    inputVector,
    previousLayerOutputVector,
    state,
  ) => {
    wMatrix->Matrix.forEachRow(rowIndex => {
      wMatrix->Matrix.forEachCol(colIndex => {
        checkWeight(
          (
            computeError,
            (state, epsilon) => {
              let (row, col, data) = wMatrix
              let data = data->Js.Array.copy

              data[
                MatrixUtils.computeIndex(col, rowIndex, colIndex)
              ] =
                data[MatrixUtils.computeIndex(col, rowIndex, colIndex)] +. epsilon

              updateWMatrix(state, (row, col, data))
            },
            (state, epsilon) => {
              let (row, col, data) = wMatrix
              let data = data->Js.Array.copy

              data[
                MatrixUtils.computeIndex(col, rowIndex, colIndex)
              ] =
                data[MatrixUtils.computeIndex(col, rowIndex, colIndex)] -. epsilon

              updateWMatrix(state, (row, col, data))
            },
          ),
          deltaVector[rowIndex],
          (inputVector, previousLayerOutputVector->Vector.getExn(colIndex)),
          state,
        )
      })
    })
  }

  let _computeErrorForLayer3 = (labelVector, outputVector) =>
    _computeLoss(labelVector->Vector.toArray, outputVector->Vector.toArray)

  /* ! forward + backward to compute layer3 delta */

  let state = createState(2, 2, 1)

  let ((_, layer2OutputVector), (layer3Net, layer3OutputVector)) = forward(inputVector, state)

  let n = 1.0

  let layer3Delta = _bpLayer3Delta(layer3Net, layer3OutputVector, n, labelVector)

  _check(
    (
      (state, wMatrix) => {
        ...state,
        wMatrixBetweenLayer2Layer3: wMatrix,
      },
      _computeErrorForLayer3(labelVector),
      _checkWeight,
    ),
    state.wMatrixBetweenLayer2Layer3,
    layer3Delta,
    inputVector,
    layer2OutputVector->Vector.push(1.0),
    state,
  )
}

let testCheckOutputLayerGradient = () => {
  let inputVector = [-2., -1., 1.]->Vector.create
  let labelVector = [Female]->Vector.create->Vector.map(_convertLabelToFloat)

  checkGradient(inputVector, labelVector)
}

Js.log("begin test")
testCheckOutputLayerGradient()
Js.log("finish test")
