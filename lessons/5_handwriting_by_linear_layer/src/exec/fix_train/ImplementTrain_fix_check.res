type state = {
  wMatrixBetweenLayer1Layer2: Matrix.t,
  wMatrixBetweenLayer2Layer3: Matrix.t,
}

type feature = array<float>

type forwardOutput = ((Vector.t, Vector.t), (Vector.t, Vector.t))

type layer2Gradient = Matrix.t
type layer3Gradient = Matrix.t

let _createWMatrix = (getValue, firstLayerNodeCount, secondLayerNodeCount) => {
  let row = secondLayerNodeCount
  let col = firstLayerNodeCount + 1

  Matrix.create(row, col, ArraySt.range(0, row * col - 1)->ArraySt.map(_ => getValue()))
}

let createState = (layer1NodeCount, layer2NodeCount, layer3NodeCount): state => {
  wMatrixBetweenLayer1Layer2: _createWMatrix(Js.Math.random, layer1NodeCount, layer2NodeCount),
  wMatrixBetweenLayer2Layer3: _createWMatrix(Js.Math.random, layer2NodeCount, layer3NodeCount),
}

let _handleInputValueToAvoidTooLargeForSigmoid = (max, inputValue) => {
  inputValue /. (max->Obj.magic /. 10.)
}

let _activate_sigmoid = (handleInputValueToAvoidTooLargeForSigmoid, x) => {
  let x = x->handleInputValueToAvoidTooLargeForSigmoid

  DebugUtils.checkSigmoidInputTooLarge(x)

  1. /. (1. +. Js.Math.exp(-.x))
}

let _deriv_sigmoid = (handleInputValueToAvoidTooLargeForSigmoid, x) => {
  let fx = _activate_sigmoid(handleInputValueToAvoidTooLargeForSigmoid, x)

  fx *. (1. -. fx)
}

let _activate_linear = x => {
  x
}

let _deriv_linear = x => {
  1.0
}

let _forwardLayer2 = (activate, inputVector, state) => {
  let layerNet = Vector.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector)

  let layerOutputVector = layerNet->Vector.map(activate)

  (layerNet, layerOutputVector)
}

let _forwardLayer3 = (activate, layer2OutputVector, state) => {
  let layerNet = Vector.transformMatrix(
    state.wMatrixBetweenLayer2Layer3,
    /* ! 注意：此处push 1.0 */
    layer2OutputVector->Vector.push(1.0),
  )

  let layerOutputVector = layerNet->Vector.map(activate)

  (layerNet, layerOutputVector)
}

let forward = (
  (layer2Activate, layer3Activate),
  inputVector: Vector.t,
  state: state,
): forwardOutput => {
  let (layer2Net, layer2OutputVector) = _forwardLayer2(layer2Activate, inputVector, state)

  let (layer3Net, layer3OutputVector) = _forwardLayer3(layer3Activate, layer2OutputVector, state)

  ((layer2Net, layer2OutputVector), (layer3Net, layer3OutputVector))
}

let _bpLayer3Delta = (deriv, layer3Net, layer3OutputVector, n, labelVector) => {
  layer3OutputVector->Vector.mapi((layer3OutputValue, i) => {
    let d_E_d_value = -2. /. n *. (labelVector->Vector.getExn(i) -. layer3OutputValue)

    let d_y_net_value = deriv(layer3Net->Vector.getExn(i))

    d_E_d_value *. d_y_net_value
  })
}

let _bpLayer2Delta = (deriv, layer2Net, layer3Delta, state) => {
  layer2Net->Vector.mapi((layer2NetValue, i) => {
    Vector.dot(
      layer3Delta,
      MatrixUtils.getCol(
        Matrix.getRowCount(state.wMatrixBetweenLayer2Layer3),
        Matrix.getColCount(state.wMatrixBetweenLayer2Layer3),
        i,
        Matrix.getData(state.wMatrixBetweenLayer2Layer3),
      ),
    ) *.
    deriv(layer2NetValue)
  })
}

let backward = (
  ((layer2Net, layer2OutputVector), (layer3Net, layer3OutputVector)): forwardOutput,
  n: float,
  labelVector: Vector.t,
  inputVector: Vector.t,
  state: state,
): (layer2Gradient, layer3Gradient) => {
  let layer3Delta = _bpLayer3Delta(
    _deriv_sigmoid(
      _handleInputValueToAvoidTooLargeForSigmoid(
        Matrix.getColCount(state.wMatrixBetweenLayer2Layer3),
      ),
    ),
    layer3Net,
    layer3OutputVector,
    n,
    labelVector,
  )

  let layer2Delta = _bpLayer2Delta(
    _deriv_sigmoid(
      _handleInputValueToAvoidTooLargeForSigmoid(
        Matrix.getColCount(state.wMatrixBetweenLayer1Layer2),
      ),
    ),
    layer2Net,
    layer3Delta,
    state,
  )

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

let _createInputVector = (feature: feature) => {
  feature->Vector.create->Vector.push(1.0)
}

let _getOutputNumber = outputVector => {
  let (_, maxValueIndex) =
    outputVector->Vector.reducei((. (maxValue, maxValueIndex), value, index) => {
      value >= maxValue ? (value, Some(index)) : (maxValue, maxValueIndex)
    }, (0., None))

  maxValueIndex->OptionSt.getExn
}

let _isCorrectInference = (labelVector, predictVector) => {
  _getOutputNumber(labelVector) == _getOutputNumber(predictVector)
}

let _getCorrectRate = (correctCount, errorCount) => {
  (correctCount->Obj.magic /. (correctCount->Obj.magic +. errorCount->Obj.magic) *. 100.)
    ->Obj.magic ++ "%"
}

let _checkSampleCount = sampleCount => {
  sampleCount < 10 ? Exception.throwErr("error") : ()
}

let train = (state: state, sampleCount: int): state => {
  _checkSampleCount(sampleCount)

  // let layer2LearnRate = 0.1
  let layer2LearnRate = 10.0
  let layer3LearnRate = 10.0
  // let layer3LearnRate = 1.0
  // let learnRate = 0.1
  let epochs = 50

  let mnistData = Mnist.set(sampleCount, 1)

  let features = mnistData.training->Mnist.getMnistData
  let labels = mnistData.training->Mnist.getMnistLabels

  let n = features->ArraySt.length->Obj.magic

  ArraySt.range(0, epochs - 1)->ArraySt.reduceOneParam((. state, epoch) => {
    let (state, (correctCount, errorCount)) =
      features->ArraySt.reduceOneParami((. (state, (correctCount, errorCount)), feature, i) => {
        let labelVector = labels[i]->Vector.create

        let inputVector = _createInputVector(feature)

        let (_, (_, layer3OutputVector)) as forwardOutput = forward(
          (
            _activate_sigmoid(
              _handleInputValueToAvoidTooLargeForSigmoid(
                Matrix.getColCount(state.wMatrixBetweenLayer1Layer2),
              ),
            ),
            _activate_sigmoid(
              _handleInputValueToAvoidTooLargeForSigmoid(
                Matrix.getColCount(state.wMatrixBetweenLayer2Layer3),
              ),
            ),
          ),
          inputVector,
          state,
        )

        let (layer2Gradient, layer3Gradient) =
          forwardOutput->backward(n, labelVector, inputVector, state)

        // DebugUtils.checkWeightMatrixAndGradientMatrixRadio(
        //   state.wMatrixBetweenLayer1Layer2,
        //   Matrix.multiplyScalar(layer2LearnRate , layer2Gradient),
        // )
        // DebugUtils.checkWeightMatrixAndGradientMatrixRadio(
        //   state.wMatrixBetweenLayer2Layer3,
        //   Matrix.multiplyScalar(layer3LearnRate, layer3Gradient),
        // )

        // DebugUtils.checkGradientExplosionOrDisappear(layer2Gradient)->ignore
        // DebugUtils.checkGradientExplosionOrDisappear(layer3Gradient)->ignore

        DebugUtils.checkGradientExplosionOrDisappear(layer2Gradient->Matrix.multiplyScalar(layer2LearnRate, _))->ignore
        DebugUtils.checkGradientExplosionOrDisappear(layer3Gradient->Matrix.multiplyScalar(layer3LearnRate, _))->ignore

        (
          {
            wMatrixBetweenLayer1Layer2: Matrix.sub(
              state.wMatrixBetweenLayer1Layer2,
              layer2Gradient->Matrix.multiplyScalar(layer2LearnRate, _),
            ),
            wMatrixBetweenLayer2Layer3: Matrix.sub(
              state.wMatrixBetweenLayer2Layer3,
              layer3Gradient->Matrix.multiplyScalar(layer3LearnRate, _),
            ),
          },
          _isCorrectInference(labelVector, layer3OutputVector)
            ? (correctCount->succ, errorCount)
            : (correctCount, errorCount->succ),
        )
      }, (state, (0, 0)))

    true
      ? {
          Js.log(("getCorrectRate:", _getCorrectRate(correctCount, errorCount)))

          state
        }
      : state
  }, state)
}

let state = createState(784, 30, 10)

let state = train(state, 10)
