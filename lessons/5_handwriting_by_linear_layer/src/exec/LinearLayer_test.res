type state = {
  wMatrixBetweenLayer1Layer2: Matrix.t,
  wMatrixBetweenLayer2Layer3: Matrix.t,
}

// type feature = {
//   weight: float,
//   height: float,
// }

type feature = array<float>

// type label =
//   | Male
//   | Female

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

let _handleInputValueToAvoidTooLargeForSigmoid = (inputValue, max) => {
  inputValue /. (max->Obj.magic /. 10.)
}

let _activate_sigmoid = (max, x) => {
  let x = x->_handleInputValueToAvoidTooLargeForSigmoid(max)

  DebugUtils.checkSigmoidInputTooLarge(x)

  1. /. (1. +. Js.Math.exp(-.x))
}

let _deriv_sigmoid = (max, x) => {
  let fx = _activate_sigmoid(max, x)

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
    _deriv_sigmoid(Matrix.getColCount(state.wMatrixBetweenLayer2Layer3)),
    layer3Net,
    layer3OutputVector,
    n,
    labelVector,
  )

  let layer2Delta = _bpLayer2Delta(
    _deriv_sigmoid(Matrix.getColCount(state.wMatrixBetweenLayer1Layer2)),
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

// let _convertLabelToFloat = label =>
//   switch label {
//   | Male => 0.
//   | Female => 1.
//   }

let _computeLoss = (labels, outputs) => {
  labels->ArraySt.reduceOneParami((. result, label, i) => {
    result +. Js.Math.pow_float(~base=label -. outputs[i], ~exp=2.0)
  }, 0.) /. ArraySt.length(labels)->Obj.magic
}

// let _createInputVector = (feature: feature) => {
//   Vector.create([feature.height, feature.weight, 1.0])
// }

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

let train = (state: state, sampleCount: int): state => {
  //   let learnRate = 0.1
  // let learnRate = 3.
  let learnRate = 10.
  //   let epochs = 1000
  //   let epochs = 10
  // let epochs = 1
  let epochs = 100
  // let epochs = 80

  let mnistData = Mnist.set(sampleCount, 1)

  //   let inputs =
  //     mnistData.training->Mnist.getMnistData->ArraySt.map(d => d->Vector.create->_createInputVector)
  //   let labels = mnistData.training->Mnist.getMnistLabels->ArraySt.map(Vector.create)

  let features = mnistData.training->Mnist.getMnistData
  let labels = mnistData.training->Mnist.getMnistLabels

  //   let features = mnistData.training->Mnist.getMnistData->ArraySt.sliceFrom(-4)

  //   let labels = mnistData.training->Mnist.getMnistLabels->ArraySt.sliceFrom(-4)

  //   Js.log((data, labels))

  let n = features->ArraySt.length->Obj.magic

  ArraySt.range(0, epochs - 1)->ArraySt.reduceOneParam((. state, epoch) => {
    // let features = mnistData.training->Mnist.getMnistData
    // let labels = mnistData.training->Mnist.getMnistLabels

    // let n = features->ArraySt.length->Obj.magic

    let (state, (correctCount, errorCount)) =
      features->ArraySt.reduceOneParami((. (state, (correctCount, errorCount)), feature, i) => {
        let labelVector = labels[i]->Vector.create

        let inputVector = _createInputVector(feature)

        let (_, (_, layer3OutputVector)) as forwardOutput = forward(
          (
            _activate_sigmoid(Matrix.getColCount(state.wMatrixBetweenLayer1Layer2)),
            _activate_sigmoid(Matrix.getColCount(state.wMatrixBetweenLayer2Layer3)),
          ),
          inputVector,
          state,
        )

        let (layer2Gradient, layer3Gradient) =
          forwardOutput->backward(n, labelVector, inputVector, state)

        // Js.log(layer3Gradient)

        // DebugUtils.checkWeightMatrixAndGradientMatrixRadio(
        //   state.wMatrixBetweenLayer1Layer2,
        //   Matrix.multiplyScalar(learnRate, layer2Gradient),
        // )
        //   DebugUtils.checkWeightMatrixAndGradientMatrixRadio(
        // state.wMatrixBetweenLayer2Layer3,
        //   Matrix.multiplyScalar(learnRate, layer3Gradient),
        //   )

        (
          {
            wMatrixBetweenLayer1Layer2: Matrix.sub(
              state.wMatrixBetweenLayer1Layer2,
              layer2Gradient->Matrix.multiplyScalar(learnRate, _),
            ),
            wMatrixBetweenLayer2Layer3: Matrix.sub(
              state.wMatrixBetweenLayer2Layer3,
              layer3Gradient->Matrix.multiplyScalar(learnRate, _),
            ),
          },
          _isCorrectInference(labelVector, layer3OutputVector)
            ? (correctCount->succ, errorCount)
            : (correctCount, errorCount->succ),
        )
      }, (state, (0, 0)))

    // mod(epoch, 10) == 0
    true
      ? {
          Js.log(("getCorrectRate:", _getCorrectRate(correctCount, errorCount)))

          state
        }
      : state
  }, state)
}

let inference = (state: state, feature: feature) => {
  let inputVector = _createInputVector(feature)

  let (_, (_, layer3OutputVector)) = forward(
    (
      _activate_sigmoid(Matrix.getColCount(state.wMatrixBetweenLayer1Layer2)),
      _activate_sigmoid(Matrix.getColCount(state.wMatrixBetweenLayer2Layer3)),
    ),
    inputVector,
    state,
  )

  // layer3OutputVector->Log.printForDebug->_getOutputNumber
  layer3OutputVector
}

let inferenceWithSampleCount = (state: state, sampleCount: int) => {
  let mnistData = Mnist.set(0, sampleCount)

  // let testData = mnistData.test->Mnist.getMnistData->ArraySt.sliceFrom(-8)

  // let testLabels = mnistData.test->Mnist.getMnistLabels->ArraySt.sliceFrom(-8)

  let testData = mnistData.test->Mnist.getMnistData

  let testLabels = mnistData.test->Mnist.getMnistLabels

  let (correctCount, errorCount) =
    testData->ArraySt.reduceOneParami((. (correctCount, errorCount), data, i) => {
      _isCorrectInference(testLabels[i]->Vector.create, inference(state, data))
        ? (correctCount->succ, errorCount)
        : (correctCount, errorCount->succ)
    }, (0, 0))->Log.printForDebug

  _getCorrectRate(correctCount, errorCount)
}

let state = createState(784, 30, 10)

// let state = train(state, 200)
let state = train(state, 10)
// let state = train(state, 100)

Js.log(("inference correctRate:", inferenceWithSampleCount(state, 100)))
