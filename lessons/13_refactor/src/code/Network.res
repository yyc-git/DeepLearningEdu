type state = {
  optimizerData: Optimizer.optimizerData,
  allLayerData: array<LayerAbstractType.layerData>,
  lossData: option<LossType.data>,
}

let _getAllLayerWeightAndBias = state => {
  state.allLayerData->ArraySt.map(({
    layerName,
    state,
    getWeight,
    getBias,
  }): DebugLog.layerWeightAndBias => {
    {
      layerName: layerName,
      weight: getWeight(state),
      bias: getBias(state),
    }
  })
}

let create = (optimizerData, isLog, allLayerData) => {
  let state = {
    optimizerData: optimizerData,
    allLayerData: allLayerData,
    lossData: None,
  }

  let logState = DebugLog.create(isLog)

  let logState =
    logState->DebugLog.addInfo(
      Log.buildDebugLogMessage(
        ~description="init weight, bias",
        ~var=_getAllLayerWeightAndBias(state),
        (),
      ),
    )

  (state, logState)
}

let setLossData = (state, lossData) => {
  ...state,
  lossData: Some(lossData),
}

let _getForwardOutput = (forwardResultReverse): Vector.t => {
  forwardResultReverse->ArraySt.getFirstExn->Tuple2.getLast->Obj.magic
}

let forward = ((state, logState), input, phase) => {
  let (allLayerData, logData, forwardResultReverse, _) =
    state.allLayerData->ArraySt.reduceOneParam(
      (.
        (allLayerData, logData, arr, input),
        {layerName, state, forward, activatorData} as layerData,
      ) => {
        let (state, net, output) = forward(state, activatorData, input, phase)

        (
          allLayerData->ArraySt.push({
            ...layerData,
            state: state,
          }),
          logData->ArraySt.push(
            (
              {
                layerName: layerName,
                output: output->Obj.magic,
              }: DebugLog.forward
            ),
          ),
          Js.Array.concat(arr, [(net, output)]),
          output->Obj.magic,
        )
      },
      ([], [], [], input->Obj.magic),
    )

  let state = {
    ...state,
    allLayerData: allLayerData,
  }

  let logState =
    logState->DebugLog.addInfo(Log.buildDebugLogMessage(~description="forward", ~var=logData, ()))

  (
    (state, logState),
    forwardResultReverse->_getForwardOutput,
    forwardResultReverse->ArraySt.sliceFrom(1),
  )
}

let _getPreviousLayerNetAndOutput = (
  layerLength,
  layerIndex,
  (inputNet, inputVector),
  forwardResultReverseRemain,
) => {
  layerIndex + 1 > layerLength - 1
    ? (inputNet->Some, inputVector)
    : forwardResultReverseRemain->ArraySt.getExn(layerIndex)
}

let _getPreviousLayerActivatorData = (allLayerDataReverse, layerIndex) => {
  layerIndex + 1 > allLayerDataReverse->ArraySt.length - 1
    ? IdentityActivator.buildData()->Some
    : allLayerDataReverse
      ->ArraySt.map(({activatorData}: LayerAbstractType.layerData) => activatorData)
      ->ArraySt.getExn(layerIndex + 1)
}

let backward = (
  logState,
  labelVector,
  (inputNet, inputVector),
  {allLayerData, lossData} as state,
  (forwardResultOutput, forwardResultReverseRemain),
) => {
  let {computeDelta} = lossData->OptionSt.getExn

  let allLayerDataReverse = state.allLayerData->ArraySt.copy->ArraySt.reverse

  let layerLength = allLayerDataReverse->ArraySt.length

  let (logData, allGradientDataReverse, _) =
    allLayerDataReverse->ArraySt.reduceOneParami(
      (.
        (logData, arr, currentLayerDelta),
        {layerName, state, backward, activatorData}: LayerAbstractType.layerData,
        layerIndex,
      ) => {
        let (net, output) = _getPreviousLayerNetAndOutput(
          layerLength,
          layerIndex,
          (inputNet->Obj.magic, inputVector->Obj.magic),
          forwardResultReverseRemain,
        )

        let (previousLayerDelta, gradientData) = backward(
          _getPreviousLayerActivatorData(allLayerDataReverse, layerIndex),
          (output->Some, net),
          currentLayerDelta->OptionSt.getExn->Obj.magic,
          state,
        )

        (
          logData->ArraySt.push(
            (
              {
                layerName: layerName,
                layerDelta: currentLayerDelta->OptionSt.getExn->Obj.magic,
                gradientData: gradientData->Obj.magic,
              }: DebugLog.backward
            ),
          ),
          arr->ArraySt.push(gradientData),
          previousLayerDelta,
        )
      },
      ([], [], computeDelta(forwardResultOutput, labelVector)->Obj.magic),
    )

  let logState =
    logState->DebugLog.addInfo(Log.buildDebugLogMessage(~description="backward", ~var=logData, ()))

  (logState, allGradientDataReverse->ArraySt.reverse)
}

// let _convertInputVectorToInputs = (inputVector, inputWidth, inputHeight) => {
//   ImmutableSparseMap.createEmpty()->ImmutableSparseMap.set(
//     0,
//     Matrix.create(inputHeight, inputWidth, inputVector->Vector.toArray),
//   )
// }

let _getInputNet = inputVector => {
  inputVector
}

let _update = (state, allGradientDataSum, miniBatchSize) => {
  let optimizerData = state.optimizerData

  let (logData, allLayerData) =
    state.allLayerData->ArraySt.reduceOneParami(
      (.
        (logData, allLayerData),
        {layerName, state, getWeight, getBias, update} as layerData: LayerAbstractType.layerData,
        layerIndex,
      ) => {
        let state = update(
          state,
          (
            optimizerData.optimizerFuncs,
            optimizerData.networkHyperparam,
            Optimizer.getLayerHyperparam(optimizerData, layerIndex),
          ),
          (miniBatchSize, allGradientDataSum->ArraySt.getExn(layerIndex)),
        )

        (
          logData->ArraySt.push(
            (
              {
                layerName: layerName,
                miniBatchSize: miniBatchSize,
                weight: getWeight(state),
                bias: getBias(state),
              }: DebugLog.update
            ),
          ),
          allLayerData->ArraySt.push({
            ...layerData,
            state: state,
          }),
        )
      },
      ([], []),
    )

  (
    {
      ...state,
      optimizerData: Optimizer.updateNetworkHyperParam(optimizerData),
    },
    logData,
    allLayerData,
  )
}

let trainMiniBatchData = (
  ((state, logState), (correctCount, errorCount), lossSum),
  miniBatchData,
  miniBatchSize,
) => {
  let _isCorrectInference = (labelVector, inferenceVector) => {
    LinearLayer.getOutputNumber(labelVector) == LinearLayer.getOutputNumber(inferenceVector)
  }

  let (
    logState,
    allGradientDataSum,
    (correctCount, errorCount),
    lossSum,
  ) = miniBatchData->ArraySt.reduceOneParam(
    (.
      (logState, allGradientDataSum, (correctCount, errorCount), lossSum),
      (inputVector, labelVector),
    ) => {
      let logState = logState->DebugLog.addInfo(
        Log.buildDebugLogMessage(
          ~description="train one data",
          ~var=(
            {
              label: labelVector,
              input: inputVector,
            }: DebugLog.oneData
          ),
          (),
        ),
      )

      let ((state, logState), forwardResultOutput, forwardResultReverseRemain) = forward(
        (state, logState),
        inputVector,
        NetworkType.Train,
      )

      let (logState, allGradientData) = backward(
        logState,
        labelVector,
        (inputVector->_getInputNet, inputVector),
        state,
        (forwardResultOutput, forwardResultReverseRemain->Obj.magic),
      )

      (
        logState,
        allGradientDataSum->ArraySt.mapi((gradientDataSum, layerIndex) => {
          let {addToGradientDataSum}: LayerAbstractType.layerData =
            state.allLayerData->ArraySt.getExn(layerIndex)

          addToGradientDataSum(gradientDataSum, allGradientData->ArraySt.getExn(layerIndex))
        }),
        _isCorrectInference(labelVector, forwardResultOutput)
          ? (correctCount->succ, errorCount)
          : (correctCount, errorCount->succ),
        lossSum +. CrossEntropyLoss.compute(~output=forwardResultOutput, ~label=labelVector, ()),
      )
    },
    (
      logState,
      state.allLayerData->ArraySt.reduceOneParam((. arr, {state, createGradientDataSum}) => {
        arr->ArraySt.push(createGradientDataSum(state))
      }, []),
      (correctCount, errorCount),
      lossSum,
    ),
  )

  let (state, logData, allLayerData) = _update(state, allGradientDataSum, miniBatchSize)

  let logState =
    logState->DebugLog.addInfo(
      Log.buildDebugLogMessage(~description="weight, bias after update", ~var=logData, ()),
    )

  (
    (
      {
        ...state,
        allLayerData: allLayerData,
      },
      logState,
    ),
    (correctCount, errorCount),
    lossSum,
  )
}

module MiniBatch = {
  let partition = (data, labels, miniBatchSize) => {
    data->ArraySt.length < miniBatchSize ? Exception.throwErr("error") : ()

    let (miniBatchPartitionData, miniBatchData) =
      data->ArraySt.reduceOneParami(
        (. (miniBatchPartitionData, miniBatchData), inputVector, index) => {
          let labelVector = labels[index]

          let miniBatchData = miniBatchData->ArraySt.push((inputVector, labelVector))

          mod(index + 1, miniBatchSize) == 0
            ? {
                (miniBatchPartitionData->ArraySt.push(miniBatchData), [])
              }
            : {
                (miniBatchPartitionData, miniBatchData)
              }
        },
        ([], []),
      )

    // throw miniBatchData even if it has data

    // (
    //   miniBatchPartitionData,
    // Js.Math.floor(data->ArraySt.length->Obj.magic /. miniBatchSize->Obj.magic),
    // )
    miniBatchPartitionData
  }

  let shuffle = sampleCount => {
    let mnistData = Mnist.set(sampleCount, 1)

    // let data = mnistData.training->Mnist.getMnistData->ArraySt.sliceFrom(-4)

    // let labels = mnistData.training->Mnist.getMnistLabels->ArraySt.sliceFrom(-4)
    let data = mnistData.training->Mnist.getMnistData->ArraySt.sliceFrom(-sampleCount)

    let labels = mnistData.training->Mnist.getMnistLabels->ArraySt.sliceFrom(-sampleCount)

    (data, labels)
  }
}

let _train = (
  (state, logState),
  handleEachEpochFunc,
  (sampleCount, isShuffle),
  (epochs, miniBatchSize),
) => {
  let _getCorrectRate = (correctCount, errorCount) => {
    (correctCount->Obj.magic /. (correctCount->Obj.magic +. errorCount->Obj.magic) *. 100.)
      ->Obj.magic ++ "%"
  }

  ("begin: ", epochs, miniBatchSize)->Log.printForDebug->ignore

  let sampleData = isShuffle
    ? {
        None
      }
    : {
        let (data, labels) = MiniBatch.shuffle(sampleCount)

        let miniBatchPartitionData = MiniBatch.partition(data, labels, miniBatchSize)

        (data, labels, miniBatchPartitionData)->Some
      }

  ArraySt.range(0, epochs - 1)->ArraySt.reduceOneParam((. (state, logState), epochIndex) => {
    let logState = logState->DebugLog.addInfo({j`epoch index: ${epochIndex->Obj.magic}`})

    let (data, labels, miniBatchPartitionData) = isShuffle
      ? {
          let (data, labels) = MiniBatch.shuffle(sampleCount)

          let miniBatchPartitionData = MiniBatch.partition(data, labels, miniBatchSize)

          (data, labels, miniBatchPartitionData)
        }
      : {
          sampleData->OptionSt.getExn
        }

    let ((state, logState), (correctCount, errorCount), lossSum) =
      miniBatchPartitionData->ArraySt.reduceOneParam(
        (. ((state, logState), (correctCount, errorCount), lossSum), miniBatchData) => {
          trainMiniBatchData(
            ((state, logState), (correctCount, errorCount), lossSum),
            miniBatchData,
            miniBatchSize,
          )
        },
        ((state, logState), (0, 0), 0.),
      )

    let (state, logState) =
      (state, logState)->handleEachEpochFunc(
        epochIndex,
        lossSum /. sampleCount->Obj.magic,
        _getCorrectRate(correctCount, errorCount),
      )

    (state, logState)
  }, (state, logState))
}

let train = ((state, logState), (sampleCount, isShuffle), (epochs, miniBatchSize)) => {
  _train(
    (state, logState),
    ((state, logState), _, loss, correctRate) => {
      Js.log(("loss:", loss))
      Js.log(("getCorrectRate:", correctRate))

      (state, logState)
    },
    (sampleCount, isShuffle),
    (epochs, miniBatchSize),
  )
}

let inference = ((state, logState), (data, labels)) => {
  let ((state, logState), lossSum) =
    data->ArraySt.reduceOneParami((. ((state, logState), lossSum), input, i) => {
      let ((state, logState), forwardResultOutput, _) = forward(
        (state, logState),
        input,
        NetworkType.Inference,
      )

      (
        (state, logState),
        lossSum +.
        CrossEntropyLoss.compute(~output=forwardResultOutput, ~label=labels->ArraySt.getExn(i), ()),
      )
    }, ((state, logState), 0.))

  let sampleCount = data->ArraySt.length

  ((state, logState), lossSum /. sampleCount->Obj.magic)
}

let trainAndInference = (
  (state, logState),
  ((trainSampleCount, isShuffle), inferenceSampleCount),
  (epochs, miniBatchSize),
) => {
  _train(
    (state, logState),
    ((state, logState), _, trainLoss, trainCorrectRate) => {
      let ((state, logState), inferenceLoss) = inference(
        (state, logState),
        MiniBatch.shuffle(inferenceSampleCount),
      )

      Js.log({
        j`trainLoss: ${trainLoss->Obj.magic}, trainCorrectRate: ${trainCorrectRate->Obj.magic}, inferenceLoss: ${inferenceLoss->Obj.magic} `
      })

      (state, logState)
    },
    (trainSampleCount, isShuffle),
    (epochs, miniBatchSize),
  )
}
