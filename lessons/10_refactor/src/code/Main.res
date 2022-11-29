let _createNetwork1 = () => {
  let linearLayerLearnRate = 0.01
  // let weightDecay = 0.0

  Network.create(
    AdamWOptimizerUtils.buildNetworkAdamWOptimizerData(
      ~learnRate=linearLayerLearnRate,
      // ~weightDecay,
      (),
    ),
    true,
    [
      LinearLayer.createLayerData(
        LinearLayer.create(
          ~inLinearLayerNodeCount=28 * 28,
          ~outLinearLayerNodeCount=30,
          ~initValueMethod=Random.random,
          (),
        )->Obj.magic,
        SigmoidActivator.buildData()->Some,
      ),
      LinearLayer.createLayerData(
        LinearLayer.create(
          ~inLinearLayerNodeCount=30,
          ~outLinearLayerNodeCount=10,
          ~initValueMethod=Random.random,
          (),
        )->Obj.magic,
        SoftmaxActivator.buildData()->Some,
      ),
    ],
  )
}

let (networkState, logState) = _createNetwork1()

let networkState = networkState->Network.setLossData(CrossEntropyLoss.buildData())

let (networkState, logState) = (networkState, logState)->Network.trainAndInference(
  // ((4, false), 4),
  // ((10, false), 6),
  // ((12, true), 6),
  ((10, false), 4),
  // ((4, false), 10),
  // ((200, true), 10),
  // (30, 4),
  // (60, 1),
  // (10, 1),
  (1, 1),
)

let logState = DebugLog.createLogFile(
  logState,
  NodeExtend.joinTwo(NodeExtend.getDirname(), "/../../../../src/"),
)
