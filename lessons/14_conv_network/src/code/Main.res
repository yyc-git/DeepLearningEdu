let _createNetwork1 = () => {
  // let linearLayerLearnRate = 0.01
  // let linearLayerLearnRate = 10.
  // let weightDecay = 0.0

  Network.create(
    NoOptimizerUtils.buildNetworkNoOptimizerData(~allLearnRates=[10., 0.1], ()),
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
        // ReluActivator.buildData()->Some,
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

let _createConvNetwork = () => {
  // let learnRate = 0.001
  // let weightDecay = 0.0
  // let weightDecay = 0.0001

  // let depthNumber = 6
  let depthNumber = 1

  Network.create(
    // AdamWOptimizerUtils.buildNetworkAdamWOptimizerData(~learnRate, ~weightDecay, ()),
    // AdamWOptimizerUtils.buildNetworkAdamWOptimizerData(~learnRate, ()),

    // NoOptimizerUtils.buildNetworkNoOptimizerData(~allLearnRates=[10., 0.1], ()),
    // NoOptimizerUtils.buildNetworkNoOptimizerData(~allLearnRates=[0., 0.01, 0., 0., 10., 0.1], ()),
    NoOptimizerUtils.buildNetworkNoOptimizerData(
      ~allLearnRates=[0., 0.001, 0., 0., 0.01, 0.01],
      (),
    ),
    true,
    [
      FoldLayer.createLayerData(
        FoldLayer.create(28, 28)->Obj.magic,
        IdentityActivator.buildData()->Some,
      ),
      ConvLayer.createLayerData(
        ConvLayer.create(
          ~inputWidth=28,
          ~inputHeight=28,
          ~depthNumber=1,
          ~filterWidth=5,
          ~filterHeight=5,
          ~filterNumber=depthNumber,
          ~zeroPadding=2,
          ~stride=1,
          // ~initValueMethod=NormalHe.normal,
          ~initValueMethod=Random.random,
          (),
        )->Obj.magic,
        ReluActivator.buildData()->Some,
        // IdentityActivator.buildData()->Some,
      ),
      MaxPoolingLayer.createLayerData(
        MaxPoolingLayer.create(
          ~inputWidth=28,
          ~inputHeight=28,
          ~depthNumber,
          ~filterWidth=2,
          ~filterHeight=2,
          ~stride=2,
          (),
        )->Obj.magic,
        None,
      ),
      FlattenLayer.createLayerData(
        FlattenLayer.create(~inputWidth=14, ~inputHeight=14, ~depthNumber, ())->Obj.magic,
        IdentityActivator.buildData()->Some,
      ),
      LinearLayer.createLayerData(
        LinearLayer.create(
          ~inLinearLayerNodeCount=depthNumber * 14 * 14,
          ~outLinearLayerNodeCount=30,
          ~initValueMethod=Random.random,
          // ~initValueMethod=NormalHe.normal,
          (),
        )->Obj.magic,
        ReluActivator.buildData()->Some,
      ),
      LinearLayer.createLayerData(
        LinearLayer.create(
          ~inLinearLayerNodeCount=30,
          ~outLinearLayerNodeCount=10,
          ~initValueMethod=Random.random,
          // ~initValueMethod=NormalXavier.normal,
          (),
        )->Obj.magic,
        SoftmaxActivator.buildData()->Some,
      ),
    ],
  )
}

// let (networkState, logState) = _createNetwork1()
let (networkState, logState) = _createConvNetwork()

let networkState = networkState->Network.setLossData(CrossEntropyLoss.buildData())

let (networkState, logState) = (networkState, logState)->Network.trainAndInference(
  // ((4, false), 4),
  // ((10, false), 6),
  // ((12, true), 6),
  ((10, false), 4),
  // ((100, true), 100),
  // ((4, false), 10),
  // ((200, true), 10),
  // (30, 4),
  (60, 1),
  // (10, 1),
  // (1, 1),
  // (10, 1),
  // (5, 1),
)

let logState = DebugLog.createLogFile(
  logState,
  NodeExtend.joinTwo(NodeExtend.getDirname(), "/../../../../src/"),
)
