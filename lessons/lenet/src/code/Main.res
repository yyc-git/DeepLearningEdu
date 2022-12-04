let _createConvNetwork = () => {
  let learnRate = 0.001

  let depthNumber = 1

  Network.create(
    AdamWOptimizerUtils.buildNetworkAdamWOptimizerData(~learnRate, ()),
    false,
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
          ~initValueMethod=NormalHe.normal,
          (),
        )->Obj.magic,
        ReluActivator.buildData()->Some,
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
          ~initValueMethod=NormalHe.normal,
          (),
        )->Obj.magic,
        ReluActivator.buildData()->Some,
      ),
      LinearLayer.createLayerData(
        LinearLayer.create(
          ~inLinearLayerNodeCount=30,
          ~outLinearLayerNodeCount=10,
          ~initValueMethod=NormalXavier.normal,
          (),
        )->Obj.magic,
        SoftmaxActivator.buildData()->Some,
      ),
    ],
  )
}

let _createLeNetNetwork = () => {
  let learnRate = 0.001

  let depthNumber1 = 6
  let depthNumber2 = 16
  let flattenSize = 5

  Network.create(
    AdamWOptimizerUtils.buildNetworkAdamWOptimizerData(~learnRate, ()),
    false,
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
          ~filterNumber=depthNumber1,
          ~zeroPadding=2,
          ~stride=1,
          ~initValueMethod=NormalHe.normal,
          (),
        )->Obj.magic,
        ReluActivator.buildData()->Some,
      ),
      MaxPoolingLayer.createLayerData(
        MaxPoolingLayer.create(
          ~inputWidth=28,
          ~inputHeight=28,
          ~depthNumber=depthNumber1,
          ~filterWidth=2,
          ~filterHeight=2,
          ~stride=2,
          (),
        )->Obj.magic,
        // None,
        IdentityActivator.buildData()->Some,
      ),
      ConvLayer.createLayerData(
        ConvLayer.create(
          ~inputWidth=14,
          ~inputHeight=14,
          ~depthNumber=depthNumber1,
          ~filterWidth=5,
          ~filterHeight=5,
          ~filterNumber=depthNumber2,
          ~zeroPadding=0,
          ~stride=1,
          ~initValueMethod=NormalHe.normal,
          (),
        )->Obj.magic,
        ReluActivator.buildData()->Some,
      ),
      MaxPoolingLayer.createLayerData(
        MaxPoolingLayer.create(
          ~inputWidth=10,
          ~inputHeight=10,
          ~depthNumber=depthNumber2,
          ~filterWidth=2,
          ~filterHeight=2,
          ~stride=2,
          (),
        )->Obj.magic,
        // None,
        IdentityActivator.buildData()->Some,
      ),
      FlattenLayer.createLayerData(
        FlattenLayer.create(
          ~inputWidth=flattenSize,
          ~inputHeight=flattenSize,
          ~depthNumber=depthNumber2,
          (),
        )->Obj.magic,
        IdentityActivator.buildData()->Some,
      ),
      // TODO restore to 3 linear layers
      LinearLayer.createLayerData(
        LinearLayer.create(
          ~inLinearLayerNodeCount=depthNumber2 * flattenSize * flattenSize,
          // ~outLinearLayerNodeCount=120,
          ~outLinearLayerNodeCount=30,
          ~initValueMethod=NormalHe.normal,
          (),
        )->Obj.magic,
        ReluActivator.buildData()->Some,
      ),
      // LinearLayer.createLayerData(
      //   LinearLayer.create(
      //     ~inLinearLayerNodeCount=120,
      //     ~outLinearLayerNodeCount=84,
      //     ~initValueMethod=NormalHe.normal,
      //     (),
      //   )->Obj.magic,
      //   ReluActivator.buildData()->Some,
      // ),
      LinearLayer.createLayerData(
        LinearLayer.create(
          // ~inLinearLayerNodeCount=84,
          ~inLinearLayerNodeCount=30,
          ~outLinearLayerNodeCount=10,
          ~initValueMethod=NormalXavier.normal,
          (),
        )->Obj.magic,
        SoftmaxActivator.buildData()->Some,
      ),
    ],
  )
}

// let (networkState, logState) = _createLeNetNetwork()
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
  // (60, 1),
  // (10, 1),
  // (1, 1),
  // (10, 1),
  // (5, 1),
  (20, 1),
)

let logState = DebugLog.createLogFile(
  logState,
  NodeExtend.joinTwo(NodeExtend.getDirname(), "/../../../../src/"),
)
