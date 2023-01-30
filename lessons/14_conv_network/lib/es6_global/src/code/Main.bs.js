

import * as Path from "path";
import * as Random$Cnn from "./Random.bs.js";
import * as Network$Cnn from "./Network.bs.js";
import * as DebugLog$Cnn from "./DebugLog.bs.js";
import * as ConvLayer$Cnn from "./more_layers/ConvLayer.bs.js";
import * as FoldLayer$Cnn from "./more_layers/FoldLayer.bs.js";
import * as NodeExtend$Cnn from "./NodeExtend.bs.js";
import * as LinearLayer$Cnn from "./LinearLayer.bs.js";
import * as FlattenLayer$Cnn from "./more_layers/FlattenLayer.bs.js";
import * as ReluActivator$Cnn from "./ReluActivator.bs.js";
import * as MaxPoolingLayer$Cnn from "./more_layers/MaxPoolingLayer.bs.js";
import * as CrossEntropyLoss$Cnn from "./CrossEntropyLoss.bs.js";
import * as NoOptimizerUtils$Cnn from "./optimizer/NoOptimizerUtils.bs.js";
import * as SigmoidActivator$Cnn from "./SigmoidActivator.bs.js";
import * as SoftmaxActivator$Cnn from "./SoftmaxActivator.bs.js";
import * as IdentityActivator$Cnn from "./IdentityActivator.bs.js";

function _createNetwork1(param) {
  return Network$Cnn.create(NoOptimizerUtils$Cnn.buildNetworkNoOptimizerData([
                  10,
                  0.1
                ], undefined), true, [
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(784, 30, Random$Cnn.random, undefined, undefined), SigmoidActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(30, 10, Random$Cnn.random, undefined, undefined), SoftmaxActivator$Cnn.buildData(undefined))
            ]);
}

function _createConvNetwork(param) {
  return Network$Cnn.create(NoOptimizerUtils$Cnn.buildNetworkNoOptimizerData([
                  0,
                  0.001,
                  0,
                  0,
                  0.01,
                  0.01
                ], undefined), true, [
              FoldLayer$Cnn.createLayerData(FoldLayer$Cnn.create(28, 28), IdentityActivator$Cnn.buildData(undefined)),
              ConvLayer$Cnn.createLayerData(ConvLayer$Cnn.create(28, 28, 5, 5, 1, 2, 1, 1, Random$Cnn.random, undefined, undefined), ReluActivator$Cnn.buildData(undefined)),
              MaxPoolingLayer$Cnn.createLayerData(MaxPoolingLayer$Cnn.create(28, 28, 2, 2, 2, 1, undefined), undefined),
              FlattenLayer$Cnn.createLayerData(FlattenLayer$Cnn.create(14, 14, 1, undefined), IdentityActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(196, 30, Random$Cnn.random, undefined, undefined), ReluActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(30, 10, Random$Cnn.random, undefined, undefined), SoftmaxActivator$Cnn.buildData(undefined))
            ]);
}

var match = _createConvNetwork(undefined);

var networkState = Network$Cnn.setLossData(match[0], CrossEntropyLoss$Cnn.buildData(undefined));

var match$1 = Network$Cnn.trainAndInference([
      networkState,
      match[1]
    ], [
      [
        10,
        false
      ],
      4
    ], [
      60,
      1
    ]);

var logState = DebugLog$Cnn.createLogFile(match$1[1], Path.join(NodeExtend$Cnn.getDirname(undefined), "/../../../../src/"));

var networkState$1 = match$1[0];

export {
  _createNetwork1 ,
  _createConvNetwork ,
  networkState$1 as networkState,
  logState ,
  
}
/* match Not a pure module */
