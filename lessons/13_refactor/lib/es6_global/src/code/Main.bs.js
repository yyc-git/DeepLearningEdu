

import * as Path from "path";
import * as Random$Cnn from "./Random.bs.js";
import * as Network$Cnn from "./Network.bs.js";
import * as DebugLog$Cnn from "./DebugLog.bs.js";
import * as NodeExtend$Cnn from "./NodeExtend.bs.js";
import * as LinearLayer$Cnn from "./LinearLayer.bs.js";
import * as CrossEntropyLoss$Cnn from "./CrossEntropyLoss.bs.js";
import * as NoOptimizerUtils$Cnn from "./optimizer/NoOptimizerUtils.bs.js";
import * as SigmoidActivator$Cnn from "./SigmoidActivator.bs.js";
import * as SoftmaxActivator$Cnn from "./SoftmaxActivator.bs.js";

function _createNetwork1(param) {
  return Network$Cnn.create(NoOptimizerUtils$Cnn.buildNetworkNoOptimizerData([
                  10,
                  0.1
                ], undefined), true, [
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(784, 30, Random$Cnn.random, undefined, undefined), SigmoidActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(30, 10, Random$Cnn.random, undefined, undefined), SoftmaxActivator$Cnn.buildData(undefined))
            ]);
}

var match = _createNetwork1(undefined);

var networkState = Network$Cnn.setLossData(match[0], CrossEntropyLoss$Cnn.buildData(undefined));

var match$1 = Network$Cnn.trainAndInference([
      networkState,
      match[1]
    ], [
      [
        100,
        true
      ],
      100
    ], [
      5,
      1
    ]);

var logState = DebugLog$Cnn.createLogFile(match$1[1], Path.join(NodeExtend$Cnn.getDirname(undefined), "/../../../../src/"));

var networkState$1 = match$1[0];

export {
  _createNetwork1 ,
  networkState$1 as networkState,
  logState ,
  
}
/* match Not a pure module */
