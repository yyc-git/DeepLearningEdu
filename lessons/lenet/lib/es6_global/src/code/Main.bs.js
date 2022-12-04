

import * as Path from "path";
import * as Network$Cnn from "./Network.bs.js";
import * as DebugLog$Cnn from "./DebugLog.bs.js";
import * as NormalHe$Cnn from "./more/NormalHe.bs.js";
import * as ConvLayer$Cnn from "./more_layers/ConvLayer.bs.js";
import * as FoldLayer$Cnn from "./more_layers/FoldLayer.bs.js";
import * as NodeExtend$Cnn from "./NodeExtend.bs.js";
import * as LinearLayer$Cnn from "./LinearLayer.bs.js";
import * as FlattenLayer$Cnn from "./more_layers/FlattenLayer.bs.js";
import * as NormalXavier$Cnn from "./more/NormalXavier.bs.js";
import * as ReluActivator$Cnn from "./ReluActivator.bs.js";
import * as MaxPoolingLayer$Cnn from "./more_layers/MaxPoolingLayer.bs.js";
import * as CrossEntropyLoss$Cnn from "./CrossEntropyLoss.bs.js";
import * as SoftmaxActivator$Cnn from "./SoftmaxActivator.bs.js";
import * as IdentityActivator$Cnn from "./IdentityActivator.bs.js";
import * as AdamWOptimizerUtils$Cnn from "./optimizer/AdamWOptimizerUtils.bs.js";

function _createConvNetwork(param) {
  return Network$Cnn.create(AdamWOptimizerUtils$Cnn.buildNetworkAdamWOptimizerData(0.001, undefined, undefined, undefined, undefined, undefined), false, [
              FoldLayer$Cnn.createLayerData(FoldLayer$Cnn.create(28, 28), IdentityActivator$Cnn.buildData(undefined)),
              ConvLayer$Cnn.createLayerData(ConvLayer$Cnn.create(28, 28, 5, 5, 1, 2, 1, 1, NormalHe$Cnn.normal, undefined, undefined), ReluActivator$Cnn.buildData(undefined)),
              MaxPoolingLayer$Cnn.createLayerData(MaxPoolingLayer$Cnn.create(28, 28, 2, 2, 2, 1, undefined), undefined),
              FlattenLayer$Cnn.createLayerData(FlattenLayer$Cnn.create(14, 14, 1, undefined), IdentityActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(196, 30, NormalHe$Cnn.normal, undefined, undefined), ReluActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(30, 10, NormalXavier$Cnn.normal, undefined, undefined), SoftmaxActivator$Cnn.buildData(undefined))
            ]);
}

function _createLeNetNetwork(param) {
  return Network$Cnn.create(AdamWOptimizerUtils$Cnn.buildNetworkAdamWOptimizerData(0.001, undefined, undefined, undefined, undefined, undefined), false, [
              FoldLayer$Cnn.createLayerData(FoldLayer$Cnn.create(28, 28), IdentityActivator$Cnn.buildData(undefined)),
              ConvLayer$Cnn.createLayerData(ConvLayer$Cnn.create(28, 28, 5, 5, 6, 2, 1, 1, NormalHe$Cnn.normal, undefined, undefined), ReluActivator$Cnn.buildData(undefined)),
              MaxPoolingLayer$Cnn.createLayerData(MaxPoolingLayer$Cnn.create(28, 28, 2, 2, 2, 6, undefined), IdentityActivator$Cnn.buildData(undefined)),
              ConvLayer$Cnn.createLayerData(ConvLayer$Cnn.create(14, 14, 5, 5, 16, 0, 1, 6, NormalHe$Cnn.normal, undefined, undefined), ReluActivator$Cnn.buildData(undefined)),
              MaxPoolingLayer$Cnn.createLayerData(MaxPoolingLayer$Cnn.create(10, 10, 2, 2, 2, 16, undefined), IdentityActivator$Cnn.buildData(undefined)),
              FlattenLayer$Cnn.createLayerData(FlattenLayer$Cnn.create(5, 5, 16, undefined), IdentityActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(400, 30, NormalHe$Cnn.normal, undefined, undefined), ReluActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(30, 10, NormalXavier$Cnn.normal, undefined, undefined), SoftmaxActivator$Cnn.buildData(undefined))
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
      20,
      1
    ]);

var logState = DebugLog$Cnn.createLogFile(match$1[1], Path.join(NodeExtend$Cnn.getDirname(undefined), "/../../../../src/"));

var networkState$1 = match$1[0];

export {
  _createConvNetwork ,
  _createLeNetNetwork ,
  networkState$1 as networkState,
  logState ,
  
}
/* match Not a pure module */
