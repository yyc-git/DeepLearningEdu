'use strict';

var Path = require("path");
var Network$Cnn = require("./Network.bs.js");
var DebugLog$Cnn = require("./DebugLog.bs.js");
var NormalHe$Cnn = require("./more/NormalHe.bs.js");
var ConvLayer$Cnn = require("./more_layers/ConvLayer.bs.js");
var FoldLayer$Cnn = require("./more_layers/FoldLayer.bs.js");
var NodeExtend$Cnn = require("./NodeExtend.bs.js");
var LinearLayer$Cnn = require("./LinearLayer.bs.js");
var FlattenLayer$Cnn = require("./more_layers/FlattenLayer.bs.js");
var NormalXavier$Cnn = require("./more/NormalXavier.bs.js");
var ReluActivator$Cnn = require("./ReluActivator.bs.js");
var MaxPoolingLayer$Cnn = require("./more_layers/MaxPoolingLayer.bs.js");
var CrossEntropyLoss$Cnn = require("./CrossEntropyLoss.bs.js");
var SoftmaxActivator$Cnn = require("./SoftmaxActivator.bs.js");
var IdentityActivator$Cnn = require("./IdentityActivator.bs.js");
var AdamWOptimizerUtils$Cnn = require("./optimizer/AdamWOptimizerUtils.bs.js");

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

exports._createConvNetwork = _createConvNetwork;
exports._createLeNetNetwork = _createLeNetNetwork;
exports.networkState = networkState$1;
exports.logState = logState;
/* match Not a pure module */
