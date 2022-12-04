'use strict';

var Path = require("path");
var Random$Cnn = require("./Random.bs.js");
var Network$Cnn = require("./Network.bs.js");
var DebugLog$Cnn = require("./DebugLog.bs.js");
var ConvLayer$Cnn = require("./more_layers/ConvLayer.bs.js");
var FoldLayer$Cnn = require("./more_layers/FoldLayer.bs.js");
var NodeExtend$Cnn = require("./NodeExtend.bs.js");
var LinearLayer$Cnn = require("./LinearLayer.bs.js");
var FlattenLayer$Cnn = require("./more_layers/FlattenLayer.bs.js");
var ReluActivator$Cnn = require("./ReluActivator.bs.js");
var MaxPoolingLayer$Cnn = require("./more_layers/MaxPoolingLayer.bs.js");
var CrossEntropyLoss$Cnn = require("./CrossEntropyLoss.bs.js");
var NoOptimizerUtils$Cnn = require("./optimizer/NoOptimizerUtils.bs.js");
var SigmoidActivator$Cnn = require("./SigmoidActivator.bs.js");
var SoftmaxActivator$Cnn = require("./SoftmaxActivator.bs.js");
var IdentityActivator$Cnn = require("./IdentityActivator.bs.js");

function _createNetwork1(param) {
  return Network$Cnn.create(NoOptimizerUtils$Cnn.buildNetworkNoOptimizerData([
                  10,
                  0.1
                ], undefined), true, [
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(784, 30, Random$Cnn.random, undefined, undefined), SigmoidActivator$Cnn.buildData(undefined)),
              LinearLayer$Cnn.createLayerData(LinearLayer$Cnn.create(30, 10, Random$Cnn.random, undefined, undefined), SoftmaxActivator$Cnn.buildData(undefined))
            ]);
}

function _createNetwork2(param) {
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

var match = _createNetwork2(undefined);

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

exports._createNetwork1 = _createNetwork1;
exports._createNetwork2 = _createNetwork2;
exports.networkState = networkState$1;
exports.logState = logState;
/* match Not a pure module */
