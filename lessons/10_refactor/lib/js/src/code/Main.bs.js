'use strict';

var Path = require("path");
var Random$Cnn = require("./Random.bs.js");
var Network$Cnn = require("./Network.bs.js");
var DebugLog$Cnn = require("./DebugLog.bs.js");
var NodeExtend$Cnn = require("./NodeExtend.bs.js");
var LinearLayer$Cnn = require("./LinearLayer.bs.js");
var CrossEntropyLoss$Cnn = require("./CrossEntropyLoss.bs.js");
var SigmoidActivator$Cnn = require("./SigmoidActivator.bs.js");
var SoftmaxActivator$Cnn = require("./SoftmaxActivator.bs.js");
var AdamWOptimizerUtils$Cnn = require("./optimizer/AdamWOptimizerUtils.bs.js");

function _createNetwork1(param) {
  return Network$Cnn.create(AdamWOptimizerUtils$Cnn.buildNetworkAdamWOptimizerData(0.01, undefined, undefined, undefined, undefined, undefined), true, [
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

exports._createNetwork1 = _createNetwork1;
exports.networkState = networkState$1;
exports.logState = logState;
/* match Not a pure module */
