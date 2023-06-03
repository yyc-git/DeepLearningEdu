'use strict';

var Curry = require("rescript/lib/js/curry.js");
var Mnist = require("mnist");
var Log$Cnn = require("./Log.bs.js");
var Caml_obj = require("rescript/lib/js/caml_obj.js");
var Mnist$Cnn = require("./external/mnist.bs.js");
var Caml_array = require("rescript/lib/js/caml_array.js");
var Caml_int32 = require("rescript/lib/js/caml_int32.js");
var Tuple2$Cnn = require("./tuple/Tuple2.bs.js");
var ArraySt$Cnn = require("./ArraySt.bs.js");
var Caml_option = require("rescript/lib/js/caml_option.js");
var DebugLog$Cnn = require("./DebugLog.bs.js");
var OptionSt$Cnn = require("./OptionSt.bs.js");
var Exception$Cnn = require("./Exception.bs.js");
var Optimizer$Cnn = require("./optimizer/Optimizer.bs.js");
var LinearLayer$Cnn = require("./LinearLayer.bs.js");
var CrossEntropyLoss$Cnn = require("./CrossEntropyLoss.bs.js");
var IdentityActivator$Cnn = require("./IdentityActivator.bs.js");

function _getAllLayerWeightAndBias(state) {
  return ArraySt$Cnn.map(state.allLayerData, (function (param) {
                var state = param.state;
                return {
                        layerName: param.layerName,
                        weight: Curry._1(param.getWeight, state),
                        bias: Curry._1(param.getBias, state)
                      };
              }));
}

function create(optimizerData, isLog, allLayerData) {
  var state = {
    optimizerData: optimizerData,
    allLayerData: allLayerData,
    lossData: undefined
  };
  var logState = DebugLog$Cnn.create(isLog);
  var logState$1 = DebugLog$Cnn.addInfo(logState, Log$Cnn.buildDebugLogMessage("init weight, bias", _getAllLayerWeightAndBias(state), undefined));
  return [
          state,
          logState$1
        ];
}

function setLossData(state, lossData) {
  return {
          optimizerData: state.optimizerData,
          allLayerData: state.allLayerData,
          lossData: lossData
        };
}

function _getForwardOutput(forwardResultReverse) {
  return Tuple2$Cnn.getLast(ArraySt$Cnn.getFirstExn(forwardResultReverse));
}

function forward(param, input, phase) {
  var state = param[0];
  var match = ArraySt$Cnn.reduceOneParam(state.allLayerData, (function (param, layerData) {
          var match = Curry._4(layerData.forward, layerData.state, layerData.activatorData, param[3], phase);
          var output = match[2];
          return [
                  ArraySt$Cnn.push(param[0], {
                        layerName: layerData.layerName,
                        state: match[0],
                        forward: layerData.forward,
                        backward: layerData.backward,
                        update: layerData.update,
                        createGradientDataSum: layerData.createGradientDataSum,
                        addToGradientDataSum: layerData.addToGradientDataSum,
                        activatorData: layerData.activatorData,
                        getWeight: layerData.getWeight,
                        getBias: layerData.getBias
                      }),
                  ArraySt$Cnn.push(param[1], {
                        layerName: layerData.layerName,
                        output: output
                      }),
                  [[
                        match[1],
                        output
                      ]].concat(param[2]),
                  output
                ];
        }), [
        [],
        [],
        [],
        input
      ]);
  var forwardResultReverse = match[2];
  var state_optimizerData = state.optimizerData;
  var state_allLayerData = match[0];
  var state_lossData = state.lossData;
  var state$1 = {
    optimizerData: state_optimizerData,
    allLayerData: state_allLayerData,
    lossData: state_lossData
  };
  var logState = DebugLog$Cnn.addInfo(param[1], Log$Cnn.buildDebugLogMessage("forward", match[1], undefined));
  return [
          [
            state$1,
            logState
          ],
          Tuple2$Cnn.getLast(ArraySt$Cnn.getFirstExn(forwardResultReverse)),
          ArraySt$Cnn.sliceFrom(forwardResultReverse, 1)
        ];
}

function _getPreviousLayerNetAndOutput(layerLength, layerIndex, param, forwardResultReverseRemain) {
  if ((layerIndex + 1 | 0) > (layerLength - 1 | 0)) {
    return [
            Caml_option.some(param[0]),
            param[1]
          ];
  } else {
    return ArraySt$Cnn.getExn(forwardResultReverseRemain, layerIndex);
  }
}

function _getPreviousLayerActivatorData(allLayerDataReverse, layerIndex) {
  if ((layerIndex + 1 | 0) > (ArraySt$Cnn.length(allLayerDataReverse) - 1 | 0)) {
    return IdentityActivator$Cnn.buildData(undefined);
  } else {
    return ArraySt$Cnn.getExn(ArraySt$Cnn.map(allLayerDataReverse, (function (param) {
                      return param.activatorData;
                    })), layerIndex + 1 | 0);
  }
}

function backward(logState, labelVector, param, state, param$1) {
  var forwardResultReverseRemain = param$1[1];
  var inputVector = param[1];
  var inputNet = param[0];
  var match = OptionSt$Cnn.getExn(state.lossData);
  var allLayerDataReverse = ArraySt$Cnn.reverse(ArraySt$Cnn.copy(state.allLayerData));
  var layerLength = ArraySt$Cnn.length(allLayerDataReverse);
  var match$1 = ArraySt$Cnn.reduceOneParami(allLayerDataReverse, (function (param, param$1, layerIndex) {
          var nextLayerDelta = param[2];
          var match = _getPreviousLayerNetAndOutput(layerLength, layerIndex, [
                inputNet,
                inputVector
              ], forwardResultReverseRemain);
          var match$1 = Curry._4(param$1.backward, _getPreviousLayerActivatorData(allLayerDataReverse, layerIndex), [
                Caml_option.some(match[1]),
                match[0]
              ], OptionSt$Cnn.getExn(nextLayerDelta), param$1.state);
          var gradientData = match$1[1];
          return [
                  ArraySt$Cnn.push(param[0], {
                        layerName: param$1.layerName,
                        nextLayerDelta: OptionSt$Cnn.getExn(nextLayerDelta),
                        gradientData: gradientData
                      }),
                  ArraySt$Cnn.push(param[1], gradientData),
                  match$1[0]
                ];
        }), [
        [],
        [],
        Curry._2(match.computeDelta, param$1[0], labelVector)
      ]);
  var logState$1 = DebugLog$Cnn.addInfo(logState, Log$Cnn.buildDebugLogMessage("backward", match$1[0], undefined));
  return [
          logState$1,
          ArraySt$Cnn.reverse(match$1[1])
        ];
}

function _getInputNet(inputVector) {
  return inputVector;
}

function _update(state, allGradientDataSum, miniBatchSize) {
  var optimizerData = state.optimizerData;
  var match = ArraySt$Cnn.reduceOneParami(state.allLayerData, (function (param, layerData, layerIndex) {
          var state = Curry._3(layerData.update, layerData.state, [
                optimizerData.optimizerFuncs,
                optimizerData.networkHyperparam,
                Optimizer$Cnn.getLayerHyperparam(optimizerData, layerIndex)
              ], [
                miniBatchSize,
                ArraySt$Cnn.getExn(allGradientDataSum, layerIndex)
              ]);
          return [
                  ArraySt$Cnn.push(param[0], {
                        layerName: layerData.layerName,
                        miniBatchSize: miniBatchSize,
                        weight: Curry._1(layerData.getWeight, state),
                        bias: Curry._1(layerData.getBias, state)
                      }),
                  ArraySt$Cnn.push(param[1], {
                        layerName: layerData.layerName,
                        state: state,
                        forward: layerData.forward,
                        backward: layerData.backward,
                        update: layerData.update,
                        createGradientDataSum: layerData.createGradientDataSum,
                        addToGradientDataSum: layerData.addToGradientDataSum,
                        activatorData: layerData.activatorData,
                        getWeight: layerData.getWeight,
                        getBias: layerData.getBias
                      })
                ];
        }), [
        [],
        []
      ]);
  return [
          {
            optimizerData: Optimizer$Cnn.updateNetworkHyperParam(optimizerData),
            allLayerData: state.allLayerData,
            lossData: state.lossData
          },
          match[0],
          match[1]
        ];
}

function trainMiniBatchData(param, miniBatchData, miniBatchSize) {
  var match = param[1];
  var match$1 = param[0];
  var state = match$1[0];
  var _isCorrectInference = function (labelVector, inferenceVector) {
    return Caml_obj.caml_equal(LinearLayer$Cnn.getOutputNumber(labelVector), LinearLayer$Cnn.getOutputNumber(inferenceVector));
  };
  var match$2 = ArraySt$Cnn.reduceOneParam(miniBatchData, (function (param, param$1) {
          var labelVector = param$1[1];
          var inputVector = param$1[0];
          var match = param[2];
          var errorCount = match[1];
          var correctCount = match[0];
          var logState = DebugLog$Cnn.addInfo(param[0], Log$Cnn.buildDebugLogMessage("train one data", {
                    label: labelVector,
                    input: inputVector
                  }, undefined));
          var match$1 = forward([
                state,
                logState
              ], inputVector, /* Train */0);
          var forwardResultOutput = match$1[1];
          var match$2 = match$1[0];
          var state$1 = match$2[0];
          var match$3 = backward(match$2[1], labelVector, [
                inputVector,
                inputVector
              ], state$1, [
                forwardResultOutput,
                match$1[2]
              ]);
          var allGradientData = match$3[1];
          return [
                  match$3[0],
                  ArraySt$Cnn.mapi(param[1], (function (gradientDataSum, layerIndex) {
                          var match = ArraySt$Cnn.getExn(state$1.allLayerData, layerIndex);
                          return Curry._2(match.addToGradientDataSum, gradientDataSum, ArraySt$Cnn.getExn(allGradientData, layerIndex));
                        })),
                  _isCorrectInference(labelVector, forwardResultOutput) ? [
                      correctCount + 1 | 0,
                      errorCount
                    ] : [
                      correctCount,
                      errorCount + 1 | 0
                    ],
                  param[3] + CrossEntropyLoss$Cnn.compute(forwardResultOutput, labelVector, undefined, undefined)
                ];
        }), [
        match$1[1],
        ArraySt$Cnn.reduceOneParam(state.allLayerData, (function (arr, param) {
                return ArraySt$Cnn.push(arr, Curry._1(param.createGradientDataSum, param.state));
              }), []),
        [
          match[0],
          match[1]
        ],
        param[2]
      ]);
  var match$3 = match$2[2];
  var match$4 = _update(state, match$2[1], miniBatchSize);
  var state$1 = match$4[0];
  var logState = DebugLog$Cnn.addInfo(match$2[0], Log$Cnn.buildDebugLogMessage("weight, bias after update", match$4[1], undefined));
  return [
          [
            {
              optimizerData: state$1.optimizerData,
              allLayerData: match$4[2],
              lossData: state$1.lossData
            },
            logState
          ],
          [
            match$3[0],
            match$3[1]
          ],
          match$2[3]
        ];
}

function partition(data, labels, miniBatchSize) {
  if (ArraySt$Cnn.length(data) < miniBatchSize) {
    Exception$Cnn.throwErr("error");
  }
  return ArraySt$Cnn.reduceOneParami(data, (function (param, inputVector, index) {
                  var miniBatchPartitionData = param[0];
                  var labelVector = Caml_array.get(labels, index);
                  var miniBatchData = ArraySt$Cnn.push(param[1], [
                        inputVector,
                        labelVector
                      ]);
                  if (Caml_int32.mod_(index + 1 | 0, miniBatchSize) === 0) {
                    return [
                            ArraySt$Cnn.push(miniBatchPartitionData, miniBatchData),
                            []
                          ];
                  } else {
                    return [
                            miniBatchPartitionData,
                            miniBatchData
                          ];
                  }
                }), [
                [],
                []
              ])[0];
}

function shuffle(sampleCount) {
  var mnistData = Mnist.set(sampleCount, 1);
  var data = ArraySt$Cnn.sliceFrom(Mnist$Cnn.getMnistData(mnistData.training), -sampleCount | 0);
  var labels = ArraySt$Cnn.sliceFrom(Mnist$Cnn.getMnistLabels(mnistData.training), -sampleCount | 0);
  return [
          data,
          labels
        ];
}

var MiniBatch = {
  partition: partition,
  shuffle: shuffle
};

function _train(param, handleEachEpochFunc, param$1, param$2) {
  var miniBatchSize = param$2[1];
  var epochs = param$2[0];
  var isShuffle = param$1[1];
  var sampleCount = param$1[0];
  var _getCorrectRate = function (correctCount, errorCount) {
    return correctCount / (correctCount + errorCount) * 100 + "%";
  };
  Log$Cnn.printForDebug([
        "begin: ",
        epochs,
        miniBatchSize
      ]);
  var sampleData;
  if (isShuffle) {
    sampleData = undefined;
  } else {
    var match = shuffle(sampleCount);
    var labels = match[1];
    var data = match[0];
    var miniBatchPartitionData = partition(data, labels, miniBatchSize);
    sampleData = [
      data,
      labels,
      miniBatchPartitionData
    ];
  }
  return ArraySt$Cnn.reduceOneParam(ArraySt$Cnn.range(0, epochs - 1 | 0), (function (param, epochIndex) {
                var logState = DebugLog$Cnn.addInfo(param[1], "epoch index: " + epochIndex);
                var match;
                if (isShuffle) {
                  var match$1 = shuffle(sampleCount);
                  var labels = match$1[1];
                  var data = match$1[0];
                  var miniBatchPartitionData = partition(data, labels, miniBatchSize);
                  match = [
                    data,
                    labels,
                    miniBatchPartitionData
                  ];
                } else {
                  match = OptionSt$Cnn.getExn(sampleData);
                }
                var match$2 = ArraySt$Cnn.reduceOneParam(match[2], (function (param, miniBatchData) {
                        var match = param[1];
                        var match$1 = param[0];
                        return trainMiniBatchData([
                                    [
                                      match$1[0],
                                      match$1[1]
                                    ],
                                    [
                                      match[0],
                                      match[1]
                                    ],
                                    param[2]
                                  ], miniBatchData, miniBatchSize);
                      }), [
                      [
                        param[0],
                        logState
                      ],
                      [
                        0,
                        0
                      ],
                      0
                    ]);
                var match$3 = match$2[1];
                var match$4 = match$2[0];
                var match$5 = Curry._4(handleEachEpochFunc, [
                      match$4[0],
                      match$4[1]
                    ], epochIndex, match$2[2] / sampleCount, _getCorrectRate(match$3[0], match$3[1]));
                return [
                        match$5[0],
                        match$5[1]
                      ];
              }), [
              param[0],
              param[1]
            ]);
}

function train(param, param$1, param$2) {
  return _train([
              param[0],
              param[1]
            ], (function (param, param$1, loss, correctRate) {
                console.log([
                      "loss:",
                      loss
                    ]);
                console.log([
                      "getCorrectRate:",
                      correctRate
                    ]);
                return [
                        param[0],
                        param[1]
                      ];
              }), [
              param$1[0],
              param$1[1]
            ], [
              param$2[0],
              param$2[1]
            ]);
}

function inference(param, param$1) {
  var labels = param$1[1];
  var data = param$1[0];
  var match = ArraySt$Cnn.reduceOneParami(data, (function (param, input, i) {
          var match = param[0];
          var match$1 = forward([
                match[0],
                match[1]
              ], input, /* Inference */1);
          var match$2 = match$1[0];
          return [
                  [
                    match$2[0],
                    match$2[1]
                  ],
                  param[1] + CrossEntropyLoss$Cnn.compute(match$1[1], ArraySt$Cnn.getExn(labels, i), undefined, undefined)
                ];
        }), [
        [
          param[0],
          param[1]
        ],
        0
      ]);
  var match$1 = match[0];
  var sampleCount = ArraySt$Cnn.length(data);
  return [
          [
            match$1[0],
            match$1[1]
          ],
          match[1] / sampleCount
        ];
}

function trainAndInference(param, param$1, param$2) {
  var inferenceSampleCount = param$1[1];
  var match = param$1[0];
  return _train([
              param[0],
              param[1]
            ], (function (param, param$1, trainLoss, trainCorrectRate) {
                var match = inference([
                      param[0],
                      param[1]
                    ], shuffle(inferenceSampleCount));
                var match$1 = match[0];
                console.log("trainLoss: " + trainLoss + ", trainCorrectRate: " + trainCorrectRate + ", inferenceLoss: " + match[1] + " ");
                return [
                        match$1[0],
                        match$1[1]
                      ];
              }), [
              match[0],
              match[1]
            ], [
              param$2[0],
              param$2[1]
            ]);
}

exports._getAllLayerWeightAndBias = _getAllLayerWeightAndBias;
exports.create = create;
exports.setLossData = setLossData;
exports._getForwardOutput = _getForwardOutput;
exports.forward = forward;
exports._getPreviousLayerNetAndOutput = _getPreviousLayerNetAndOutput;
exports._getPreviousLayerActivatorData = _getPreviousLayerActivatorData;
exports.backward = backward;
exports._getInputNet = _getInputNet;
exports._update = _update;
exports.trainMiniBatchData = trainMiniBatchData;
exports.MiniBatch = MiniBatch;
exports._train = _train;
exports.train = train;
exports.inference = inference;
exports.trainAndInference = trainAndInference;
/* mnist Not a pure module */
