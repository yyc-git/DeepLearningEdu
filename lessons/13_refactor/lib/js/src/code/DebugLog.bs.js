'use strict';

var Fs = require("fs");
var Time$Cnn = require("./Time.bs.js");

function create(isLog) {
  return {
          isLog: isLog,
          logInfo: ""
        };
}

function addInfo(state, info) {
  if (state.isLog) {
    return {
            isLog: state.isLog,
            logInfo: state.logInfo + info + "\n"
          };
  } else {
    return state;
  }
}

function clearInfo(state) {
  if (state.isLog) {
    return {
            isLog: state.isLog,
            logInfo: ""
          };
  } else {
    return state;
  }
}

function createLogFile(state, path) {
  if (state.isLog) {
    Fs.writeFileSync(path + Time$Cnn.getNow(undefined) + ".txt", state.logInfo);
    return clearInfo(state);
  } else {
    return state;
  }
}

exports.create = create;
exports.addInfo = addInfo;
exports.clearInfo = clearInfo;
exports.createLogFile = createLogFile;
/* fs Not a pure module */
