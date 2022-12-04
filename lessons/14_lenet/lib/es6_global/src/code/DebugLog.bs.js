

import * as Fs from "fs";
import * as Time$Cnn from "./Time.bs.js";

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

export {
  create ,
  addInfo ,
  clearInfo ,
  createLogFile ,
  
}
/* fs Not a pure module */
