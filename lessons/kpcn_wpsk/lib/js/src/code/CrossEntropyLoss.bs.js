'use strict';

var Vector$Cnn = require("./Vector.bs.js");

var computeDelta = Vector$Cnn.sub;

function compute(output, label, epsionOpt, param) {
  var epsion = epsionOpt !== undefined ? epsionOpt : 1e-10;
  return -Vector$Cnn.dot(label, Vector$Cnn.multiplyScalar(Vector$Cnn.map(Vector$Cnn.addScalar(output, epsion), (function (prim) {
                        return Math.log(prim);
                      })), 1 / (1 + epsion)));
}

function buildData(param) {
  return {
          computeDelta: computeDelta
        };
}

exports.computeDelta = computeDelta;
exports.compute = compute;
exports.buildData = buildData;
/* No side effect */
