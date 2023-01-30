'use strict';


var throwErr = ((err) => { throw err; });

var buildErr = ((message) => { return new Error(message); });

function notImplement(param) {
  return throwErr("not implement");
}

exports.throwErr = throwErr;
exports.buildErr = buildErr;
exports.notImplement = notImplement;
/* No side effect */
