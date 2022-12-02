


var throwErr = ((err) => { throw err; });

var buildErr = ((message) => { return new Error(message); });

function notImplement(param) {
  return throwErr("not implement");
}

export {
  throwErr ,
  buildErr ,
  notImplement ,
  
}
/* No side effect */
