'use strict';


function create(x, y) {
  return [
          x,
          y
        ];
}

function getFirst(param) {
  return param[0];
}

function getLast(param) {
  return param[1];
}

exports.create = create;
exports.getFirst = getFirst;
exports.getLast = getLast;
/* No side effect */
