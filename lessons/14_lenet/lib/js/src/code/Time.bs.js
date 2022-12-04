'use strict';


var getNow = (() => { return new Date().getTime()});

exports.getNow = getNow;
/* No side effect */
