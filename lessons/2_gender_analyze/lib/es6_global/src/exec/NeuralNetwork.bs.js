


function createState(param) {
  return {
          weight13: Math.random(),
          weight14: Math.random(),
          weight23: Math.random(),
          weight24: Math.random(),
          weight35: Math.random(),
          weight45: Math.random(),
          bias3: Math.random(),
          bias4: Math.random(),
          bias5: Math.random()
        };
}

function train(state, allSampleData) {
  return 1;
}

function _activateFunc(x) {
  return x;
}

function _convert(x) {
  if (x !== 0) {
    if (x !== 1) {
      return /* InValid */2;
    } else {
      return /* Female */1;
    }
  } else {
    return /* Male */0;
  }
}

function forward(state, sampleData) {
  return 1;
}

function inference(state, sampleData) {
  return _convert(1);
}

createState(undefined);

var allSampleData = [
  {
    weight: 50,
    height: 150
  },
  {
    weight: 51,
    height: 149
  },
  {
    weight: 60,
    height: 172
  },
  {
    weight: 90,
    height: 188
  }
];

allSampleData.forEach(function (sampleData) {
      console.log(_convert(1));
      
    });

var state = 1;

export {
  createState ,
  train ,
  _activateFunc ,
  _convert ,
  forward ,
  inference ,
  allSampleData ,
  state ,
  
}
/*  Not a pure module */
