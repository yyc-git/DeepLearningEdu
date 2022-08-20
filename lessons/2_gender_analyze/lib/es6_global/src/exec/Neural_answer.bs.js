


function createState(param) {
  return {
          weight1: Math.random(),
          weight2: Math.random(),
          bias: Math.random()
        };
}

function train(state, sampleData) {
  return {
          weight1: 1.0,
          weight2: -2.0,
          bias: -49.0
        };
}

function _activateFunc(x) {
  return x;
}

function _convert(x) {
  if (x === 0) {
    return /* Male */0;
  }
  if (x === 1) {
    return /* Female */1;
  }
  throw {
        RE_EXN_ID: "Match_failure",
        _1: [
          "Neural_answer.res",
          33,
          2
        ],
        Error: new Error()
      };
}

function inference(state, sampleData) {
  return _convert(sampleData.height * state.weight1 + sampleData.weight * state.weight2 + state.bias);
}

var state = createState(undefined);

var gender = inference({
      weight1: 1.0,
      weight2: -2.0,
      bias: -49.0
    }, {
      weight: 50,
      height: 150
    });

console.log(gender);

export {
  createState ,
  train ,
  _activateFunc ,
  _convert ,
  inference ,
  state ,
  gender ,
  
}
/* state Not a pure module */
