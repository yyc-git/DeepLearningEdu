


function createState(param) {
  return {
          weight13: 0.1,
          weight14: 0.1,
          weight23: 0.1,
          weight24: 0.1,
          weight35: 0.1,
          weight45: 0.1,
          bias3: 0.1,
          bias4: 0.1,
          bias5: 0.1
        };
}

function _activateFunc(x) {
  return 1 / (1 + Math.exp(-x));
}

var Neural_forward_answer = {};

function forward(state, feature) {
  var net = feature.height * state.weight1 + feature.weight * state.weight2 + state.bias;
  return [
          net,
          _activateFunc(net)
        ];
}

var Neural_forward = {
  forward: forward
};

function forward$1(state, feature) {
  var match = forward({
        weight1: state.weight13,
        weight2: state.weight23,
        bias: state.bias3
      }, feature);
  var y3 = match[1];
  var match$1 = forward({
        weight1: state.weight14,
        weight2: state.weight24,
        bias: state.bias4
      }, feature);
  var y4 = match$1[1];
  var match$2 = forward({
        weight1: state.weight35,
        weight2: state.weight45,
        bias: state.bias5
      }, {
        weight: y3,
        height: y4
      });
  return [
          y3,
          y4,
          match$2[1]
        ];
}

var state = {
  weight13: 0.1,
  weight14: 0.1,
  weight23: 0.1,
  weight24: 0.1,
  weight35: 0.1,
  weight45: 0.1,
  bias3: 0.1,
  bias4: 0.1,
  bias5: 0.1
};

var feature = {
  weight: 50,
  height: 150
};

console.log(forward$1(state, feature));

export {
  createState ,
  _activateFunc ,
  Neural_forward_answer ,
  Neural_forward ,
  forward$1 as forward,
  state ,
  feature ,
  
}
/*  Not a pure module */
