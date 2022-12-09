

import * as NP$8_cnn from "../NP.bs.js";
import * as Log$8_cnn from "../Log.bs.js";
import * as Matrix$8_cnn from "../Matrix.bs.js";
import * as ArraySt$8_cnn from "../ArraySt.bs.js";
import * as LayerUtils$8_cnn from "../LayerUtils.bs.js";
import * as MatrixUtils$8_cnn from "../MatrixUtils.bs.js";
import * as ImmutableSparseMap$8_cnn from "../sparse_map/ImmutableSparseMap.bs.js";

function create(inputWidth, inputHeight, depthNumber, filterWidth, filterHeight, stride) {
  var outputWidth = LayerUtils$8_cnn.computeOutputSize(inputWidth, filterWidth, 0, stride);
  var outputHeight = LayerUtils$8_cnn.computeOutputSize(inputHeight, filterHeight, 0, stride);
  return {
          inputWidth: inputWidth,
          inputHeight: inputHeight,
          depthNumber: depthNumber,
          filterWidth: filterWidth,
          filterHeight: filterHeight,
          stride: stride,
          outputWidth: outputWidth,
          outputHeight: outputHeight
        };
}

function forward(state, inputs) {
  var outputRow = state.outputHeight;
  var outputCol = state.outputWidth;
  return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, state.depthNumber - 1 | 0), (function (outputMap, depthIndex) {
                var input = ImmutableSparseMap$8_cnn.getExn(inputs, depthIndex);
                var output = ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, outputRow - 1 | 0), (function (output, rowIndex) {
                        return ArraySt$8_cnn.reduceOneParam(ArraySt$8_cnn.range(0, outputCol - 1 | 0), (function (output, colIndex) {
                                      return MatrixUtils$8_cnn.setValue(output, NP$8_cnn.max(LayerUtils$8_cnn.getConvolutionRegion2D(input, rowIndex, colIndex, state.filterWidth, state.filterHeight, state.stride)), rowIndex, colIndex);
                                    }), output);
                      }), Matrix$8_cnn.create(outputRow, outputCol, []));
                return ImmutableSparseMap$8_cnn.set(outputMap, depthIndex, output);
              }), ImmutableSparseMap$8_cnn.createEmpty(undefined, undefined));
}

function init(param) {
  var inputs = NP$8_cnn.createMatrixMapByDataArr([
        [
          [
            0,
            1,
            1,
            0
          ],
          [
            2,
            3,
            2,
            2
          ],
          [
            1,
            0,
            0,
            2
          ],
          [
            0,
            1,
            1,
            0
          ]
        ],
        [
          [
            1,
            0,
            2,
            2
          ],
          [
            0,
            5,
            0,
            2
          ],
          [
            1,
            2,
            1,
            2
          ],
          [
            1,
            0,
            0,
            0
          ]
        ]
      ]);
  var state = create(4, 4, 2, 2, 2, 2);
  return [
          inputs,
          state
        ];
}

function test(param) {
  var outputMap = forward(param[1], param[0]);
  Log$8_cnn.printForDebug([
        "f:",
        outputMap
      ]);
  
}

var Test = {
  init: init,
  test: test
};

test(init(undefined));

export {
  create ,
  forward ,
  Test ,
  
}
/*  Not a pure module */
