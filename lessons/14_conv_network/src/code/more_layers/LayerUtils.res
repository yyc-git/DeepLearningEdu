// TODO move to Layer?

let computeOutputSize = (inputSize, filterSize, zeroPadding, stride) => {
  (inputSize - filterSize + 2 * zeroPadding) / stride + 1
}

let getConvolutionRegion2D = (input, rowIndex, colIndex, filterWidth, filterHeight, stride) => {
  NP.getMatrixRegion(colIndex * stride, filterWidth, rowIndex * stride, filterHeight, input)
}

let getConvolutionRegion3D = (inputs, rowIndex, colIndex, filterWidth, filterHeight, stride) => {
  inputs->ImmutableSparseMap.map((. input) => {
    getConvolutionRegion2D(input, rowIndex, colIndex, filterWidth, filterHeight, stride)
  })
}

let createPreviousLayerDeltaMap = ((depthNumber, inputWidth, inputHeight)) => {
  NP.zeroMatrixMap(depthNumber, inputHeight, inputWidth)
}
