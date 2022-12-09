type depthIndex = int

type state = {
  inputWidth: int,
  inputHeight: int,
  depthNumber: int,
  filterWidth: int,
  filterHeight: int,
  stride: int,
  outputWidth: int,
  outputHeight: int,
}

let create = (inputWidth, inputHeight, depthNumber, filterWidth, filterHeight, stride) => {
  let outputWidth = LayerUtils.computeOutputSize(inputWidth, filterWidth, 0, stride)
  let outputHeight = LayerUtils.computeOutputSize(inputHeight, filterHeight, 0, stride)

  {
    inputWidth: inputWidth,
    inputHeight: inputHeight,
    depthNumber: depthNumber,
    filterWidth: filterWidth,
    filterHeight: filterHeight,
    stride: stride,
    outputWidth: outputWidth,
    outputHeight: outputHeight,
  }
}

let forward = (
  state,
  inputs: ImmutableSparseMapType.t<depthIndex, Matrix.t>,
): ImmutableSparseMapType.t<depthIndex, Matrix.t> => {
  //TODO implement
  Obj.magic(1)
}
