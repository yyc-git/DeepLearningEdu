type filterIndex = int

type depthIndex = int

type state = {
  inputWidth: int,
  inputHeight: int,
  depthNumber: int,
  filterWidth: int,
  filterHeight: int,
  filterNumber: int,
  filterStates: ImmutableSparseMapType.t<filterIndex, Filter.state>,
  zeroPadding: int,
  stride: int,
  outputWidth: int,
  outputHeight: int,
  leraningRate: float,
}

let forward = (activate, state, inputs: ImmutableSparseMapType.t<depthIndex, Matrix.t>) => {
  //TODO implement
  Obj.magic(1)
}
