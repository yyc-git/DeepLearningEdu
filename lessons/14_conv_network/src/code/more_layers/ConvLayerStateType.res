type filterIndex = int

type depthIndex = int

type state = {
  inputWidth: int,
  inputHeight: int,
  depthNumber: int,
  filterWidth: int,
  filterHeight: int,
  filterNumber: int,
  filterStates: ImmutableSparseMapType.t<filterIndex, FilterStateType.state>,
  zeroPadding: int,
  stride: int,
  outputWidth: int,
  outputHeight: int,
}
