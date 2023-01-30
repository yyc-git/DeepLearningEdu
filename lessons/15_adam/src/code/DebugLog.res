// TODO bdd test

type state = {
  isLog: bool,
  logInfo: string,
}

type layerWeightAndBias = {
  layerName: LayerAbstractType.layerName,
  weight: option<LayerAbstractType.weight>,
  bias: option<LayerAbstractType.bias>,
}

type oneData = {
  label: Vector.t,
  input: Vector.t,
}

type forward = {
  layerName: LayerAbstractType.layerName,
  output: LayerAbstractType.outputData,
}

type nextLayerDelta

type backward = {
  layerName: LayerAbstractType.layerName,
  nextLayerDelta: nextLayerDelta,
  gradientData: (LayerAbstractType.weightGradient, LayerAbstractType.biasGradient),
}

type update = {
  layerName: LayerAbstractType.layerName,
  miniBatchSize: int,
  weight: option<LayerAbstractType.weight>,
  bias: option<LayerAbstractType.bias>,
}

let create = isLog => {
  isLog: isLog,
  logInfo: "",
}

let addInfo = (state, info) => {
  state.isLog
    ? {
        ...state,
        logInfo: state.logInfo ++ info ++ "\n",
      }
    : state
}

let clearInfo = state => {
  state.isLog
    ? {
        ...state,
        logInfo: "",
      }
    : state
}

let createLogFile = (state, path) => {
  state.isLog
    ? {
        NodeExtend.writeFileStringSync(
          {
            j`${path}${Time.getNow()}.txt`
          },
          state.logInfo,
        )

        state->clearInfo
      }
    : state
}
