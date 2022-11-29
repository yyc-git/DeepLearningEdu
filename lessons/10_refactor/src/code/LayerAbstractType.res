type none

type activatorData = ActivatorType.data

type input<'data> = 'data

type state

type net<'data> = 'data

type output<'data> = 'data

type forward<'inputData, 'outputData> = (
  state,
  option<activatorData>,
  input<'inputData>,
  NetworkType.phase,
) => // option<state>,
(state, option<net<'outputData>>, output<'outputData>)

type previousLayerActivatorData = ActivatorType.data
// type layerActivatorData = ActivatorType.data

type previousLayerNet<'data> = 'data

type previousLayerOutput<'data> = 'data

type currentLayerDelta<'data> = 'data

type previousLayerDelta<'data> = 'data

type bpDelta<'inputData, 'outputData> = (
  option<previousLayerActivatorData>,
  // option<layerActivatorData>,
  (option<previousLayerOutput<'inputData>>, option<previousLayerNet<'inputData>>),
  currentLayerDelta<'outputData>,
  state,
) => option<previousLayerDelta<'inputData>>

type gradientData<'weightGradient, 'biasGradient> = ('weightGradient, 'biasGradient)

type computeGradient<'inputData, 'outputData, 'weightGradient, 'biasGradient> = (
  input<'inputData>,
  currentLayerDelta<'outputData>,
  option<state>,
) => gradientData<'weightGradient, 'biasGradient>

type backward<'inputData, 'outputData, 'weightGradient, 'biasGradient> = (
  option<previousLayerActivatorData>,
  // option<layerActivatorData>,
  (option<previousLayerOutput<'inputData>>, option<previousLayerNet<'inputData>>),
  currentLayerDelta<'outputData>,
  state,
) => (option<previousLayerDelta<'inputData>>, option<gradientData<'weightGradient, 'biasGradient>>)

type learnRate = float

type t = int

type beta1 = float

type beta2 = float

type epsion = float

type weightDecay = float

type miniBatchSize = int

type update<'weightGradientDataSum, 'biasGradientDataSum> = (
  state,
  (Optimizer.optimizerFuncs, Optimizer.networkHyperparam, Optimizer.layerHyperparam),
  (miniBatchSize, option<('weightGradientDataSum, 'biasGradientDataSum)>),
) => state

type createGradientDataSum<'weightGradientDataSum, 'biasGradientDataSum> = state => option<(
  'weightGradientDataSum,
  'biasGradientDataSum,
)>

type addToGradientDataSum<'weightGradient, 'biasGradient> = (
  option<('weightGradient, 'biasGradient)>,
  option<('weightGradient, 'biasGradient)>,
) => option<('weightGradient, 'biasGradient)>

type inputData

type outputData

type weightGradient

type biasGradient

// type layerData<'inputData, 'outputData, 'weightGradient, 'biasGradient> = {
//   state: state,
//   forward: forward<'inputData, 'outputData>,
//   backward: backward<'inputData, 'outputData, 'weightGradient, 'biasGradient>,
//   update: update<'weightGradient, 'biasGradient>,
//   createGradientDataSum: createGradientDataSum<'weightGradient, 'biasGradient>,
//   addToGradientDataSum: addToGradientDataSum<'weightGradient, 'biasGradient>,
//   activatorData: option<ActivatorType.data>,
// }

// type createLayerData<'inputData, 'outputData, 'weightGradient, 'biasGradient> = unit => layerData<
//   'inputData,
//   'outputData,
//   'weightGradient,
//   'biasGradient,
// >

type layerName = [#fold | #conv | #maxPooling | #flatten | #linear | #dropout]

type weight

type bias

// type vWeight = weightGradient

// type vBias = biasGradient

// type sWeight = weightGradient

// type sBias = biasGradient

// type adamData = {
//   vWeight: vWeight,
//   vBias: vBias,
//   sWeight: sWeight,
//   sBias: sBias,
// }

type layerData = {
  layerName: layerName,
  state: state,
  forward: forward<inputData, outputData>,
  backward: backward<inputData, outputData, weightGradient, biasGradient>,
  update: update<weightGradient, biasGradient>,
  createGradientDataSum: createGradientDataSum<weightGradient, biasGradient>,
  addToGradientDataSum: addToGradientDataSum<weightGradient, biasGradient>,
  activatorData: option<ActivatorType.data>,
  getWeight: state => option<weight>,
  getBias: state => option<bias>,
}

type createLayerData = (state, option<ActivatorType.data>) => layerData
