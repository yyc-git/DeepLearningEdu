// type state = {
//   wMatrixBetweenLayer1Layer2: Matrix.t,
//   wMatrixBetweenLayer2Layer3: Matrix.t,
// }

// type feature = {
//   weight: float,
//   height: float,
// }

// type forwardOutput = ((Vector.t, Vector.t), (Vector.t, Vector.t))

// let _createWMatrix = (getValueFunc, firstLayerNodeCount, secondLayerNodeCount) => {
//   let row = secondLayerNodeCount
//   let col = firstLayerNodeCount + 1

//   Matrix.create(row, col, ArraySt.range(0, row * col - 1)->ArraySt.map(_ => getValueFunc()))
// }

// let createState = (layer1NodeCount, layer2NodeCount, layer3NodeCount): state => {
//   wMatrixBetweenLayer1Layer2: _createWMatrix(() => 0.1, layer1NodeCount, layer2NodeCount),
//   wMatrixBetweenLayer2Layer3: _createWMatrix(() => 0.1, layer2NodeCount, layer3NodeCount),
// }

// let _activateFunc = x => {
//   1. /. (1. +. Js.Math.exp(-.x))
// }

// // let forward = (state: state, feature: feature) => {
// let forward = (feature: feature, state: state): forwardOutput => {
//   let inputVector = Vector.create([feature.height, feature.weight, 1.0])

//   let layer2Net = Vector.transformMatrix(state.wMatrixBetweenLayer1Layer2, inputVector)

//   let layer2OutputVector = layer2Net->Vector.map(_activateFunc)

//   let layer3Net = Vector.transformMatrix(
//     state.wMatrixBetweenLayer2Layer3,
//     /* ! 注意：此处push 1.0 */
//     layer2OutputVector->Vector.push(1.0),
//   )

//   let layer3OutputVector = layer3Net->Vector.map(_activateFunc)

//   ((layer2Net, layer2OutputVector), (layer3Net, layer3OutputVector))
// }

// // let bpDelta = () => {
// // //TODO implement
// // Obj.magic(1)
// // }

// let backward = (((layer2Net, layer2OutputVector), (layer3Net, layer3OutputVector)): forwardOutput, state: state): (
//   (Vector.t, Matrix.t),
//   (Vector.t, Matrix.t),
// ) => {
//   TODO
// }

// let _convertLabelToFloat = label =>
//   switch label {
//   | Male => 0.
//   | Female => 1.
//   }

// let _computeLoss = (labels, outputs) => {
//   // Js.log((labels, outputs))
//   labels->ArraySt.reduceOneParami((. result, label, i) => {
//     result +. Js.Math.pow_float(~base=label -. outputs[i], ~exp=2.0)
//   }, 0.) /. ArraySt.length(labels)->Obj.magic
// }

// let train = (state: state, features: array<feature>, labels: array<label>): state => {
//   // let learnRate = 0.001
//   // let epochs = 100000

//   let learnRate = 0.1
//   let epochs = 1000

//   let n = features->ArraySt.length->Obj.magic

//   ArraySt.range(0, epochs - 1)->ArraySt.reduceOneParam((. state, epoch) => {
//     let state = features->ArraySt.reduceOneParami((. state, feature, i) => {
//       let label = labels[i]->_convertLabelToFloat

//       let ((layer2Delta, layer2Gradient), (layer3Delta, layer3Gradient)) =
//         forward(feature, state)->backward(state)

//       {
//         wMatrixBetweenLayer1Layer2: Matrix.sub(
//           state.wMatrixBetweenLayer1Layer2,
//           layer2Gradient->Matrix.multiplyScalar(learnRate, _),
//         ),
//         wMatrixBetweenLayer2Layer3: Matrix.sub(
//           state.wMatrixBetweenLayer2Layer3,
//           layer3Gradient->Matrix.multiplyScalar(learnRate, _),
//         ),
//       }
//     }, state)

//     mod(epoch, 10) == 0
//       ? {
//           // Js.log(state)
//           Js.log((
//             "loss: ",
//             _computeLoss(
//               labels->ArraySt.map(_convertLabelToFloat),
//               features->ArraySt.map(feature => {
//                 let (_, (_, y5)) = forward(feature, state)

//                 y5
//               }),
//             ),
//           ))

//           state
//         }
//       : state
//   }, state)
// }

// let state = createState()

// let features = [
//   {
//     weight: 50.,
//     height: 150.,
//   },
//   {
//     weight: 51.,
//     height: 149.,
//   },
//   {
//     weight: 60.,
//     height: 172.,
//   },
//   {
//     weight: 90.,
//     height: 188.,
//   },
// ]

// let labels = [Female, Female, Male, Male]

// let state = state->train(features, labels)
