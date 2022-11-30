// let _checkMatrix = (matrix, func) => {
//   matrix->Matrix.map(func)->ignore
// }

// let _checkMatrixMap = (matrixMap, func) => {
//   matrixMap->NP.mapMatrixMap(_checkMatrix(_, func))->ignore
// }

// let _isExplosionOrDisappear = %raw(` (value) => { return Number.isNaN(value) || Math.abs(value) < 0.000000001 || !Number.isFinite(value) || Math.abs(value) > 1e9 } `)

// let _checkExplosionOrDisappear = value => {
//   _isExplosionOrDisappear(value)
//     ? {
//         Log.logForDebug({j`_checkExplosionOrDisappear fail: $value`})
//       }
//     : ()
// }

let _isExplosion = %raw(` (value) => { return !Number.isFinite(value) || Math.abs(value) > 1e9 } `)

let _checkExplosion = value => {
  _isExplosion(value)
    ? {
        Exception.throwErr({j`_checkExplosion fail: $value`})
      }
    : ()
}

let checkOutputVectorExplosion = output => {
  output->Vector.map(_checkExplosion)->ignore
}

// let checkGradientVectorExplosionOrDisappear = gradient => {
//   gradient->Vector.map(_checkExplosionOrDisappear)->ignore
// }

// let checkGradientMatrixExplosionOrDisappear = gradient => {
//   _checkMatrix(gradient, _checkExplosionOrDisappear)
// }

// let _checkWeightValueAndGradientValueRatio = (weightValue, gradientValue) => {
//   gradientValue == 0.
//     ? ()
//     : {
//         let ratio = Js.Math.abs_float(weightValue) /. Js.Math.abs_float(gradientValue)

//         ratio->Js.Math.floor > 50 || ratio->Js.Math.floor < 5
//           ? Js.log({
//               j`_checkWeightValueAndGradientValueRatio fail: weightValue: $weightValue, gradientValue: $gradientValue, weighValue / gradientValue: $ratio`
//             })
//           : ()
//       }
// }

// let checkWeightVectorAndGradientVectorRatio = (weight, gradient) => {
//   weight
//   ->Vector.mapi((weightValue, i) => {
//     _checkWeightValueAndGradientValueRatio(
//       weightValue,
//       Array.unsafe_get(gradient->Vector.toArray, i),
//     )
//   })
//   ->ignore
// }

// let checkWeightMatrixAndGradientMatrixRatio = (weight, gradient) => {
//   weight
//   ->Matrix.mapi((weightValue, i) => {
//     _checkWeightValueAndGradientValueRatio(
//       weightValue,
//       Array.unsafe_get(gradient->Matrix.getData, i),
//     )
//   })
//   ->ignore
// }

// let checkWeightMapAndGradientMapRatio = (weightMap, gradientMap) => {
//   weightMap
//   ->ImmutableSparseMap.mapi((. weight, i) => {
//     checkWeightMatrixAndGradientMatrixRatio(weight, gradientMap->ImmutableSparseMap.getExn(i))
//   })
//   ->ignore
// }

// let checkSigmoidInputTooLarge = input => {
//   input > 7. ? Log.logForDebug({j`input for sigmoid is too large: $input`}) : ()
// }

// let _checkValueChangeRatio = (originValue, updateValue) => {
//   originValue == updateValue
//     ? 0
//     : originValue == 0.
//     ? {
//       let ratio = Js.Math.abs_float(originValue -. updateValue)

//       let bigFail = ratio->Js.Math.floor > 2
//       let smallFail = ratio < 0.000001

//       bigFail ? 2 : smallFail ? 3 : 1
//     }
//     : {
//         let ratio = Js.Math.abs_float(originValue -. updateValue) /. Js.Math.abs_float(originValue)

//         let bigFail = ratio->Js.Math.floor > 2 || updateValue->_isExplosion
//         let smallFail = ratio < 0.0001

//         bigFail ? 2 : smallFail ? 3 : 1
//       }
// }

// let checkWeightMatrixChangeRatio = (originWeight, updateWeight) => {
//   let (sum, zero, bigFail, smallFail) =
//     originWeight
//     ->Matrix.mapi((originValue, i) => {
//       _checkValueChangeRatio(originValue, Array.unsafe_get(updateWeight->Matrix.getData, i))
//     })
//     ->NP.reduceMatrix(((sum, zero, bigFail, smallFail), result, _, _) => {
//       let sum = sum + 1

//       switch result {
//       | 0 => (sum, zero + 1, bigFail, smallFail)
//       | 1 => (sum, zero, bigFail, smallFail)
//       | 2 => (sum, zero, bigFail + 1, smallFail)
//       | 3 => (sum, zero, bigFail, smallFail + 1)
//       }
//     }, (0, 0, 0, 0))

//   let failRatio = (bigFail + smallFail)->Obj.magic /. sum->Obj.magic *. 100.

//   j`sum: $sum, zero: $zero, bigFail: $bigFail, smallFail: $smallFail, failRatio: ${failRatio->Obj.magic} %`
//   ->Log.printForDebug
//   ->ignore
// }

// let checkBiasVectorChangeRatio = (originBiasVector, updateBiasVector) => {
//   let (sum, zero, bigFail, smallFail) =
//     originBiasVector
//     ->Vector.mapi((originValue, i) => {
//       _checkValueChangeRatio(originValue, Array.unsafe_get(updateBiasVector->Vector.toArray, i))
//     })
//     ->Vector.reducei((. (sum, zero, bigFail, smallFail), result, _) => {
//       let sum = sum + 1

//       switch result {
//       | 0 => (sum, zero + 1, bigFail, smallFail)
//       | 1 => (sum, zero, bigFail, smallFail)
//       | 2 => (sum, zero, bigFail + 1, smallFail)
//       | 3 => (sum, zero, bigFail, smallFail + 1)
//       }
//     }, (0, 0, 0, 0))

//   let failRatio = (bigFail + smallFail)->Obj.magic /. sum->Obj.magic *. 100.

//   j`sum: $sum, zero: $zero, bigFail: $bigFail, smallFail: $smallFail, failRatio: ${failRatio->Obj.magic} %`
//   ->Log.printForDebug
//   ->ignore
// }

// let checkBiasChangeRatio = (originBias, updateBias) => {
//   _checkValueChangeRatio(originBias, updateBias)->Log.printForDebug->ignore
// }

// let getConvLayerData = (convLayerState: ConvLayerStateType.state, filterIndex) => {
//   let filterState: FilterStateType.state =
//     convLayerState.filterStates->ImmutableSparseMap.getExn(filterIndex)

//   (filterState.weights, filterState.bias)
// }

// let getLinearLayerData = (linearLayerState: LinearLayerStateType.state) => {
//   (linearLayerState.weight, linearLayerState.bias)
// }

// let logMatrixValues = (matrix, count) => {
//   matrix->Matrix.getData->Js.Array.slice(~start=0, ~end_=count)->Log.printForDebug->ignore
// }

let _isGradientExplosionOrDisappear = %raw(` (gradient) => { return Number.isNaN(gradient) || (gradient !== 0.0 && Math.abs(gradient) < 0.0000001)|| !Number.isFinite(gradient) || Math.abs(gradient) > 1.0 } `)

let checkGradientExplosionOrDisappear = gradient => {
  gradient->Matrix.mapi((value, i) => {
    _isGradientExplosionOrDisappear(value)
      ? {
          Js.log({j`checkGradientExplosionOrDisappear fail: $value`})
        }
      : ()
  })
}

let _checkWeightValueAndGradientValueRadio = (weightValue, gradientValue) => {
  gradientValue == 0.
    ? ()
    : {
        let radio = Js.Math.abs_float(weightValue) /. Js.Math.abs_float(gradientValue)

        radio > 5000. || (Js.Math.abs_float(weightValue) > 0.001 && radio < 0.1)
          ? Js.log({
              j`checkWeightValueAndGradientValueRadio fail: $weightValue,  $gradientValue, radio: $radio`
            })
          : ()
      }
}

let checkWeightVectorAndGradientVectorRadio = (weight, gradient) => {
  weight
  ->Vector.mapi((weightValue, i) => {
    _checkWeightValueAndGradientValueRadio(
      weightValue,
      Array.unsafe_get(gradient->Vector.toArray, i),
    )
  })
  ->ignore
}

let checkWeightMatrixAndGradientMatrixRadio = (weight, gradient) => {
  weight
  ->Matrix.mapi((weightValue, i) => {
    _checkWeightValueAndGradientValueRadio(
      weightValue,
      Array.unsafe_get(gradient->Matrix.getData, i),
    )
  })
  ->ignore
}

// let checkWeightMapAndGradientMapRadio = (weightMap, gradientMap) => {
//   weightMap
//   ->ImmutableSparseMap.mapi((. weight, i) => {
//     checkWeightMatrixAndGradientMatrixRadio(weight, gradientMap->ImmutableSparseMap.getExn(i))
//   })
//   ->ignore
// }

let checkSigmoidInputTooLarge = input => {
  Js.Math.abs_float(input) > 7. ? Js.log({j`input for sigmoid is too large: $input`}) : ()
}
