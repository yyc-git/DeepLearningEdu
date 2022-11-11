// let checkMatrix = (matrix, func) => {
//   matrix->Matrix.getData->Matrix.map(func)->ignore
// }

// let checkMatrixMap = (matrixMap, func) => {
//   matrixMap->NP.mapMatrixMap(checkMatrix(_, func))->ignore
// }

let _isGradientExplosionOrDisappear = %raw(` (gradient) => { return Number.isNaN(gradient) || Math.abs(gradient) < 0.000000001 || !Number.isFinite(gradient) } `)

let checkGradientExplosionOrDisappear = gradient => {
  _isGradientExplosionOrDisappear(gradient)
    ? {
        Js.log({j`checkGradientExplosionOrDisappear fail: $gradient`})
      }
    : ()
}

let _checkWeightValueAndGradientValueRadio = (weightValue, gradientValue) => {
  gradientValue == 0.
    ? ()
    : {
        let radio = 
          Js.Math.abs_float(weightValue) /. Js.Math.abs_float(gradientValue)
	
	
        radio -> Js.Math.floor > 50 || radio-> Js.Math.floor < 5
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
  input > 7. ? Js.log({j`input for sigmoid is too large: $input`}) : ()
}
