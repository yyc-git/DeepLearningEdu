open Meta3dBsJestCucumber
open Cucumber
open Expect
open Operators

let check = (updateWeight, checkWeight, weight, weightGradient: Matrix.t, state) => {
  weight->Matrix.forEachRow(rowIndex => {
    weight->Matrix.forEachCol(colIndex => {
      checkWeight(
        (
          (state, epsilon) => {
            let (row, col, data) = weight
            let data = data->Js.Array.copy

            data[
              MatrixUtils.computeIndex(col, rowIndex, colIndex)
            ] =
              data[MatrixUtils.computeIndex(col, rowIndex, colIndex)] +. epsilon

            updateWeight(state, (row, col, data))
          },
          (state, epsilon) => {
            let (row, col, data) = weight
            let data = data->Js.Array.copy

            data[
              MatrixUtils.computeIndex(col, rowIndex, colIndex)
            ] =
              data[MatrixUtils.computeIndex(col, rowIndex, colIndex)] -. epsilon

            updateWeight(state, (row, col, data))
          },
        ),
        MatrixUtils.getValue(weightGradient, rowIndex, colIndex),
        state,
      )
    })
  })
}

let checkWeight = (
  ((forward, computeError, expect), activatorData, previousOutputVector),
  (updateWMatrixByAddEpsilon, updateWMatrixBySubEpsilon),
  actualGradient,
  state: LinearLayerStateType.state,
) => {
  open LinearLayer

  let epsilon = 10e-4

  let newState1 = updateWMatrixByAddEpsilon(state, epsilon)

  let (newState1, _, outputVector) = forward(
    newState1->Obj.magic,
    activatorData,
    previousOutputVector,
    NetworkType.Train,
  )

  let error1 = computeError(outputVector)

  let newState2 = updateWMatrixBySubEpsilon(state, epsilon)

  let (newState2, _, outputVector) = forward(
    newState2->Obj.magic,
    activatorData,
    previousOutputVector,
    NetworkType.Train,
  )

  let error2 = computeError(outputVector)

  let expectedGradient = (error1 -. error2) /. (2. *. epsilon)

  let expectedGradient = FloatTool.truncateFloatValue(expectedGradient, 4)
  let actualGradient = FloatTool.truncateFloatValue(actualGradient, 4)

  (expectedGradient->expect == actualGradient)->ignore
}
