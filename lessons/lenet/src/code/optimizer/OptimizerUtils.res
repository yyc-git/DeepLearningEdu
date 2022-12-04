let updateWeight = (
  update,
  weight,
  (learnRate, t, (beta1, beta2, epsion)),
  gradientDataSum,
  miniBatchSize,
  (vWeight, sWeight),
) => {
  let (newWeightValueArr, vtForWeightArr, stForWeightArr) =
    weight->NP.reduceMatrix(
      ((newWeightValueArr, vtArr, stArr), weightValue, rowIndex, colIndex) => {
        let (newWeightValue, (vt, st)) = update(
          weightValue,
          (learnRate, t, (beta1, beta2, epsion)),
          vWeight->Obj.magic->MatrixUtils.getValue(rowIndex, colIndex),
          sWeight->Obj.magic->MatrixUtils.getValue(rowIndex, colIndex),
          gradientDataSum->MatrixUtils.getValue(rowIndex, colIndex) /. miniBatchSize->Obj.magic,
        )

        (
          newWeightValueArr->ArraySt.push(newWeightValue),
          vtArr->ArraySt.push(vt),
          stArr->ArraySt.push(st),
        )
      },
      ([], [], []),
    )

  let (rowCount, colCount) = (Matrix.getRowCount(weight), Matrix.getColCount(weight))

  (
    Matrix.create(rowCount, colCount, newWeightValueArr),
    (
      Matrix.create(rowCount, colCount, vtForWeightArr),
      Matrix.create(rowCount, colCount, stForWeightArr),
    ),
  )
}
