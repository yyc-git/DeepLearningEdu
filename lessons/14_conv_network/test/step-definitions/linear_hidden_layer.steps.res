open Meta3dBsJestCucumber
open Cucumber
open Expect
open Operators

let feature = loadFeature("./test/features/linear_hidden_layer.feature")

defineFeature(feature, test => {
  test(."check layer weight gradient", ({given, \"when", \"and", then}) => {
    let stateForLinearLayer1 = ref(Obj.magic(1))
    let stateForLinearLayer2 = ref(Obj.magic(1))
    let result = ref(Obj.magic(1))
    let previousLayerOutputVector = [-2., -1.]->Vector.create
    let outLinearLayerNodeCount = 2
    let outputNet = ref(Obj.magic(1))
    let outputVector = ref(Obj.magic(1))
    let linearLayer1ActivatorData = SigmoidActivatorForCheckGradientTool.buildData()->Some
    let linearLayer2ActivatorData = IdentityActivator.buildData()->Some
    let linearLayer2Delta = ref(Obj.magic(1))
    let linearLayer1Delta = ref(Obj.magic(1))
    let actualGradient = ref(Obj.magic(1))
    let computeError = ref(Obj.magic(1))

    let _createAndInitLayer = given => {
      given("create and init layer", () => {
        let inLinearLayerNodeCount = 2
        let outLinearLayerNodeCount = 2

        let s = LinearLayer.create(~inLinearLayerNodeCount, ~outLinearLayerNodeCount, ())

        let s = {
          ...s,
          weight: Matrix.create(
            outLinearLayerNodeCount,
            inLinearLayerNodeCount,
            [-1.5, 2.5, -0.5, 1.5],
            // [1., 1., 1., 1.],
          ),
          bias: [1., 1.5],
          // bias: [1., 1.],
        }

        stateForLinearLayer1 := s

        let s = {
          ...s,
          weight: Matrix.create(
            outLinearLayerNodeCount,
            inLinearLayerNodeCount,
            // [-1.5, 2.5, -0.5, 1.5],
            [1.5, 0.5, -2., 1.],
          ),
          // bias: [1., 1.5],
          bias: [1., 1.],
        }

        stateForLinearLayer2 := s
      })
    }

    let _checkWeight = (
      computeError,
      (updateWMatrixByAddEpsilon, updateWMatrixBySubEpsilon),
      actualGradient,
      stateForLinearLayer1: LinearLayerStateType.state,
    ) => {
      open LinearLayer

      let epsilon = 10e-4

      let stateForLinearLayer1_1 = updateWMatrixByAddEpsilon(stateForLinearLayer1, epsilon)

      let (_, _, outputVector) = forward(
        stateForLinearLayer1_1->Obj.magic,
        linearLayer1ActivatorData,
        previousLayerOutputVector,
        NetworkType.Train,
      )

      let (_, _, outputVector) = forward(
        stateForLinearLayer2.contents->Obj.magic,
        linearLayer2ActivatorData,
        outputVector,
        NetworkType.Train,
      )

      let error1 = computeError(outputVector)

      let stateForLinearLayer1_2 = updateWMatrixBySubEpsilon(stateForLinearLayer1, epsilon)

      let (_, _, outputVector) = forward(
        stateForLinearLayer1_2->Obj.magic,
        linearLayer1ActivatorData,
        previousLayerOutputVector,
        NetworkType.Train,
      )

      let (_, _, outputVector) = forward(
        stateForLinearLayer2.contents->Obj.magic,
        linearLayer2ActivatorData,
        outputVector,
        NetworkType.Train,
      )

      let error2 = computeError(outputVector)

      let expectedGradient = (error1 -. error2) /. (2. *. epsilon)

      let expectedGradient = FloatTool.truncateFloatValue(expectedGradient, 4)
      let actualGradient = FloatTool.truncateFloatValue(actualGradient, 4)

      (expectedGradient->expect == actualGradient)->ignore
    }

    _createAndInitLayer(given)

    \"when"("design computeError and prepare next layer delta with identity activator", () => {
      computeError :=
        (
          output => {
            output->Vector.sum
          }
        )

      linearLayer2Delta :=
        ArraySt.range(0, outLinearLayerNodeCount - 1)->ArraySt.map(_ => 1.)->Vector.create
    })

    \"and"("forward layer", () => {
      let (_, outputNet_, outputVector_) = LinearLayer.forward(
        stateForLinearLayer1.contents->Obj.magic,
        linearLayer1ActivatorData,
        previousLayerOutputVector,
        NetworkType.Train,
      )

      outputVector := outputVector_
      outputNet := outputNet_
    })

    \"and"("compute layer delta", () => {
      linearLayer1Delta :=
        LinearLayer.bpDelta(
          linearLayer1ActivatorData,
          (outputVector.contents->Some, outputNet.contents),
          linearLayer2Delta.contents,
          stateForLinearLayer2.contents->Obj.magic,
        )
    })

    \"and"("compute layer weight gradient as actual weight gradient", () => {
      let (weightGradient, biasGradient) = LinearLayer.computeGradient(
        previousLayerOutputVector,
        linearLayer1Delta.contents->OptionSt.getExn,
        None,
      )

      actualGradient := weightGradient
    })

    \"and"("compute expect weight gradient by derivative definition", () => {
      ()
    })

    then("expect weight gradient should equal actual weight gradient", () => {
      CheckGradientTool.check((state: LinearLayerStateType.state, weight) => {
        ...state,
        weight: weight,
      }, _checkWeight(
        computeError.contents,
      ), stateForLinearLayer1.contents.weight, actualGradient.contents, stateForLinearLayer1.contents)
    })
  })
})
