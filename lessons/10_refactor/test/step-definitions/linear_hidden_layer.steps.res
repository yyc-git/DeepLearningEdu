open Meta3dBsJestCucumber
open Cucumber
open Expect
open Operators

let feature = loadFeature("./test/features/linear_hidden_layer.feature")

defineFeature(feature, test => {
  let state = ref(Obj.magic(1))

  test(."check layer weight gradient", ({given, \"when", \"and", then}) => {
    let result = ref(Obj.magic(1))
    let previousLayerOutputVector = [-2., -1.]->Vector.create
    let outLinearLayerNodeCount = 2
    let outputNet = ref(Obj.magic(1))
    let outputVector = ref(Obj.magic(1))
    let activatorData = SigmoidActivatorForCheckGradientTool.buildData()->Some
    let nextLayerActivatorData = IdentityActivator.buildData()->Some
    let nextLayerDelta = ref(Obj.magic(1))
    let layerDelta = ref(Obj.magic(1))
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

        state := s
      })
    }

    let _checkWeight = (
      computeError,
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
        previousLayerOutputVector,
        NetworkType.Train,
      )

      let (newState1, _, outputVector) = forward(
        state->Obj.magic,
        nextLayerActivatorData,
        outputVector,
        NetworkType.Train,
      )

      let error1 = computeError(outputVector)

      let newState2 = updateWMatrixBySubEpsilon(state, epsilon)

      let (newState2, _, outputVector) = forward(
        newState2->Obj.magic,
        activatorData,
        previousLayerOutputVector,
        NetworkType.Train,
      )

      let (newState2, _, outputVector) = forward(
        state->Obj.magic,
        nextLayerActivatorData,
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

      nextLayerDelta :=
        ArraySt.range(0, outLinearLayerNodeCount - 1)->ArraySt.map(_ => 1.)->Vector.create
    })

    \"and"("forward layer", () => {
      let (_, outputNet_, outputVector_) = LinearLayer.forward(
        state.contents->Obj.magic,
        activatorData,
        previousLayerOutputVector,
        NetworkType.Train,
      )

      outputVector := outputVector_
      outputNet := outputNet_
    })

    \"and"("compute layer delta", () => {
      layerDelta :=
        LinearLayer.bpDelta(
          activatorData,
          (outputVector.contents->Some, outputNet.contents),
          nextLayerDelta.contents,
          state.contents->Obj.magic,
        )
    })

    \"and"("compute layer weight gradient as actual weight gradient", () => {
      let (weightGradient, biasGradient) = LinearLayer.computeGradient(
        previousLayerOutputVector,
        layerDelta.contents->OptionSt.getExn,
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
      ), state.contents.weight, actualGradient.contents, state.contents)
    })
  })
})
