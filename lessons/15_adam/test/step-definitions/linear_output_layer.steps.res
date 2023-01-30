open Meta3dBsJestCucumber
open Cucumber
open Expect
open Operators

open Sinon

let feature = loadFeature("./test/features/linear_output_layer.feature")

defineFeature(feature, test => {
  let state = ref(Obj.magic(1))

  test(."check layer weight gradient", ({given, \"when", \"and", then}) => {
    let result = ref(Obj.magic(1))
    let activatorData = SoftmaxActivator.buildData()->Some
    let previousLayerOutputVector = [-1., 0.5]->Vector.create
    let labelVector = [1., 0.]->Vector.create
    let outputVector = ref(Obj.magic(1))
    let nextLayerDelta = ref(Obj.magic(1))
    let actualGradient = ref(Obj.magic(1))

    let _createAndInitLayer = given => {
      given("create and init layer", () => {
        let inLinearLayerNodeCount = 2
        let outLinearLayerNodeCount = 2

        let s = LinearLayer.create(~inLinearLayerNodeCount, ~outLinearLayerNodeCount, ())

        let s = {
          ...s,
          weight: Matrix.create(outLinearLayerNodeCount, inLinearLayerNodeCount, [1., 10., 1., 5.]),
          bias: [0.5, 1.5],
        }

        state := s
      })
    }

    let _computeError = (label, output) => {
      CrossEntropyLoss.compute(~output, ~label, ~epsion=0., ())
    }


    _createAndInitLayer(given)

    \"when"("forward layer", () => {
      let (_, _, outputVector_) = LinearLayer.forward(
        state.contents->Obj.magic,
        activatorData,
        previousLayerOutputVector,
        NetworkType.Train,
      )

      outputVector := outputVector_
    })

    \"and"("compute layer delta", () => {
      nextLayerDelta := CrossEntropyLoss.computeDelta(outputVector.contents, labelVector)
    })

    \"and"("compute layer weight gradient as actual weight gradient", () => {
      let (weightGradient, biasGradient) = LinearLayer.computeGradient(
        previousLayerOutputVector,
        nextLayerDelta.contents,
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
      }, CheckGradientTool.checkWeight((
        (LinearLayer.forward, _computeError(labelVector), expect),
        activatorData,
        previousLayerOutputVector,
      )), state.contents.weight, actualGradient.contents, state.contents)
    })
  })
})
