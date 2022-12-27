- TODO rename lastLayerDeltaMap to currentLayerDeltaMap
- TODO rename backward, bpDeltaMap->deltaMap to nextLayerDeltaMap
- TODO rename LayerUtils.createLastLayerDeltaMap to createCurrentLayerDeltaMap

- TODO backward should receive ( inputs, inputNets )
    - TODO bpDeltaMap should use inputNets

- TODO change bpGradient->receive paddedInputs  to inputs
- TODO change bpGradient->receive deltaMap to currentLayerDeltaMap
- TODO rename bpGradient to computeGradient