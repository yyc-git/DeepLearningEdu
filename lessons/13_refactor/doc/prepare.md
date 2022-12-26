重构LinearLayer，分离bias, weight

add conv gradient check test



- TODO need rename LayerAbstractType->bpDelta:
    -  "currentLayerDelta" to nextLayerDelta
    -  returned "previousLayerDelta" to currentLayerDelta
  
    -  TODO also rename related implement code