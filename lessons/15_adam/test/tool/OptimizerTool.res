let buildLayerAdamWData = (
  ~learnRate=0.01,
  ~t=1,
  ~beta1=0.9,
  ~beta2=0.999,
  ~epsion=1e-6,
  // ~weightDecay=0.0,
  (),
) => {
  (
    AdamW.buildData(),
    ({t: t}: Optimizer.networkHyperparam),
    (
      {
        learnRate: learnRate,
        // weightDecay: weightDecay,
        beta1: beta1,
        beta2: beta2,
        epsion: epsion,
      }: Optimizer.layerHyperparam
    ),
  )
}
