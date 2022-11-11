why use softmax

why use cross entropy loss




\delta
  \delta_j = dE / dnetj
  (why replace activator, \delta equation not change?
    show derivation)

compute loss
  loss sum / sample count


compare:
use softmax instead of sigmoid increase convergence speed?




# Next

find "input for sigmoid is too large" reason


debug:
<!-- checkGradientExplosionOrDisappear -->
checkWeightVectorAndGradientVectorRadio (50 to 50000): find gradient disappear, find why




solve:
overflow
NaN