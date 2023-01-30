// let uniform = (~range=1e-4, ()) => {
//   (Js.Math.random() *. 2. -. 1.) *. range
// }

// let random = () => {
//   Js.Math.random()
// }

let constant = value => {
  value
}

let normal = %raw(` (randomFunc, mean, std) => {
    var y1;
    var y2;

      var x1, x2, w;

      do {

        x1 = 2.0 * randomFunc() - 1.0;

        x2 = 2.0 * randomFunc() - 1.0;

        w = x1 * x1 + x2 * x2;

      } while (w >= 1.0);

      w = Math.sqrt((-2.0 * Math.log(w)) / w);

      y1 = x1 * w;

      y2 = x2 * w;

    var retval = mean + std * y1;

    if (retval > 0)

      return retval;

    return -retval;


} `)
