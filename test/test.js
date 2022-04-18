const MLR = require("../lib/index");

const x = [
  [0, 0],
  [1, 2],
  [2, 3],
  [3, 4]
];
// Y0 = X0 * 2, Y1 = X1 * 2, Y2 = X0 + X1
const y = [
  [0, 0, 0],
  [2, 4, 3],
  [4, 6, 5],
  [6, 8, 7]
];
// const x=[
//   [42.8, 40.0],
//   [63.5, 93.5],
//   [37.5, 35.5]
// ]
// const y=[[37], [50], [34]]

const mlr = new MLR(x, y, { intercept:true, statistics:false });
console.log(mlr);
console.log(mlr.predict([3, 3]));