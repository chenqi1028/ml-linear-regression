# ml-linear-regression

<p align="left">
  <a href="https://npmcharts.com/compare/ml-linear-regression?minimal=true"><img src="https://img.shields.io/npm/dm/ml-linear-regression.svg?sanitize=true"></a>
  <a href="https://www.npmjs.com/package/ml-linear-regression"><img src="https://img.shields.io/npm/v/ml-linear-regression.svg?sanitize=true" alt="Version"></a>
  <a href="https://www.npmjs.com/package/ml-linear-regression"><img src="https://img.shields.io/npm/l/ml-linear-regression.svg?sanitize=true" alt="License"></a>
</p>

Multivariate linear regression.

## Installation

`npm install --save ml-linear-regression`

## API

### new MLR(x, y[, options])

**Arguments**

- `x`: Matrix containing the inputs
- `y`: Matrix containing the outputs

**Options**

- `intercept`: boolean indicating if intercept terms should be computed (default: true)
- `statistics`: boolean for calculating and returning regression statistics (default: true)

## Usage

```js
import MLR from "ml-regression-multivariate-linear";

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
const mlr = new MLR(x, y);
console.log(mlr);
//weights Non standardized regression coefficient
//stdWeights Standardized regression coefficient
console.log(mlr.predict([3, 3]));
// [6, 6, 6]
```

## License

[MIT](./LICENSE)
