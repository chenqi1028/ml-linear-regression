const MLR = require("ml-regression-multivariate-linear");
const math = require("mathjs")

export default class MlLinearRegression {
  constructor(x, y){
    const flipX = [],flipY = []
    for (let i = 0; i < x.length; i++) {
      for (let j = 0; j < x[i].length; j++) {
        if (!flipX[j]) {
          flipX[j] = []
        }
        flipX[j].push(Number(x[i][j]));
      }
    }
    for (let i = 0; i < y.length; i++) {
      for (let j = 0; j < y[i].length; j++) {
        if (!flipY[j]) {
          flipY[j] = []
        }
        flipY[j].push(Number(y[i][j]));
      }
    }

    const stdArrX = [];
    const stdArrY = [];
    for (let i = 0; i < flipX.length; i++) {
      stdArrX[i] = math.std(flipX[i])
    }
    for (let i = 0; i < flipY.length; i++) {
      stdArrY[i] = math.std(flipY[i])
    }
    
    const mlr = new MLR(x,y);

    this.stdWeight = JSON.parse(JSON.stringify(mlr.weights));
    for (let i = 0; i < mlr.weights.length-1; i++) {
      for (let j = 0; j < mlr.weights[i].length; j++) {
        this.stdWeight[i][j] = mlr.weights[i][j] * stdArrX[i] / stdArrY[j]
      }
    }
    this.stdWeight.splice(-1)

    this.statistics = mlr.statistics
    this.weights = mlr.weights;
    this.inputs = mlr.inputs;
    this.outputs = mlr.outputs;
    this.intercept = mlr.intercept;
    this.stdError = mlr.stdError;
    this.stdErrorMatrix = mlr.stdErrorMatrix;
    this.stdErrors = mlr.stdErrors;
    this.tStats = mlr.tStats;
  }

  predict(x) {
    if (Array.isArray(x)) {
      if (typeof x[0] === 'number') {
        return this._predict(x);
      } else if (Array.isArray(x[0])) {
        const y = new Array(x.length);
        for (let i = 0; i < x.length; i++) {
          y[i] = this._predict(x[i]);
        }
        return y;
      }
    } else if (Matrix.isMatrix(x)) {
      const y = new Matrix(x.rows, this.outputs);
      for (let i = 0; i < x.rows; i++) {
        y.setRow(i, this._predict(x.getRow(i)));
      }
      return y;
    }
    throw new TypeError('x must be a matrix or array of numbers');
  }

  _predict(x) {
    const result = new Array(this.outputs);
    if (this.intercept) {
      for (let i = 0; i < this.outputs; i++) {
        result[i] = this.weights[this.inputs][i];
      }
    } else {
      result.fill(0);
    }
    for (let i = 0; i < this.inputs; i++) {
      for (let j = 0; j < this.outputs; j++) {
        result[j] += this.weights[i][j] * x[i];
      }
    }
    return result;
  }
}
