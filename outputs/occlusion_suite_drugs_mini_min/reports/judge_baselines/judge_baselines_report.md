# Judge Baselines Summary (openai_gpt5mini)

- Tau/choice agreement: 0.834
- DFQ/choice agreement: 0.838
- QE/choice agreement: 0.835
- Tau driver vs premise alignment: 0.690
- DFQ driver vs premise alignment: 0.691
- QE driver vs premise alignment: 0.690
- Pairwise driver vs premise alignment: n/a
- Tau OK rate: 1.000
- Pairwise OK rate: 0.000
- Pairwise consistency rate: n/a
- Pairwise cycle rate: n/a
- Pairwise mirror-complete rate: 0.000
- Pairwise mirror-consistency rate: n/a
- Pairwise mirror-inconsistency rate: n/a
- Tau/weights rank correlation: 0.800
- DFQ/weights rank correlation: 0.800
- QE/weights rank correlation: 0.800

## Stage A Weights
{
  "E": 0.40149455074559204,
  "A": 0.16951973301079212,
  "S": 0.21884475092769468,
  "D": 0.21014096531592116
}

## Behavioral Attribution
{
  "occlude_equalize": {
    "by_attribute": {
      "E": {
        "delta_pA": -0.04833333333333332,
        "n": 400,
        "ci95": [
          -0.12004166666666667,
          0.022937499999999982
        ]
      },
      "A": {
        "delta_pA": 0.015000000000000005,
        "n": 400,
        "ci95": [
          -0.015041666666666663,
          0.04666666666666667
        ]
      },
      "S": {
        "delta_pA": -0.0033333333333333314,
        "n": 400,
        "ci95": [
          -0.04583333333333334,
          0.0375
        ]
      },
      "D": {
        "delta_pA": 0.0025,
        "n": 400,
        "ci95": [
          -0.03710416666666666,
          0.04293749999999999
        ]
      }
    },
    "normalized": {
      "E": 0.6987951807228915,
      "A": 0.2168674698795182,
      "S": 0.04819277108433733,
      "D": 0.03614457831325302
    }
  },
  "occlude_drop": {
    "by_attribute": {
      "E": {
        "delta_pA": -0.01416666666666669,
        "n": 400,
        "ci95": [
          -0.07420833333333333,
          0.054604166666666655
        ]
      },
      "A": {
        "delta_pA": -0.005833333333333333,
        "n": 400,
        "ci95": [
          -0.040479166666666656,
          0.02877083333333331
        ]
      },
      "S": {
        "delta_pA": 0.025833333333333333,
        "n": 400,
        "ci95": [
          -0.021666666666666667,
          0.06543749999999998
        ]
      },
      "D": {
        "delta_pA": 0.0033333333333333327,
        "n": 400,
        "ci95": [
          -0.035833333333333335,
          0.045
        ]
      }
    },
    "normalized": {
      "E": 0.2881355932203393,
      "A": 0.1186440677966101,
      "S": 0.525423728813559,
      "D": 0.06779661016949148
    }
  }
}
