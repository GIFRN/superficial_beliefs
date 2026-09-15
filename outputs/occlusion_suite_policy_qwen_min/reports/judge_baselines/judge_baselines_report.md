# Judge Baselines Summary (local_qwen3_14b_vllm_minimal)

- Tau/choice agreement: 0.781
- DFQ/choice agreement: 0.760
- QE/choice agreement: 0.780
- Tau driver vs premise alignment: 0.616
- DFQ driver vs premise alignment: 0.614
- QE driver vs premise alignment: 0.607
- Pairwise driver vs premise alignment: n/a
- Tau OK rate: 1.000
- Pairwise OK rate: 0.000
- Pairwise consistency rate: n/a
- Pairwise cycle rate: n/a
- Pairwise mirror-complete rate: 0.000
- Pairwise mirror-consistency rate: n/a
- Pairwise mirror-inconsistency rate: n/a
- Tau/weights rank correlation: 1.000
- DFQ/weights rank correlation: 1.000
- QE/weights rank correlation: 1.000

## Stage A Weights
{
  "E": 0.3839214951335835,
  "A": 0.20986622284404155,
  "S": 0.23613500263286388,
  "D": 0.17007727938951112
}

## Behavioral Attribution
{
  "occlude_equalize": {
    "by_attribute": {
      "E": {
        "delta_pA": -0.00916666666666667,
        "n": 400,
        "ci95": [
          -0.07381249999999999,
          0.05754166666666664
        ]
      },
      "A": {
        "delta_pA": -0.004166666666666663,
        "n": 400,
        "ci95": [
          -0.04670833333333333,
          0.037500000000000006
        ]
      },
      "S": {
        "delta_pA": -0.0425,
        "n": 400,
        "ci95": [
          -0.09170833333333332,
          0.0037708333333333183
        ]
      },
      "D": {
        "delta_pA": -0.0058333333333333345,
        "n": 400,
        "ci95": [
          -0.04254166666666666,
          0.030833333333333334
        ]
      }
    },
    "normalized": {
      "E": 0.1486486486486487,
      "A": 0.0675675675675675,
      "S": 0.6891891891891893,
      "D": 0.09459459459459461
    }
  },
  "occlude_drop": {
    "by_attribute": {
      "E": {
        "delta_pA": -0.06333333333333332,
        "n": 400,
        "ci95": [
          -0.13166666666666665,
          0.0008333333333333348
        ]
      },
      "A": {
        "delta_pA": -0.018333333333333326,
        "n": 400,
        "ci95": [
          -0.059166666666666666,
          0.02302083333333324
        ]
      },
      "S": {
        "delta_pA": -0.03416666666666666,
        "n": 400,
        "ci95": [
          -0.08416666666666665,
          0.010437499999999985
        ]
      },
      "D": {
        "delta_pA": 0.0016666666666666718,
        "n": 400,
        "ci95": [
          -0.041270833333333326,
          0.04166666666666667
        ]
      }
    },
    "normalized": {
      "E": 0.5390070921985816,
      "A": 0.15602836879432622,
      "S": 0.2907801418439716,
      "D": 0.014184397163120614
    }
  }
}
