# Judge Baselines Summary (local_ministral3_14b_instruct_vllm_minimal)

- Tau/choice agreement: 0.723
- DFQ/choice agreement: 0.714
- QE/choice agreement: 0.722
- Tau driver vs premise alignment: 0.535
- DFQ driver vs premise alignment: 0.535
- QE driver vs premise alignment: 0.535
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
  "E": 0.2993201004189427,
  "A": 0.13349347274677265,
  "S": 0.35480410041564764,
  "D": 0.21238232641863694
}

## Behavioral Attribution
{
  "occlude_equalize": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.01875000000000001,
        "n": 400,
        "ci95": [
          -0.023812499999999997,
          0.05875
        ]
      },
      "A": {
        "delta_pA": 0.029583333333333336,
        "n": 400,
        "ci95": [
          -0.002499999999999997,
          0.06355208333333333
        ]
      },
      "S": {
        "delta_pA": 0.03624999999999999,
        "n": 400,
        "ci95": [
          -0.018135416666666664,
          0.08085416666666666
        ]
      },
      "D": {
        "delta_pA": 0.010416666666666664,
        "n": 400,
        "ci95": [
          -0.026708333333333327,
          0.045
        ]
      }
    },
    "normalized": {
      "E": 0.19736842105263167,
      "A": 0.31140350877192985,
      "S": 0.38157894736842096,
      "D": 0.10964912280701752
    }
  },
  "occlude_drop": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.02625,
        "n": 400,
        "ci95": [
          -0.018749999999999996,
          0.06752083333333332
        ]
      },
      "A": {
        "delta_pA": -0.01541666666666666,
        "n": 400,
        "ci95": [
          -0.05335416666666666,
          0.016906249999999973
        ]
      },
      "S": {
        "delta_pA": 0.02125,
        "n": 400,
        "ci95": [
          -0.02922916666666666,
          0.06791666666666667
        ]
      },
      "D": {
        "delta_pA": -0.006249999999999997,
        "n": 400,
        "ci95": [
          -0.0404375,
          0.023333333333333338
        ]
      }
    },
    "normalized": {
      "E": 0.3795180722891567,
      "A": 0.22289156626506018,
      "S": 0.3072289156626507,
      "D": 0.0903614457831325
    }
  }
}
