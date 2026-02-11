# Judge Baselines Summary (openai_gpt5nano)

- Tau/choice agreement: 0.590
- Tau driver vs premise alignment: 0.427
- Pairwise driver vs premise alignment: n/a
- Tau OK rate: 0.993
- Pairwise OK rate: 0.000

## Stage A Weights
{
  "E": 0.347090586456788,
  "A": 0.17479035389536796,
  "S": 0.2007255966525044,
  "D": 0.2773934629953397
}

## Behavioral Attribution
{
  "occlude_equalize": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.4,
        "n": 1
      }
    },
    "normalized": {
      "E": 1.0
    }
  },
  "occlude_drop": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.0,
        "n": 1
      },
      "A": {
        "delta_pA": -0.5,
        "n": 1
      },
      "S": {
        "delta_pA": -0.19999999999999996,
        "n": 1
      }
    },
    "normalized": {
      "E": 0.0,
      "A": 0.7142857142857143,
      "S": 0.28571428571428564
    }
  }
}
