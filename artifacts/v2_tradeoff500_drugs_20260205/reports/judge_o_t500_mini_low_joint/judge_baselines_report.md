# Judge Baselines Summary (openai_gpt5mini)

- Tau/choice agreement: 0.742
- Tau driver vs premise alignment: 0.633
- Pairwise driver vs premise alignment: n/a
- Tau OK rate: 1.000
- Pairwise OK rate: 0.000

## Stage A Weights
{
  "E": 0.292964315322273,
  "A": 0.2591207767668328,
  "S": 0.27447216122307033,
  "D": 0.1734427466878238
}

## Behavioral Attribution
{
  "occlude_equalize": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.0,
        "n": 1
      }
    },
    "normalized": {}
  },
  "occlude_drop": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.0,
        "n": 1
      },
      "A": {
        "delta_pA": 0.09999999999999998,
        "n": 1
      },
      "S": {
        "delta_pA": -0.1,
        "n": 1
      }
    },
    "normalized": {
      "E": 0.0,
      "A": 0.49999999999999994,
      "S": 0.5000000000000001
    }
  }
}
