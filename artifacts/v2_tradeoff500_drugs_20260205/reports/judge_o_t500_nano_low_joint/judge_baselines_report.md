# Judge Baselines Summary (openai_gpt5nano)

- Tau/choice agreement: 0.683
- Tau driver vs premise alignment: 0.517
- Pairwise driver vs premise alignment: n/a
- Tau OK rate: 0.995
- Pairwise OK rate: 0.000

## Stage A Weights
{
  "E": 0.23993316688157487,
  "A": 0.2694979074673538,
  "S": 0.27957378478041095,
  "D": 0.2109951408706604
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
        "delta_pA": -0.6,
        "n": 1
      },
      "S": {
        "delta_pA": -0.19999999999999998,
        "n": 1
      }
    },
    "normalized": {
      "E": 0.0,
      "A": 0.75,
      "S": 0.25
    }
  }
}
