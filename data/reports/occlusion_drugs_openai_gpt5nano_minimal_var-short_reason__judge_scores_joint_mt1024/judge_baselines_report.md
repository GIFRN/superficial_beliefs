# Judge Baselines Summary (openai_gpt5nano)

- Tau/choice agreement: 0.643
- Tau driver vs premise alignment: 0.445
- Pairwise driver vs premise alignment: n/a
- Tau OK rate: 0.965
- Pairwise OK rate: 0.000

## Stage A Weights
{
  "E": 0.33769631061496047,
  "A": 0.1821504604255244,
  "S": 0.20179003743765633,
  "D": 0.2783631915218588
}

## Behavioral Attribution
{
  "occlude_equalize": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.09,
        "n": 10
      },
      "A": {
        "delta_pA": 0.15555555555555556,
        "n": 9
      },
      "S": {
        "delta_pA": 0.0,
        "n": 12
      },
      "D": {
        "delta_pA": 0.25833333333333336,
        "n": 12
      }
    },
    "normalized": {
      "E": 0.17861080485115763,
      "A": 0.3087100330760749,
      "S": 0.0,
      "D": 0.5126791620727673
    }
  },
  "occlude_drop": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.002222222222222218,
        "n": 15
      },
      "A": {
        "delta_pA": -0.12000000000000002,
        "n": 10
      },
      "S": {
        "delta_pA": 0.023333333333333338,
        "n": 10
      },
      "D": {
        "delta_pA": -0.1142857142857143,
        "n": 7
      }
    },
    "normalized": {
      "E": 0.008552229688454472,
      "A": 0.4618204031765425,
      "S": 0.08979841172877215,
      "D": 0.4398289554062309
    }
  }
}
