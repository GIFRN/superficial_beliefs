# Judge Baselines Summary (openai_gpt5mini)

- Tau/choice agreement: 0.807
- Tau driver vs premise alignment: 0.642
- Pairwise driver vs premise alignment: n/a
- Tau OK rate: 1.000
- Pairwise OK rate: 0.000

## Stage A Weights
{
  "E": 0.30957019337362773,
  "A": 0.19240304772178363,
  "S": 0.24222720322614205,
  "D": 0.25579955567844664
}

## Behavioral Attribution
{
  "occlude_equalize": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.029999999999999992,
        "n": 10
      },
      "A": {
        "delta_pA": 0.05555555555555555,
        "n": 9
      },
      "S": {
        "delta_pA": 0.15,
        "n": 12
      },
      "D": {
        "delta_pA": 0.18333333333333335,
        "n": 12
      }
    },
    "normalized": {
      "E": 0.0716180371352785,
      "A": 0.13262599469496023,
      "S": 0.35809018567639256,
      "D": 0.43766578249336874
    }
  },
  "occlude_drop": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.06666666666666667,
        "n": 15
      },
      "A": {
        "delta_pA": 0.06,
        "n": 10
      },
      "S": {
        "delta_pA": 0.14666666666666667,
        "n": 10
      },
      "D": {
        "delta_pA": 0.0,
        "n": 7
      }
    },
    "normalized": {
      "E": 0.24390243902439027,
      "A": 0.21951219512195122,
      "S": 0.5365853658536586,
      "D": 0.0
    }
  }
}
