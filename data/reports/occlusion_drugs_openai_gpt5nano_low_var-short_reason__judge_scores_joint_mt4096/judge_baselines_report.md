# Judge Baselines Summary (openai_gpt5nano)

- Tau/choice agreement: 0.770
- Tau driver vs premise alignment: 0.574
- Pairwise driver vs premise alignment: n/a
- Tau OK rate: 0.987
- Pairwise OK rate: 0.000

## Stage A Weights
{
  "E": 0.24025147749899614,
  "A": 0.2715096674939006,
  "S": 0.27356603199250334,
  "D": 0.21467282301459992
}

## Behavioral Attribution
{
  "occlude_equalize": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.04,
        "n": 10
      },
      "A": {
        "delta_pA": -0.022222222222222223,
        "n": 9
      },
      "S": {
        "delta_pA": 0.13333333333333333,
        "n": 12
      },
      "D": {
        "delta_pA": 0.14722222222222223,
        "n": 12
      }
    },
    "normalized": {
      "E": 0.1166936790923825,
      "A": 0.06482982171799027,
      "S": 0.3889789303079416,
      "D": 0.42949756888168555
    }
  },
  "occlude_drop": {
    "by_attribute": {
      "E": {
        "delta_pA": 0.05111111111111111,
        "n": 15
      },
      "A": {
        "delta_pA": 0.04666666666666667,
        "n": 10
      },
      "S": {
        "delta_pA": 0.09333333333333335,
        "n": 10
      },
      "D": {
        "delta_pA": -0.028571428571428574,
        "n": 7
      }
    },
    "normalized": {
      "E": 0.2326589595375722,
      "A": 0.21242774566473988,
      "S": 0.4248554913294798,
      "D": 0.13005780346820808
    }
  }
}
