# Attribute Confusion Heatmaps

## Inputs
- mismatch file: `outputs/final_same_order/reports/FINAL_ATTRIBUTE_LEVEL_RESULTS_MISMATCHES.csv`
- conditional file: `outputs/final_same_order/reports/FINAL_ATTRIBUTE_LEVEL_RESULTS_CONDITIONALS.csv`

## Normalization
- Each panel is row-normalized so every actor-stated attribute row sums to 1.
- Rows and columns follow canonical internal order `E, A, S, D`, displayed with theme-local labels.

## Total N Per Theme
- drugs: choice model N = 9600.0; judge N = 9600.0
- policy: choice model N = 9600.0; judge N = 9600.0
- software: choice model N = 9600.0; judge N = 9600.0

## Largest Off-Diagonal Cells
- Drugs / Choice model: Durability -> Efficacy = 0.27
- Drugs / Judge: Durability -> Efficacy = 0.54
- Policy / Choice model: Implementation Ease -> Effectiveness = 0.30
- Policy / Judge: Implementation Ease -> Effectiveness = 0.40
- Software / Choice model: Adoption Ease -> Capability = 0.29
- Software / Judge: Adoption Ease -> Capability = 0.34

## Validation
- Max row-sum deviation after normalization: 0.000000
- Qualitative expected-top-mismatch checks were disabled for this run; only normalization checks were enforced.
