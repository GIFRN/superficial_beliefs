# Occlusion Suite Results Summary

This file is the single paper-facing summary for the completed drugs occlusion suite.

## Scope

- model family: `GPT-5-mini`
- effort: `minimal`
- judge: `tau`

## Dataset Structure And Construction

The analyzed occlusion-suite dataset is `data/occlusion_suite/themes/drugs/test`.

It was constructed from the final same-order full test split, then expanded into matched intervention families.

- objective: `5-year overall patient outcome`
- total trials: `3600`
- matched base trials: `400`
- base configurations: `73`
- block composition: `B3` only

Each matched family contains `9` rows:

- `1` baseline `short_reason` row
- `4` `occlude_equalize` rows, one per attribute
- `4` `occlude_drop` rows, one per attribute

The intervention target counts are balanced across attributes, and the released suite does not include `occlude_swap`.

## Headline Results

- baseline-only Stage A accuracy: `0.917`
- full Stage A evaluation accuracy: `0.897`
- Stage B top-driver alignment: `0.837`
- Stage B weight alignment: `0.413`
- Stage B rank correlation: `0.800`
- judge choice agreement: `0.836`
- judge driver-vs-premise alignment: `0.675`

## Intervention Pattern

- baseline ranking: `E > S > D > A`
- `occlude_drop` ranking: `E > D > S > A`
- `occlude_equalize` ranking: `E > D > S > A`

Choice-flip rates:

- `occlude_drop`: `E 0.537`, `D 0.223`, `S 0.235`, `A 0.145`
- `occlude_equalize`: `E 0.517`, `D 0.195`, `S 0.217`, `A 0.113`

Premise-flip rates:

- `occlude_drop`: `E 0.655`, `D 0.315`, `S 0.360`, `A 0.307`
- `occlude_equalize`: `E 0.620`, `D 0.292`, `S 0.310`, `A 0.260`

## File

- `outputs/occlusion_suite_drugs_mini_min/reports/OCCLUSION_SUITE_RESULTS.csv`
