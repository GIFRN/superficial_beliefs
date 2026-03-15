# Method C Results Summary

This file is the single paper-facing Method C summary for the currently completed run.

## Scope

Method C is currently complete and analyzed for one condition:

- model family: `GPT-5-mini`
- effort: `minimal`
- judge: `tau`

The Method C suite is best interpreted as an intervention-based validation of the revealed-preference account. Starting from matched baseline prompts, it selectively neutralizes or removes evidence for one attribute while keeping the rest of the item family fixed.

## Headline Results

The baseline-only `short_reason` rows recover a clear preference ranking:

- weight order: `E > D > S > A`
- baseline-only Stage A accuracy: `0.873`

The full Method C fit remains strong:

- Stage A evaluation accuracy: `0.880`
- Stage B top-driver alignment: `0.877`
- Stage B weight alignment: `0.379`
- judge choice agreement: `0.802`
- judge driver-vs-premise alignment: `0.654`

## Intervention Pattern

The main Method C result is that the intervention ranking exactly matches the baseline revealed-preference ranking for both manipulation types:

- `occlude_drop`: `E > D > S > A`
- `occlude_equalize`: `E > D > S > A`

Occluding a more important attribute causes larger behavioral disruption.

Directional choice effects (`delta_favored_mean`) for `occlude_drop`:

- `E`: `-0.458`
- `D`: `-0.354`
- `S`: `-0.306`
- `A`: `-0.232`

Directional choice effects for `occlude_equalize`:

- `E`: `-0.466`
- `D`: `-0.355`
- `S`: `-0.310`
- `A`: `-0.238`

Choice-flip rates follow the same order:

- `occlude_drop`: `E 0.386`, `D 0.280`, `S 0.277`, `A 0.210`
- `occlude_equalize`: `E 0.366`, `D 0.274`, `S 0.266`, `A 0.204`

Premise-flip rates also track the same ranking:

- `occlude_drop`: `E 0.559`, `D 0.376`, `S 0.356`, `A 0.296`
- `occlude_equalize`: `E 0.528`, `D 0.371`, `S 0.323`, `A 0.276`

## Interpretation

Method C therefore supports a compact claim:

- the latent preference ranking inferred from baseline choices is `E > D > S > A`
- targeted interventions recover exactly the same ranking
- choice changes and premise changes move together in the predicted direction
- `occlude_drop` and `occlude_equalize` behave very similarly

That is strong evidence that the revealed-preference structure is not merely descriptive: it tracks how behavior and stated reasons shift when prompt-visible evidence for a specific attribute is experimentally manipulated.

## File

The companion numerical table is:

- [METHODC_RESULTS_20260315.csv](/vol/bitbucket/gif22/superficial_beliefs/artifacts/v4_matchedsuite_drugs_20260208/reports/METHODC_RESULTS_20260315.csv)
