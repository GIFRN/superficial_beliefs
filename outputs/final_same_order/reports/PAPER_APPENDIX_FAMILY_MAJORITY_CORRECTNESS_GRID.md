# Paper Appendix Family Majority Correctness Grid

- Each row is one substantive theme x model-family x effort condition.
- Majority correctness is computed over the 12 responses for each base family: 4 prompt variants x 3 samples.
- For driver metrics, correctness is evaluated draw-by-draw against that draw's revealed driver rather than a single fixed family-level target.
- The markdown table shows the share of complete families whose response stream is majority-correct.

| theme | family | effort | actor_choice_family_majority_correct_rate | self_report_driver_family_majority_correct_rate | score_judge_choice_family_majority_correct_rate | score_judge_driver_family_majority_correct_rate |
| --- | --- | --- | --- | --- | --- | --- |
| Drugs | GPT-5-mini | low | 0.810 [0.735, 0.880] | 0.730 [0.640, 0.810] | 0.860 [0.790, 0.920] | 0.760 [0.670, 0.840] |
| Drugs | GPT-5-mini | minimal | 0.890 [0.830, 0.945] | 0.660 [0.580, 0.750] | 0.890 [0.820, 0.950] | 0.790 [0.705, 0.870] |
| Drugs | GPT-5-nano | low | 0.850 [0.780, 0.910] | 0.640 [0.550, 0.730] | 0.800 [0.720, 0.875] | 0.580 [0.490, 0.675] |
| Drugs | GPT-5-nano | minimal | 0.940 [0.890, 0.990] | 0.410 [0.320, 0.500] | 0.760 [0.680, 0.850] | 0.360 [0.270, 0.460] |
| Drugs | Ministral-3-14B | low | 0.740 [0.660, 0.820] | 0.320 [0.240, 0.410] | 0.810 [0.740, 0.885] | 0.650 [0.550, 0.750] |
| Drugs | Ministral-3-14B | minimal | 0.910 [0.850, 0.960] | 0.490 [0.400, 0.590] | 0.660 [0.570, 0.760] | 0.540 [0.450, 0.650] |
| Drugs | Qwen3-14B | low | 0.800 [0.720, 0.875] | 0.590 [0.485, 0.680] | 0.670 [0.575, 0.760] | 0.550 [0.450, 0.650] |
| Drugs | Qwen3-14B | minimal | 0.750 [0.660, 0.830] | 0.380 [0.290, 0.480] | 0.720 [0.630, 0.800] | 0.430 [0.325, 0.535] |
| Policy | GPT-5-mini | low | 0.920 [0.860, 0.970] | 0.880 [0.820, 0.940] | 0.780 [0.690, 0.860] | 0.850 [0.780, 0.910] |
| Policy | GPT-5-mini | minimal | 0.890 [0.830, 0.950] | 0.800 [0.720, 0.880] | 0.810 [0.740, 0.880] | 0.830 [0.750, 0.900] |
| Policy | GPT-5-nano | low | 0.920 [0.860, 0.970] | 0.760 [0.670, 0.850] | 0.770 [0.680, 0.840] | 0.700 [0.610, 0.780] |
| Policy | GPT-5-nano | minimal | 0.930 [0.880, 0.975] | 0.490 [0.390, 0.590] | 0.750 [0.670, 0.825] | 0.390 [0.300, 0.500] |
| Policy | Ministral-3-14B | low | 0.730 [0.640, 0.830] | 0.390 [0.290, 0.485] | 0.800 [0.720, 0.880] | 0.680 [0.580, 0.770] |
| Policy | Ministral-3-14B | minimal | 0.950 [0.910, 0.990] | 0.700 [0.605, 0.790] | 0.720 [0.630, 0.800] | 0.590 [0.500, 0.680] |
| Policy | Qwen3-14B | low | 0.900 [0.830, 0.950] | 0.800 [0.715, 0.875] | 0.800 [0.720, 0.880] | 0.720 [0.630, 0.820] |
| Policy | Qwen3-14B | minimal | 0.920 [0.860, 0.970] | 0.630 [0.530, 0.720] | 0.640 [0.540, 0.730] | 0.320 [0.240, 0.410] |
| Software | GPT-5-mini | low | 0.920 [0.860, 0.970] | 0.790 [0.710, 0.880] | 0.540 [0.440, 0.630] | 0.540 [0.420, 0.635] |
| Software | GPT-5-mini | minimal | 0.910 [0.860, 0.960] | 0.630 [0.530, 0.720] | 0.840 [0.760, 0.900] | 0.650 [0.560, 0.740] |
| Software | GPT-5-nano | low | 0.810 [0.740, 0.880] | 0.550 [0.455, 0.645] | 0.810 [0.740, 0.880] | 0.550 [0.450, 0.650] |
| Software | GPT-5-nano | minimal | 0.960 [0.920, 0.990] | 0.200 [0.120, 0.270] | 0.350 [0.260, 0.450] | 0.070 [0.030, 0.120] |
| Software | Ministral-3-14B | low | 0.740 [0.650, 0.820] | 0.330 [0.240, 0.420] | 0.790 [0.710, 0.860] | 0.480 [0.370, 0.575] |
| Software | Ministral-3-14B | minimal | 0.920 [0.870, 0.970] | 0.410 [0.310, 0.510] | 0.650 [0.550, 0.740] | 0.550 [0.450, 0.660] |
| Software | Qwen3-14B | low | 0.890 [0.830, 0.950] | 0.650 [0.550, 0.740] | 0.780 [0.710, 0.870] | 0.690 [0.610, 0.770] |
| Software | Qwen3-14B | minimal | 0.880 [0.810, 0.940] | 0.500 [0.395, 0.610] | 0.710 [0.625, 0.800] | 0.270 [0.190, 0.370] |
