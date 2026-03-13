# Canonical Balanced Split Verification

## Split Hygiene
- Train/test config overlap count: 0

## Size Summary
| split | original_rows | mirrored_rows | actual_total | expected_total_if_doubled |
| --- | --- | --- | --- | --- |
| train | 381 | 381 | 762 | 762 |
| test | 97 | 97 | 194 | 194 |

## Counts By Split
| split | n_trials | n_configs | n_mirror_pairs | same_order_rows | different_order_rows |
| --- | --- | --- | --- | --- | --- |
| train | 762 | 285 | 381 | 196 | 566 |
| test | 194 | 71 | 97 | 54 | 140 |

## Train Counts By Manipulation
| value | count |
| --- | --- |
| occlude_drop | 130 |
| occlude_equalize | 170 |
| short_reason | 462 |

## Test Counts By Manipulation
| value | count |
| --- | --- |
| occlude_drop | 36 |
| occlude_equalize | 42 |
| short_reason | 116 |

## Train Counts By Attribute Target
| value | count |
| --- | --- |
| <missing> | 462 |
| A | 76 |
| D | 74 |
| E | 78 |
| S | 72 |

## Test Counts By Attribute Target
| value | count |
| --- | --- |
| <missing> | 116 |
| A | 12 |
| D | 22 |
| E | 26 |
| S | 18 |

## Train Counts By is_mirrored
| value | count |
| --- | --- |
| False | 381 |
| True | 381 |

## Test Counts By is_mirrored
| value | count |
| --- | --- |
| False | 97 |
| True | 97 |

## Train Counts By labelA
| value | count |
| --- | --- |
| A | 381 |
| B | 381 |

## Test Counts By labelA
| value | count |
| --- | --- |
| A | 97 |
| B | 97 |

## Sample Original/Mirror Pairs
| split | mirror_pair_id | original_trial_id | mirror_trial_id | original_labelA | mirror_labelA | original_profile_A | original_profile_B | mirror_profile_A | mirror_profile_B | order_id_A | order_id_B | original_deltas | mirror_deltas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | mp::B3-0003::occlude_equalize::S::0::0::673820070::1::3 | T-00213 | T-00213__mirror | A | B | E:Low|A:Low|S:High|D:Medium | E:Low|A:Medium|S:Low|D:Low | E:Low|A:Medium|S:Low|D:Low | E:Low|A:Low|S:High|D:Medium | 1 | 3 | {"delta_A": -1, "delta_D": 1, "delta_E": 0, "delta_S": 0, "delta_base_A": -1, "delta_base_D": 1, "delta_base_E": 0, "delta_base_S": 2} | {"delta_A": 1, "delta_D": -1, "delta_E": 0, "delta_S": 0, "delta_base_A": 1, "delta_base_D": -1, "delta_base_E": 0, "delta_base_S": -2} |
| train | mp::B3-0004::short_reason::::0::2::1340301466::1::0 | T-00216 | T-00216__mirror | B | A | E:High|A:High|S:Medium|D:Low | E:Low|A:High|S:High|D:Low | E:Low|A:High|S:High|D:Low | E:High|A:High|S:Medium|D:Low | 1 | 0 | {"delta_A": 0, "delta_D": 0, "delta_E": 2, "delta_S": -1, "delta_base_A": 0, "delta_base_D": 0, "delta_base_E": 2, "delta_base_S": -1} | {"delta_A": 0, "delta_D": 0, "delta_E": -2, "delta_S": 1, "delta_base_A": 0, "delta_base_D": 0, "delta_base_E": -2, "delta_base_S": 1} |
| train | mp::B3-0005::short_reason::::0::0::940087482::1::0 | T-00217 | T-00217__mirror | A | B | E:Medium|A:Low|S:Medium|D:Medium | E:Medium|A:Medium|S:Medium|D:Low | E:Medium|A:Medium|S:Medium|D:Low | E:Medium|A:Low|S:Medium|D:Medium | 1 | 0 | {"delta_A": -1, "delta_D": 1, "delta_E": 0, "delta_S": 0, "delta_base_A": -1, "delta_base_D": 1, "delta_base_E": 0, "delta_base_S": 0} | {"delta_A": 1, "delta_D": -1, "delta_E": 0, "delta_S": 0, "delta_base_A": 1, "delta_base_D": -1, "delta_base_E": 0, "delta_base_S": 0} |
| train | mp::B3-0006::short_reason::::0::1::724690652::2::1 | T-00219 | T-00219__mirror | A | B | E:High|A:Medium|S:Low|D:Medium | E:High|A:High|S:Low|D:Low | E:High|A:High|S:Low|D:Low | E:High|A:Medium|S:Low|D:Medium | 2 | 1 | {"delta_A": -1, "delta_D": 1, "delta_E": 0, "delta_S": 0, "delta_base_A": -1, "delta_base_D": 1, "delta_base_E": 0, "delta_base_S": 0} | {"delta_A": 1, "delta_D": -1, "delta_E": 0, "delta_S": 0, "delta_base_A": 1, "delta_base_D": -1, "delta_base_E": 0, "delta_base_S": 0} |
| train | mp::B3-0008::short_reason::::0::0::1938821095::3::1 | T-00223 | T-00223__mirror | A | B | E:Low|A:High|S:Medium|D:Medium | E:High|A:Low|S:Low|D:Low | E:High|A:Low|S:Low|D:Low | E:Low|A:High|S:Medium|D:Medium | 3 | 1 | {"delta_A": 2, "delta_D": 1, "delta_E": -2, "delta_S": 1, "delta_base_A": 2, "delta_base_D": 1, "delta_base_E": -2, "delta_base_S": 1} | {"delta_A": -2, "delta_D": -1, "delta_E": 2, "delta_S": -1, "delta_base_A": -2, "delta_base_D": -1, "delta_base_E": 2, "delta_base_S": -1} |
| train | mp::B3-0009::occlude_equalize::A::0::2::3713401645::3::2 | T-00226 | T-00226__mirror | B | A | E:High|A:High|S:High|D:Low | E:Low|A:Medium|S:Medium|D:High | E:Low|A:Medium|S:Medium|D:High | E:High|A:High|S:High|D:Low | 3 | 2 | {"delta_A": 0, "delta_D": -2, "delta_E": 2, "delta_S": 1, "delta_base_A": 1, "delta_base_D": -2, "delta_base_E": 2, "delta_base_S": 1} | {"delta_A": 0, "delta_D": 2, "delta_E": -2, "delta_S": -1, "delta_base_A": -1, "delta_base_D": 2, "delta_base_E": -2, "delta_base_S": -1} |
| test | mp::B3-0001::occlude_equalize::S::0::2::1509427542::1::1 | T-00210 | T-00210__mirror | B | A | E:High|A:High|S:Medium|D:Medium | E:Low|A:High|S:Medium|D:High | E:Low|A:High|S:Medium|D:High | E:High|A:High|S:Medium|D:Medium | 1 | 1 | {"delta_A": 0, "delta_D": -1, "delta_E": 2, "delta_S": 0, "delta_base_A": 0, "delta_base_D": -1, "delta_base_E": 2, "delta_base_S": 0} | {"delta_A": 0, "delta_D": 1, "delta_E": -2, "delta_S": 0, "delta_base_A": 0, "delta_base_D": 1, "delta_base_E": -2, "delta_base_S": 0} |
| test | mp::B3-0002::occlude_drop::D::0::1::681664089::3::2 | T-00212 | T-00212__mirror | B | A | E:Medium|A:Low|S:Low|D:High | E:Low|A:Medium|S:High|D:Medium | E:Low|A:Medium|S:High|D:Medium | E:Medium|A:Low|S:Low|D:High | 3 | 2 | {"delta_A": -1, "delta_D": 0, "delta_E": 1, "delta_S": -2, "delta_base_A": -1, "delta_base_D": 1, "delta_base_E": 1, "delta_base_S": -2} | {"delta_A": 1, "delta_D": 0, "delta_E": -1, "delta_S": 2, "delta_base_A": 1, "delta_base_D": -1, "delta_base_E": -1, "delta_base_S": 2} |
| test | mp::B3-0002::short_reason::::0::2::4158869863::2::0 | T-00211 | T-00211__mirror | A | B | E:Low|A:Medium|S:High|D:Medium | E:Medium|A:Low|S:Low|D:High | E:Medium|A:Low|S:Low|D:High | E:Low|A:Medium|S:High|D:Medium | 2 | 0 | {"delta_A": 1, "delta_D": -1, "delta_E": -1, "delta_S": 2, "delta_base_A": 1, "delta_base_D": -1, "delta_base_E": -1, "delta_base_S": 2} | {"delta_A": -1, "delta_D": 1, "delta_E": 1, "delta_S": -2, "delta_base_A": -1, "delta_base_D": 1, "delta_base_E": 1, "delta_base_S": -2} |
| test | mp::B3-0011::short_reason::::0::0::2564876764::3::1 | T-00230 | T-00230__mirror | B | A | E:Medium|A:High|S:Low|D:Low | E:Low|A:Medium|S:High|D:High | E:Low|A:Medium|S:High|D:High | E:Medium|A:High|S:Low|D:Low | 3 | 1 | {"delta_A": 1, "delta_D": -2, "delta_E": 1, "delta_S": -2, "delta_base_A": 1, "delta_base_D": -2, "delta_base_E": 1, "delta_base_S": -2} | {"delta_A": -1, "delta_D": 2, "delta_E": -1, "delta_S": 2, "delta_base_A": -1, "delta_base_D": 2, "delta_base_E": -1, "delta_base_S": 2} |
| test | mp::B3-0011::short_reason::::0::0::3416886853::1::0 | T-00229 | T-00229__mirror | A | B | E:Low|A:Medium|S:High|D:High | E:Medium|A:High|S:Low|D:Low | E:Medium|A:High|S:Low|D:Low | E:Low|A:Medium|S:High|D:High | 1 | 0 | {"delta_A": -1, "delta_D": 2, "delta_E": -1, "delta_S": 2, "delta_base_A": -1, "delta_base_D": 2, "delta_base_E": -1, "delta_base_S": 2} | {"delta_A": 1, "delta_D": -2, "delta_E": 1, "delta_S": -2, "delta_base_A": 1, "delta_base_D": -2, "delta_base_E": 1, "delta_base_S": -2} |
| test | mp::B3-0014::short_reason::::0::0::2381910663::2::3 | T-00236 | T-00236__mirror | B | A | E:High|A:Low|S:High|D:High | E:Medium|A:High|S:High|D:Low | E:Medium|A:High|S:High|D:Low | E:High|A:Low|S:High|D:High | 2 | 3 | {"delta_A": -2, "delta_D": 2, "delta_E": 1, "delta_S": 0, "delta_base_A": -2, "delta_base_D": 2, "delta_base_E": 1, "delta_base_S": 0} | {"delta_A": 2, "delta_D": -2, "delta_E": -1, "delta_S": 0, "delta_base_A": 2, "delta_base_D": -2, "delta_base_E": -1, "delta_base_S": 0} |

## Issues
- Train issues: 0
- Test issues: 0