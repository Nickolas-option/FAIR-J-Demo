# FAIR-J Audit: Issues Found in Smoke Test Results

## Context

This document records statistical and methodological issues discovered during the audit of the smoke test run produced by Daniil's commit `53ceda5` ("Add bedrock model support, fix small bugs, add report for 5 paraphrases, 1 seed").

The audit was performed against:
- Run directory: `runs/2026-05-13T21-41-40__multi_6models__fed_shikib_dataset__0-2/`
- 6 Bedrock judge models: `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4`, `claude-opus-4-5`, `claude-opus-4-7`, `gpt-oss-120b`
- Dataset: `data/fed/fed_shikib_dataset.jsonl` (375 examples total)
- Rubric: `data/fed/fed_fairj_rubric.json` (6 criteria, scale 0–2)
- Paraphrases per criterion: 5
- Seeds: 1

## Smoke Test Configuration vs. Actual Data

| Parameter | Configured | Actually used in run |
|---|---|---|
| Dataset size | 375 rows | **20 unique examples** (limited via `--limit-examples`) |
| Paraphrase variants per criterion | 5 | 5 |
| Delete variants | 1 per criterion | 6 (one per criterion) |
| Seeds | 1 | 1 |
| Criteria | 6 | 6 |
| Total perturbations | — | 37 (1 baseline + 30 paraphrases + 6 deletions) |
| Score log rows per model | — | 4320 |
| Call log rows per model | — | 740 |

The smoke test successfully proved the Bedrock integration works end-to-end. However, the audit revealed methodological issues in the FAIR-J core analysis layer (not in Daniil's Bedrock code) that cause reported p-values, MDE, and significance claims to be incorrect.

---

## Issue 1: Pseudo-replication in statistical tests (CRITICAL)

### Symptom

The `n_pairs` field in every `stat_tests.paraphrases_vs_baseline` block reports `600`, suggesting 600 independent paired observations. In reality, those 600 entries come from **20 unique examples repeated across 30 paraphrase perturbations** (5 paraphrases × 6 criteria). The 600 pairs are not independent — they are clustered by example.

Example: for criterion `fed_diverse`, the score log contains:
- 20 baseline rows (one per example)
- 600 paraphrase rows = 20 examples × 30 paraphrase perturbations
  - including 5 paraphrases that target `fed_diverse` directly
  - plus 25 paraphrases that target other criteria but still record a `fed_diverse` score

The paired t-test, Wilcoxon signed-rank test, and permutation test in `core_output.json` all treat these 600 as independent.

### Root cause in code

- File: `src/fair_j/score_grouping.py`
- Function: `collect_grouped_comparisons()` (lines 53–89)
- Behavior: pairs each non-baseline score row with its matching baseline by `(example_id, criterion_id, seed)`, appends the difference to a flat `differences` list. No deduplication or aggregation per example.

- File: `src/fair_j/stats_utils.py`
- Function: `build_stat_test_block()` (line 94)
- Behavior: passes `len(differences)` as `n_pairs` directly into stat tests and MDE calculation.

### Impact on reported numbers

- All `p_value` fields in `paired_permutation_test`, `wilcoxon_signed_rank`, `paired_t_test` are **inflated in significance** (too small).
- Example: Claude Haiku 4.5 reports `paired_t_test.p_value = 7.8e-11` for `__total__` paraphrases vs baseline. With correct clustering by `example_id` (effective n ≈ 20 instead of 600), the actual p-value would be orders of magnitude larger.
- The `mde_mean_difference` and `mde_effect_size_dz` values are also under-estimated by roughly a factor of √30 ≈ 5.5.

### Suggested fix

Two viable approaches:

**Option A (minimal change, recommended):** Aggregate differences per example before running the test. For each `example_id`, compute the mean difference across all perturbations targeting that observation. Then run the stat test on N = number of unique examples. This makes `n_pairs` equal to the true effective sample size.

**Option B (more rigorous):** Implement clustered bootstrap resampling grouped by `example_id`. This pattern is already used in the upstream FAIR-J project (see `finetune.md`, "Source-cluster bootstrap 95% CI"), but is not wired into the `stat_tests` block in `evaluate_judge.py`.

Recommendation: implement Option A as the immediate fix, optionally add Option B as a higher-quality alternative behind a flag.

---

## Issue 2: Same-criterion and cross-criterion paraphrase effects are merged (HIGH)

### Symptom

When evaluating stability of criterion `fed_diverse`, the `paraphrases_vs_baseline` differences pool together:
- `paraphrase__fed_diverse__*` — paraphrasing the rubric text of `fed_diverse` itself (direct effect)
- `paraphrase__fed_fluent__*`, `paraphrase__fed_relevant__*`, etc. — paraphrasing other criteria (indirect / spillover effect)

These are conceptually different signals and should be reported separately.

### Root cause in code

- File: `src/fair_j/score_grouping.py`
- Function: `perturbation_group_name()` (lines 183–188)
- Behavior: only checks the prefix (`paraphrase__` / `delete__`), ignoring which criterion was perturbed.
- Function: `collect_grouped_comparisons()` then dumps both same-criterion and cross-criterion paraphrases into one `"paraphrases"` bucket.

### Why this matters

A judge model that is unstable under paraphrasing of *unrelated* criteria is a much worse signal than one that drifts only when its *own* criterion is paraphrased. Currently both are indistinguishable in `core_output.json` and in the HTML report.

### Suggested fix

In `collect_grouped_comparisons()`, parse the perturbation name to extract the target criterion ID. Split `"paraphrases"` into two sub-buckets:

- `"paraphrases_same_criterion"` — target criterion equals `row.criterion_id`
- `"paraphrases_cross_criterion"` — target criterion differs from `row.criterion_id`

Emit separate `differences`, `bias`, `mad`, and `stat_tests` blocks for each. Update `report_html.py` accordingly.

---

## Issue 3: `__total__` Kendall tau is not the average of per-criterion Kendall tau (CONFUSION)

### Symptom

In `core_output.json`, `ranking_consistency.__total__.kendall_paraphrases_mean` does **not** equal the simple average of `kendall_paraphrases_mean` across criteria. Concrete values from the audit:

| Model | `__total__` Kendall | Average of per-criterion Kendall |
|---|---|---|
| `claude-opus-4` | 0.8429 | 0.9255 |
| `claude-sonnet-4-6` | 0.8089 | 0.8645 |
| `claude-opus-4-7` | 0.9387 | 0.8906 |
| `claude-haiku-4-5` | 0.8521 | 0.8305 |
| `claude-opus-4-5` | 0.8778 | 0.8857 |
| `gpt-oss-120b` | 0.5793 | 0.6425 |

The `comparison.html` summary card labels its value as `Avg Kendall paraphrases`, which corresponds to the average-of-criteria figure, not to `__total__`.

### Root cause in code

- File: `src/fair_j/evaluate_judge.py`
- Function: `build_total_score_rows()` (lines 184–215)
- Behavior: creates synthetic score rows with `criterion_id = "__total__"` whose value is the **sum of per-criterion baseline-aligned scores** for each `(example_id, perturbation, seed)`. Kendall tau is then computed on these summed scores.

This is a valid quantity (ranking stability of the total score), but it is not the same as the average per-criterion Kendall tau, and the report does not currently make this distinction explicit.

### Suggested fix

- Add a separate field `avg_kendall_paraphrases_across_criteria` computed as the plain mean of per-criterion `kendall_paraphrases_mean`.
- Keep `__total__.kendall_paraphrases_mean` as is, but document it clearly as "Kendall tau over summed scores across criteria".
- In `report_html.py`, label both quantities distinctly so readers know which one they are looking at.

---

## Issue 4: `n_unique_examples` is not reported (TRANSPARENCY)

### Symptom

`core_output.json` reports `n_pairs` in every stat test block but never reports the number of unique `example_id` values that produced those pairs. A reader cannot assess effective sample size without going back to the raw `score_log.jsonl`.

### Root cause in code

- File: `src/fair_j/stats_utils.py`
- Function: `build_stat_test_block()` (lines 83–116)
- Behavior: receives only flat lists of `baseline`, `perturbation`, and `differences` values — example identity is already lost by the time the test block is built.

### Suggested fix

Extend the data structure returned by `collect_grouped_comparisons()` so that each comparison group also tracks `example_ids: list[str]`. Then `build_stat_test_block()` can add:

- `n_unique_examples`
- `n_perturbations`
- `pairs_per_example` (sanity check)

This is essential transparency once Issue 1 is fixed — readers need to see both numbers to interpret the test correctly.

---

## Issue 5: MDE is computed on the inflated `n_pairs` (CRITICAL — follows from Issue 1)

### Symptom

`mde_mean_difference` and `mde_effect_size_dz` are computed under the assumption of 600 independent observations. With effective n ≈ 20, the true minimum detectable effect is ≈ √30 ≈ 5.5× larger than what is reported.

### Root cause in code

- File: `src/fair_j/stats_utils.py`
- Function: `build_stat_test_block()` (line 89): `mde_effect_size_dz = paired_mde_effect_size_dz(len(differences))`
- Function: `paired_mde_effect_size_dz()` (lines 202–216): uses `n_pairs` directly in `TTestPower.solve_power()`.

### Suggested fix

Resolved automatically when Issue 1 is fixed (Option A: aggregate per example; `len(differences)` then equals the true number of unique examples). If Option B (clustered bootstrap) is chosen instead, MDE should be computed against effective number of clusters.

---

## Non-issues observed during audit (not problems, just context)

### Ties from the 0–2 scale

Baseline score distributions on a 3-point scale (0/1/2) with only 20 examples produce 40–90% tied pairs for most criteria. This is a property of the **dataset choice for the smoke test**, not a code bug. Kendall tau-b correctly accounts for ties via the formula `τ_b = (C − D) / √((C+D+T_x)(C+D+T_y))` (visible in `score_grouping.py::count_pair_relationships()`). However, with this many ties, the resulting tau values are highly sensitive to single examples flipping. This will resolve naturally with a larger dataset or a finer scale.

### `seeds=1`

With a single seed, `std_seed_default` is `null` everywhere. This is expected for a smoke test, but means no estimate of inherent model stochasticity is available. Future runs should use `seeds ≥ 3` for any quantitative comparisons between models.

### Bedrock integration code quality

The Bedrock client code added by Daniil (`model_clients.py`, `openrouter_judge.py::call_bedrock_judge`, `perturbations.py::create_paraphrase_completion`) is clean, has appropriate retry logic with backoff for `ThrottlingException` / `TooManyRequestsException`, handles markdown-fenced JSON output, and correctly skips API key validation for `bedrock/*` model prefixes. No issues found here.

---

## Severity summary

| # | Issue | Severity | Blocks publishing results? |
|---|---|---|---|
| 1 | Pseudo-replication in stat tests (n_pairs inflated 30x) | Critical | Yes |
| 2 | Same/cross-criterion paraphrase effects merged | High | No (but obscures interpretation) |
| 3 | `__total__` Kendall ≠ avg-of-criteria Kendall | Medium (clarity) | No |
| 4 | `n_unique_examples` not reported | Medium (transparency) | No |
| 5 | MDE computed on inflated `n_pairs` | Critical (follows from #1) | Yes |

Issues 1 and 5 must be fixed before any p-value or MDE figure from the current `core_output.json` files can be cited in writing. Issues 2–4 are quality-of-reporting improvements that should accompany the fix.

---

## Qualitative findings that survive the audit

Even with the methodological issues above, two qualitative claims are robust to the small sample size and the inflated `n_pairs`:

1. **GPT-OSS 120B is dramatically less stable than the Claude models on this rubric.** Its `mad_paraphrases = 1.0117` on a 0–2 scale means the average absolute shift in score under paraphrasing exceeds half the scale span — visible directly in the raw scores, no test required.

2. **The Claude family has materially lower paraphrase MAD than GPT-OSS.** The gap (≈ 0.25–0.45 for Claude vs. 1.01 for GPT-OSS) is large enough to be obvious without statistical testing.

Fine-grained ordering within the Claude family (`opus-4-7` vs. `opus-4-5` vs. `opus-4` vs. `sonnet-4-6` vs. `haiku-4-5`) is **not** supported by these 20 examples and should not be reported as a finding until the dataset is scaled up.

---

## Next actions

This document records what was found. Awaiting direction on:

1. Which fixes to implement first (suggested order: Issue 1 → Issue 5 (auto) → Issue 4 → Issue 2 → Issue 3).
2. Whether to fix in this fork (`FAIR-J-Demo-Daniil/`) or upstream in the main `FAIR-J/` repo.
3. Whether to re-run the smoke test after fixes to confirm the corrected p-values and MDE.
