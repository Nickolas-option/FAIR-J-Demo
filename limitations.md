# Current limitations

- If a criterion is absent in a delete-variant, that missing case currently does not contribute to the corresponding `std_*`, `bias_*`, or `mad_*` block for that criterion.
- `ranking_consistency` currently skips any `(criterion, perturbation, seed)` case in which Kendall tau cannot be computed, for example because there are fewer than two shared examples or because SciPy returns `NaN` due to ties/degenerate rankings.
- `wilcoxon_signed_rank` is currently run with `zero_method="wilcox"`. If all paired differences are zero, or if the resulting input is not valid for the test, the current implementation returns `null` for the p-value.
- `paired_t_test` currently returns `null` when there are fewer than two paired observations or when SciPy returns `NaN` because the paired sample is degenerate.
- The real OpenRouter-backed judge path is now wired in `adapter.py`, but it has not yet been live-tested in this environment with a real API key.
- Real paraphrase generation now exists for `paraphrase_model` values through OpenRouter. Local placeholder paraphrases are no longer supported.
