# Within-Subject Spatial Decoding (per-electrode searchlight)

Trains one within-subject classifier **per electrode** (each using only that channel's
markers, filtered by the active feature family + mRMR `k`) and maps decodability across
the scalp. The per-channel statistic is the **group-mean AUC across subjects**.
Significance uses a **max-statistic permutation test (FWER)** across the 64 electrodes —
the same logic as the Section-2 CBPT correction.

Design: `docs/superpowers/specs/2026-06-08-spatial-decoding-design.md` (§6, §8).
Shared core: `utils/spatial_decoding_utils.py`. All parameters live in `config.yaml`.

## How it works

- **TRUE statistic** — per channel, the group-mean (across subjects) of the
  `n_runs`-averaged within-subject AUC (`run_within_spatial_decoding.py`, no `--perm_idx`).
- **NULL of the max** — for each of `n_permutations` draws, ONE within-subject label
  shuffle is scored across ALL channels (`--perm_idx P`); the per-draw
  max-over-channels group AUC forms the family-wise null.
- **Inference** — an electrode is significant iff its true group AUC exceeds the
  `(1-alpha)` quantile of the max-null; per-channel FWER p = `(1+#{max_null ≥ true})/(1+N)`.

## Run order

```bash
ENV=~/miniforge3/envs/ML/bin/python
HERE=within_subject_pipeline/spatial_decoding

cd mw_classification_pipeline
# 0. Build per-channel caches ONCE.
$ENV scripts/precompute_spatial_cache.py --config $HERE/config.yaml
# 1. True per-channel group AUC (one task per dimension).
sbatch $HERE/run_true_slurm.sh
# 2. Permutation max-null (array over dimension × permutation block; size
#    5 * ceil(n_permutations / PERMS_PER_JOB); defaults → 125 tasks, --array=0-124).
sbatch $HERE/run_perm_slurm.sh
# 3. Merge per dimension.
for C in on_vs_off_within_median valence_within_median selfother_within_median \
         time_within_median confidence_within_median; do
  $ENV $HERE/merge_spatial_results.py \
    --results_dir results/MW_Classification/SpatialDecoding/WithinSubject/$C/all/rf
done
# 4. Combined 5-dimension paper panel.
$ENV scripts/generate_spatial_panel.py \
  --pipeline_dir results/MW_Classification/SpatialDecoding/WithinSubject
```

## Outputs

```
results/MW_Classification/SpatialDecoding/WithinSubject/{contrast}/all/rf/
  true/true_per_channel_auc.csv     # channel, n_features, mean_auc (group), std_auc
  perms/perm-{P}.csv                # per-permutation per-channel group AUC (one shuffle)
  per_channel_metrics.csv           # channel, mean_auc, perm_p (FWER), sig, fwer_threshold
  topomap_auc.png / topomap_sig.png
  used_config.yaml
combined/topomap_panel_auc.png
```

## Notes

- Contrast names are lowercase here (`on_vs_off_within_median`), matching the
  within-subject `config.yaml` `label_contrasts`.
- If you change `n_permutations`, update the perm array size accordingly.
- `feature_selection.method`, `feature_selection.k`, `n_runs`, classifier params — all
  from `config.yaml`.
