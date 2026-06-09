# LOSO Spatial Decoding (per-electrode searchlight)

Trains one LOSO classifier **per electrode** (each using only that channel's markers,
filtered by the active feature family + mRMR `k`) and maps decodability across the
scalp. Significance uses a **max-statistic permutation test (FWER)** across the 64
electrodes — the same logic as the Section-2 CBPT correction.

Design: `docs/superpowers/specs/2026-06-08-spatial-decoding-design.md` (§6, §8).
Shared core: `utils/spatial_decoding_utils.py`. All parameters live in `config.yaml`.

## How it works

- **TRUE statistic** — per channel, the `n_runs`-averaged AUC (`run_*_decoding.py`,
  no `--perm_idx`).
- **NULL of the max** — for each of `permutation_runs` draws, ONE within-subject label
  shuffle is scored across ALL channels (`--perm_idx P`); the per-draw
  max-over-channels AUC forms the family-wise null.
- **Inference** — an electrode is significant iff its true AUC exceeds the `(1-alpha)`
  quantile of the max-null; per-channel FWER p = `(1+#{max_null ≥ true})/(1+N)`.

## Run order

**Always submit from the `mw_classification_pipeline/` root** (the SLURM scripts rely on
`$SLURM_SUBMIT_DIR` being that directory).

```bash
ENV=~/miniforge3/envs/ML/bin/python
SD=loso_pipeline/spatial_decoding
cd mw_classification_pipeline

# 0. Build the per-channel caches ONCE (avoids array races; ~minutes per contrast).
$ENV scripts/precompute_spatial_cache.py --config $SD/config.yaml

# 1. True per-channel AUC — array over (dimension × channel) = 5×64 = 320 tasks.
sbatch $SD/run_true_slurm.sh        # writes true/channel-{CH}.csv shards

# 2. Permutation max-null — array over (dimension × permutation block).
#    Array size MUST be 5 * ceil(permutation_runs / PERMS_PER_JOB). Defaults:
#    permutation_runs=500, PERMS_PER_JOB=20 → 125 tasks (#SBATCH --array=0-124).
sbatch $SD/run_perm_slurm.sh

# 3. Merge per dimension → per_channel_metrics.csv + topomap_auc.png + topomap_sig.png
for C in ON_vs_OFF_within_median valence_within_median selfother_within_median \
         time_within_median confidence_within_median; do
  $ENV $SD/merge_spatial_results.py \
    --results_dir results/MW_Classification/SpatialDecoding/LOSO/$C/all/rf
done

# 4. Combined 5-dimension paper panel
$ENV scripts/generate_spatial_panel.py \
  --pipeline_dir results/MW_Classification/SpatialDecoding/LOSO
```

## Outputs

```
results/MW_Classification/SpatialDecoding/LOSO/{contrast}/all/rf/
  true/true_per_channel_auc.csv     # channel, n_features, mean_auc, std_auc
  perms/perm-{P}.csv                # per-permutation per-channel AUC (one shuffle)
  per_channel_metrics.csv           # channel, mean_auc, perm_p (FWER), sig, fwer_threshold
  topomap_auc.png / topomap_sig.png
  used_config.yaml
combined/topomap_panel_auc.png
```

## Notes

- If you change `permutation_runs` in `config.yaml`, update the perm array size
  (`#SBATCH --array`) accordingly: `5 * ceil(permutation_runs / PERMS_PER_JOB)`.
- A single-channel TRUE shard is also supported (`--channel CH`) for a finer
  (dimension × channel) true array; `merge` reads `true/*.csv` either way.
- `feature_selection_method`, `k`, `n_runs`, classifier params — all from `config.yaml`.
