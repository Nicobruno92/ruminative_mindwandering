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

**Always submit from the `mw_classification_pipeline/` root** (the SLURM scripts rely on
`$SLURM_SUBMIT_DIR` being that directory).

```bash
ENV=~/miniforge3/envs/ML/bin/python
SD=within_subject_pipeline/spatial_decoding
cd mw_classification_pipeline
# 0. Build per-channel caches ONCE.
$ENV scripts/precompute_spatial_cache.py --config $SD/config.yaml
# 1. True per-channel group AUC — array over (dimension × channel) = 320 tasks.
sbatch $SD/run_true_slurm.sh        # writes true/channel-{CH}.csv shards
# 2. Permutation max-null (array over dimension × permutation block; size
#    5 * ceil(n_permutations / PERMS_PER_JOB); defaults → 125 tasks, --array=0-124).
sbatch $SD/run_perm_slurm.sh
# 3. Merge per dimension.
for C in on_vs_off_within_median valence_within_median selfother_within_median \
         time_within_median confidence_within_median; do
  $ENV $SD/merge_spatial_results.py \
    --results_dir results/MW_Classification/SpatialDecoding/WithinSubject/$C/all/rf
done
# 4. Combined 5-dimension paper panel.
$ENV scripts/generate_spatial_panel.py \
  --pipeline_dir results/MW_Classification/SpatialDecoding/WithinSubject

# 5. (optional) Per-marker SHAP for the FWER-significant electrode cluster only —
#    re-fits just those channels with SHAP enabled; see "SHAP for the significant
#    cluster" below.
$ENV scripts/extract_cluster_shap.py \
  --results_dir results/MW_Classification/SpatialDecoding/WithinSubject/$C/all/rf
```

## Outputs

```
results/MW_Classification/SpatialDecoding/WithinSubject/{contrast}/all/rf/
  true/true_per_channel_auc.csv     # channel, n_features, mean_auc (group), std_auc
  perms/perm-{P}.csv                # per-permutation per-channel group AUC (one shuffle)
  per_channel_metrics.csv           # channel, mean_auc, perm_p (FWER), sig, fwer_threshold
  topomap_auc.png / topomap_sig.png
  used_config.yaml
  shap/{channel}/...                # only if config.yaml's spatial_decoding.save_shap: true
  shap_cluster/{channel}/...        # only if extract_cluster_shap.py was run (step 5)
  shap_cluster/cluster_shap_summary.csv
combined/topomap_panel_auc.png
```

## Notes

- Contrast names are lowercase here (`on_vs_off_within_median`), matching the
  within-subject `config.yaml` `label_contrasts`.
- If you change `n_permutations`, update the perm array size accordingly.
- `feature_selection.method`, `feature_selection.k`, `n_runs`, classifier params — all
  from `config.yaml`.

## SHAP for the significant cluster

The searchlight never computes SHAP by default (`spatial_decoding.save_shap: false`)
— one explainer pass per electrode per subject per run per permutation would be far
too slow across the full 64-channel × `n_permutations` grid. Two ways to get
per-marker SHAP importance instead:

- **Post-hoc, cluster-only (recommended)**: after step 3 (merge) has produced
  `per_channel_metrics.csv`, run `scripts/extract_cluster_shap.py --results_dir ...`
  (step 5 above). It reads the FWER-`sig` channels and `used_config.yaml`, re-fits
  only those channels with SHAP on, and writes `shap_cluster/cluster_shap_summary.csv`
  (one row per channel × marker, `mean_abs_shap` averaged over all run×subject pickles).
- **Full future run**: set `spatial_decoding.save_shap: true` in `config.yaml` before
  submitting `run_true_slurm.sh` — every channel in the TRUE-mode array gets SHAP
  pickles under `shap/{channel}/`. Only combine this with a `channels:` subset
  (rather than `"all"`) unless you actually want SHAP on all 64 electrodes.
