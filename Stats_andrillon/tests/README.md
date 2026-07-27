# Stats_andrillon tests

The previous `test_permutation.py` and `test_clustering.py` were removed: they
imported `lmm_permutation` and `cluster_detection`, modules deleted in an
earlier refactor. Their functions (`fit_lmm_with_permutations`,
`find_clusters_andrillon`, `ClusterResult`, …) no longer exist under any name,
and the current implementation has a different API, so the files could only fail
at import — a false "test suite is broken" signal rather than real coverage.

The functionality they used to cover is now tested here:

| Current code | Test |
|---|---|
| `Statistics/cluster_test.py` — `find_clusters_from_pvalues`, `freedman_lane_permutation`, `get_channel_adjacency` | `Statistics/tests/test_cluster_permutation.py` |
| `Stats_andrillon/omnibus_test.py` — family-level omnibus + cross-dimension correction | `tests/test_omnibus_test.py` |
| Null-distribution retention (`return_null_distributions`) | `tests/test_omnibus_test.py`, plus an end-to-end pipeline smoke run |

Run them with:

```bash
conda activate eeg
python -m pytest Statistics/tests/test_cluster_permutation.py tests/test_omnibus_test.py -q
```

Not yet covered by unit tests (integration-tested via the SLURM pipeline only):
`run_marker_analysis`, family resolution from `selected_markers`/`feature_families`,
and derived-ratio markers. Worth adding if this pipeline sees further changes.
