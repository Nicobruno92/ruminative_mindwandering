# GLMM for Objective Behavioural Markers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the misspecified Gaussian LMM on objective behavioural markers with a binomial/Gamma GLMM as the primary model, keeping the Gaussian and transformed fits as pre-registered sensitivity analyses, and add the residual diagnostics that were never present.

**Architecture:** A thin Python layer serialises model data to CSV and a spec to JSON, invokes `lme4::glmer` through an absolute-path `Rscript` subprocess, and parses the coefficient table back into **exactly the tidy schema `fit_lmm` already returns** — so every existing plotting function keeps working untouched. Response construction and transformations are pure functions in their own module. Diagnostics are computed for all tracks.

**Tech Stack:** Python 3 (pandas, numpy, statsmodels, scipy, PyYAML, matplotlib), R 4.6.0 with `lme4` 2.0.1, pytest.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md`. Read it before starting.
- **Python interpreter (all commands):** `/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python`. `conda activate` does **not** work in this environment; always invoke by absolute path.
- **Rscript binary:** `/network/iss/apps/lang/r/rcran/4.6.0/bin/Rscript`. Verified to work with a clean environment — do **not** shell out to `module load`.
- **Repo root:** `/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering`. All commands run from here.
- **No `try`/`except` in scientific scripts.** Let errors surface. Subprocess failures are surfaced by checking the return code and raising with R's stderr attached — that is a raise, not a catch-and-continue.
- **No hardcoded parameters.** Everything configurable lives in `glmm_config.yaml`.
- **No magic numbers.** Constants go to config or the top of the module.
- **Type hints required** on every new/modified function signature.
- **Numpy-style docstrings** on every module, function and class.
- **Explicit `random_state`** for every stochastic operation.
- **Never commit files > 10 MB.** Results are gitignored.
- **Deviation from spec §5, deliberate:** the spec listed three new modules. This plan adds a fourth, `response_transforms.py`, holding the pure response-construction and transformation functions. Keeping them out of `glmm_backend.py` gives that module a single responsibility (the R boundary) and makes the transforms testable without R.

---

## File Structure

| File | Responsibility |
|---|---|
| `Behavior/Objective_Markers/glmm_config.yaml` | Marker→family map, response columns, transforms, Rscript path, optimizer, output dir names |
| `Behavior/Objective_Markers/response_transforms.py` | Pure functions: binomial response construction, empirical logit, log transform |
| `Behavior/Objective_Markers/glmm_fit.R` | Fits one `glmer`, writes coefficient + diagnostic CSVs |
| `Behavior/Objective_Markers/glmm_backend.py` | Python↔R boundary; returns `fit_lmm`-compatible tidy frames |
| `Behavior/Objective_Markers/diagnostics.py` | Residual diagnostics + plots for all tracks |
| `Behavior/Objective_Markers/lmm_probe_dimensions.py` | Modified: three-track dispatch, `total_errors` redefinition, `model_comparison.csv` |
| `tests/test_glmm_objective_markers.py` | All unit + round-trip tests |

---

## Task 1: Config and pure response transforms

**Files:**
- Create: `Behavior/Objective_Markers/glmm_config.yaml`
- Create: `Behavior/Objective_Markers/response_transforms.py`
- Test: `tests/test_glmm_objective_markers.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `build_binomial_response(df, success_col, total_col) -> pd.DataFrame` with added columns `_succ`, `_fail`; rows where `total_col == 0` removed.
  - `empirical_logit(successes, totals, correction=0.5) -> np.ndarray`
  - `log_transform(values) -> np.ndarray`
  - `load_glmm_config(path=None) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_glmm_objective_markers.py`:

```python
"""Tests for the GLMM objective-markers backend.

Covers pure response transforms, config loading, the Python-R round trip,
and the count-consistency assumptions the binomial models rely on.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "Behavior" / "Objective_Markers"))

from response_transforms import (  # noqa: E402
    build_binomial_response,
    empirical_logit,
    load_glmm_config,
    log_transform,
)

CONFIG_PATH = REPO_ROOT / "Behavior" / "Objective_Markers" / "glmm_config.yaml"


def test_config_loads_and_has_all_markers():
    cfg = load_glmm_config(CONFIG_PATH)
    families = cfg["markers"]
    assert set(families) == {
        "omission_rate", "commission_rate", "total_errors", "rtcv",
    }
    assert families["omission_rate"]["family"] == "binomial"
    assert families["rtcv"]["family"] == "Gamma"
    assert Path(cfg["r"]["rscript_path"]).exists()


def test_build_binomial_response_adds_success_failure():
    df = pd.DataFrame({"k": [0, 2, 3], "n": [4, 4, 3]})
    out = build_binomial_response(df, "k", "n")
    assert list(out["_succ"]) == [0, 2, 3]
    assert list(out["_fail"]) == [4, 2, 0]


def test_build_binomial_response_drops_zero_denominator():
    df = pd.DataFrame({"k": [0, 1, 0], "n": [0, 1, 2]})
    out = build_binomial_response(df, "k", "n")
    assert len(out) == 2
    assert 0 not in list(out["n"])


def test_empirical_logit_matches_hand_computation():
    got = empirical_logit(np.array([0, 2]), np.array([4, 4]))
    expected = np.array([
        np.log(0.5 / 4.5),
        np.log(2.5 / 2.5),
    ])
    np.testing.assert_allclose(got, expected)


def test_empirical_logit_finite_at_boundaries():
    got = empirical_logit(np.array([0, 4]), np.array([4, 4]))
    assert np.all(np.isfinite(got))


def test_log_transform_rejects_non_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        log_transform(np.array([1.0, 0.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -v -o addopts=""
```

Expected: FAIL with `ModuleNotFoundError: No module named 'response_transforms'`.

- [ ] **Step 3: Write the config**

Create `Behavior/Objective_Markers/glmm_config.yaml`:

```yaml
# Configuration for the GLMM re-specification of objective behavioural markers.
# See docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md

r:
  # Absolute path: verified to work without `module load` (R 4.6.0, lme4 2.0.1).
  rscript_path: /network/iss/apps/lang/r/rcran/4.6.0/bin/Rscript
  script: glmm_fit.R
  optimizer: bobyqa
  maxfun: 100000

markers:
  omission_rate:
    family: binomial
    success_col: n_omissions
    total_col: n_go
    olre: true
  commission_rate:
    family: binomial
    success_col: n_commissions
    total_col: n_nogo
    olre: true
  total_errors:
    family: binomial
    success_col: _total_error_count
    total_col: n_trials_window
    olre: true
  rtcv:
    family: Gamma
    link: log
    response_col: rtcv
    olre: false

# Sensitivity track: transformed response fitted with statsmodels.mixedlm
transforms:
  omission_rate:   {type: empirical_logit, success_col: n_omissions,        total_col: n_go}
  commission_rate: {type: empirical_logit, success_col: n_commissions,      total_col: n_nogo}
  total_errors:    {type: empirical_logit, success_col: _total_error_count, total_col: n_trials_window}
  rtcv:            {type: log,             response_col: rtcv}

# Haldane correction constant for the empirical logit
empirical_logit_correction: 0.5

tracks:
  glmm:                   {output_subdir: glmm,                   primary: true}
  sensitivity_gaussian:   {output_subdir: sensitivity_gaussian,   primary: false}
  sensitivity_transformed: {output_subdir: sensitivity_transformed, primary: false}
  sensitivity_olre:       {output_subdir: sensitivity_olre,       primary: false}

diagnostics:
  output_subdir: diagnostics
  n_residual_bins: 40

random_state: 42
```

- [ ] **Step 4: Write the implementation**

Create `Behavior/Objective_Markers/response_transforms.py`:

```python
"""Pure response-construction and transformation functions for the GLMM track.

These are deliberately free of any I/O or R dependency so they can be unit
tested in isolation. See
``docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "glmm_config.yaml"


def load_glmm_config(path: Path | str | None = None) -> dict:
    """Load the GLMM configuration YAML.

    Parameters
    ----------
    path : Path | str | None
        Config location. Defaults to ``glmm_config.yaml`` beside this module.

    Returns
    -------
    dict
        Parsed configuration.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return yaml.safe_load(open(config_path))


def build_binomial_response(
    df: pd.DataFrame, success_col: str, total_col: str
) -> pd.DataFrame:
    """Add ``_succ``/``_fail`` columns for a ``cbind()`` binomial response.

    Rows whose denominator is zero carry no information about the proportion
    and are dropped. In the ``n10`` dataset this removes the 847 probes with
    no no-go trials from the commission model.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level dataframe.
    success_col : str
        Column holding the event count (numerator).
    total_col : str
        Column holding the trial count (denominator).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``_succ`` and ``_fail`` added and zero-denominator
        rows removed.

    Raises
    ------
    ValueError
        If any numerator exceeds its denominator.
    """
    out = df[df[total_col] > 0].copy()
    if (out[success_col] > out[total_col]).any():
        n_bad = int((out[success_col] > out[total_col]).sum())
        raise ValueError(
            f"{n_bad} rows have {success_col} > {total_col}; "
            "the binomial response is undefined."
        )
    out["_succ"] = out[success_col].astype(int)
    out["_fail"] = (out[total_col] - out[success_col]).astype(int)
    return out


def empirical_logit(
    successes: np.ndarray, totals: np.ndarray, correction: float = 0.5
) -> np.ndarray:
    """Haldane-corrected empirical logit.

    ``log((y + c) / (n - y + c))`` stays finite at ``y = 0`` and ``y = n``,
    which plain ``logit`` does not. This is the standard treatment for
    proportions with small denominators — here as small as 1-4 no-go trials.

    Parameters
    ----------
    successes : np.ndarray
        Event counts.
    totals : np.ndarray
        Trial counts.
    correction : float
        Continuity constant added to both numerator and denominator.

    Returns
    -------
    np.ndarray
        Empirical logit values.
    """
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    return np.log((successes + correction) / (totals - successes + correction))


def log_transform(values: np.ndarray) -> np.ndarray:
    """Natural log of a strictly positive response.

    Parameters
    ----------
    values : np.ndarray
        Response values; must all be > 0.

    Returns
    -------
    np.ndarray
        Log-transformed values.

    Raises
    ------
    ValueError
        If any value is <= 0.
    """
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        raise ValueError("log_transform requires strictly positive values.")
    return np.log(values)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -v -o addopts=""
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add Behavior/Objective_Markers/glmm_config.yaml \
        Behavior/Objective_Markers/response_transforms.py \
        tests/test_glmm_objective_markers.py
git commit -m "feat(behavior): add GLMM config and pure response transforms

Haldane-corrected empirical logit is required because the rate markers are
proportions over 1-4 no-go trials, where a plain logit is undefined at the
65% of probes with zero events."
```

---

## Task 2: R fitting script and Python backend

**Files:**
- Create: `Behavior/Objective_Markers/glmm_fit.R`
- Create: `Behavior/Objective_Markers/glmm_backend.py`
- Modify: `tests/test_glmm_objective_markers.py`

**Interfaces:**
- Consumes: `build_binomial_response`, `load_glmm_config` from Task 1.
- Produces:
  - `fit_glmm(data, marker, predictors, config, olre=False, mcc_method="fdr_bh", mcc_alpha=0.05) -> pd.DataFrame`
    returning columns `predictor, estimate, std_error, t_value, z_value, p_value, conf_lower, conf_upper, p_fdr, significant_fdr, converged, dispersion, n_obs`.

- [ ] **Step 1: Write the failing round-trip test**

Append to `tests/test_glmm_objective_markers.py`:

```python
from glmm_backend import fit_glmm  # noqa: E402

TIDY_COLUMNS = [
    "predictor", "estimate", "std_error", "t_value", "z_value",
    "p_value", "conf_lower", "conf_upper", "p_fdr", "significant_fdr",
]


def _synthetic_binomial(random_state: int = 42) -> pd.DataFrame:
    """Binomial data with a known onoff log-odds effect of 0.8."""
    rng = np.random.default_rng(random_state)
    n_subjects, n_probes, n_trials = 60, 40, 20
    true_beta = 0.8
    rows = []
    for s in range(n_subjects):
        subj_intercept = rng.normal(0.0, 0.5)
        for _ in range(n_probes):
            onoff = rng.normal(0.0, 1.0)
            eta = -1.0 + subj_intercept + true_beta * onoff
            p = 1.0 / (1.0 + np.exp(-eta))
            rows.append({
                "subject": f"S{s:02d}",
                "onoff": onoff,
                "n_events": rng.binomial(n_trials, p),
                "n_total": n_trials,
            })
    return pd.DataFrame(rows)


def test_fit_glmm_recovers_known_effect():
    cfg = load_glmm_config(CONFIG_PATH)
    df = _synthetic_binomial()
    marker_spec = {
        "family": "binomial",
        "success_col": "n_events",
        "total_col": "n_total",
    }
    res = fit_glmm(
        data=df, marker="synthetic", predictors=["onoff"],
        config=cfg, marker_spec=marker_spec,
    )
    beta = float(res.loc[res["predictor"] == "onoff", "estimate"].iloc[0])
    assert beta == pytest.approx(0.8, abs=0.15), f"recovered {beta}"
    assert bool(res.loc[res["predictor"] == "onoff", "significant_fdr"].iloc[0])


def test_fit_glmm_returns_tidy_schema():
    cfg = load_glmm_config(CONFIG_PATH)
    df = _synthetic_binomial()
    marker_spec = {
        "family": "binomial",
        "success_col": "n_events",
        "total_col": "n_total",
    }
    res = fit_glmm(
        data=df, marker="synthetic", predictors=["onoff"],
        config=cfg, marker_spec=marker_spec,
    )
    for col in TIDY_COLUMNS:
        assert col in res.columns, f"missing {col}"
    assert bool(res["converged"].iloc[0])
    assert res["dispersion"].iloc[0] > 0
    # t_value is an alias of the Wald z, kept so plotting code needs no change
    np.testing.assert_allclose(res["t_value"].values, res["z_value"].values)
```

- [ ] **Step 2: Run to verify it fails**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -k glmm -v -o addopts=""
```

Expected: FAIL with `ModuleNotFoundError: No module named 'glmm_backend'`.

- [ ] **Step 3: Write the R script**

Create `Behavior/Objective_Markers/glmm_fit.R`:

```r
# Fit one GLMM with lme4::glmer and write a tidy coefficient table.
#
# Invoked as a subprocess by glmm_backend.py. Reads a model-data CSV and a
# JSON spec, writes a coefficient CSV and a one-row diagnostics CSV.
#
# Confidence intervals are Wald (estimate +/- 1.96 * SE), matching the normal
# approximation statsmodels' mixedlm conf_int() uses, so the GLMM and Gaussian
# tracks are directly comparable.

suppressMessages(library(lme4))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag) args[which(args == flag) + 1]

data_path <- get_arg("--data")
spec_path <- get_arg("--spec")
coef_path <- get_arg("--out-coef")
diag_path <- get_arg("--out-diag")

spec <- fromJSON(spec_path)
d <- read.csv(data_path, stringsAsFactors = FALSE)
d$subject <- factor(d$subject)

rhs <- paste(spec$predictors, collapse = " + ")
re_terms <- "(1|subject)"
if (isTRUE(spec$olre)) {
  d$obs_id <- factor(seq_len(nrow(d)))
  re_terms <- paste(re_terms, "+ (1|obs_id)")
}

ctrl <- glmerControl(optimizer = spec$optimizer,
                     optCtrl = list(maxfun = spec$maxfun))

if (spec$family == "binomial") {
  form <- as.formula(paste("cbind(`_succ`, `_fail`) ~", rhs, "+", re_terms))
  fam <- binomial()
} else if (spec$family == "Gamma") {
  form <- as.formula(paste(spec$response_col, "~", rhs, "+", re_terms))
  fam <- Gamma(link = "log")
} else {
  stop(sprintf("Unsupported family: %s", spec$family))
}

m <- glmer(form, family = fam, data = d, control = ctrl)

co <- summary(m)$coefficients
estimate <- co[, "Estimate"]
std_error <- co[, "Std. Error"]
zval <- co[, 3]
pval <- co[, 4]

coef_df <- data.frame(
  predictor  = rownames(co),
  estimate   = estimate,
  std_error  = std_error,
  z_value    = zval,
  p_value    = pval,
  conf_lower = estimate - 1.96 * std_error,
  conf_upper = estimate + 1.96 * std_error,
  stringsAsFactors = FALSE
)
# lme4 names the fixed intercept "(Intercept)"; align with statsmodels'
# "Intercept" so downstream filtering behaves identically across tracks.
coef_df$predictor[coef_df$predictor == "(Intercept)"] <- "Intercept"
write.csv(coef_df, coef_path, row.names = FALSE)

conv_msgs <- m@optinfo$conv$lme4$messages
rp <- residuals(m, type = "pearson")
diag_df <- data.frame(
  converged   = is.null(conv_msgs) || length(conv_msgs) == 0,
  conv_message = if (is.null(conv_msgs) || length(conv_msgs) == 0) ""
                 else paste(conv_msgs, collapse = "; "),
  dispersion  = sum(rp^2) / df.residual(m),
  n_obs       = nrow(d),
  n_subjects  = nlevels(droplevels(d$subject)),
  stringsAsFactors = FALSE
)
write.csv(diag_df, diag_path, row.names = FALSE)
```

- [ ] **Step 4: Verify `jsonlite` is available**

```bash
/network/iss/apps/lang/r/rcran/4.6.0/bin/Rscript \
  -e 'cat(as.character(packageVersion("jsonlite")), "\n")'
```

Expected: a version number. If it reports an error instead, install it:

```bash
/network/iss/apps/lang/r/rcran/4.6.0/bin/Rscript -e \
  'install.packages("jsonlite", repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))'
```

- [ ] **Step 5: Write the Python backend**

Create `Behavior/Objective_Markers/glmm_backend.py`:

```python
"""Python-R boundary for fitting GLMMs with lme4::glmer.

Serialises model data and a spec, invokes ``glmm_fit.R`` through an
absolute-path ``Rscript`` subprocess, and parses the result into **exactly the
tidy schema returned by** ``lmm_probe_dimensions.fit_lmm`` so that all existing
plotting code works unchanged.

See ``docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md``.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from response_transforms import build_binomial_response

MODULE_DIR = Path(__file__).resolve().parent


def _run_r(
    model_data: pd.DataFrame, spec: dict, rscript_path: str, script_path: Path
) -> tuple[pd.DataFrame, pd.Series]:
    """Invoke the R fitting script and return its coefficient and diag tables.

    Parameters
    ----------
    model_data : pd.DataFrame
        Rows to fit; must already be complete-case.
    spec : dict
        Serialised model specification consumed by ``glmm_fit.R``.
    rscript_path : str
        Absolute path to the Rscript binary.
    script_path : Path
        Absolute path to ``glmm_fit.R``.

    Returns
    -------
    coef_df : pd.DataFrame
        Fixed-effect coefficient table.
    diag : pd.Series
        One-row diagnostics (converged, dispersion, n_obs, n_subjects).

    Raises
    ------
    RuntimeError
        If R exits non-zero. R's stderr is attached to the message.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_file = tmp_path / "data.csv"
        spec_file = tmp_path / "spec.json"
        coef_file = tmp_path / "coef.csv"
        diag_file = tmp_path / "diag.csv"

        model_data.to_csv(data_file, index=False)
        spec_file.write_text(json.dumps(spec))

        proc = subprocess.run(
            [
                rscript_path, str(script_path),
                "--data", str(data_file),
                "--spec", str(spec_file),
                "--out-coef", str(coef_file),
                "--out-diag", str(diag_file),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"glmm_fit.R failed (exit {proc.returncode}).\n"
                f"--- stderr ---\n{proc.stderr}\n"
                f"--- stdout ---\n{proc.stdout}"
            )
        return pd.read_csv(coef_file), pd.read_csv(diag_file).iloc[0]


def fit_glmm(
    data: pd.DataFrame,
    marker: str,
    predictors: list[str],
    config: dict,
    marker_spec: dict,
    olre: bool = False,
    mcc_method: str = "fdr_bh",
    mcc_alpha: float = 0.05,
) -> pd.DataFrame:
    """Fit one GLMM and return a tidy predictor-level results dataframe.

    The returned frame is column-compatible with ``fit_lmm`` so the existing
    forest/scatter plotting functions consume it unchanged. ``t_value`` holds
    the Wald z statistic (duplicated into ``z_value``, which is its honest
    name) because the plotting code reads ``row["t_value"]`` directly.

    Parameters
    ----------
    data : pd.DataFrame
        Probe-level dataframe with predictors and response/count columns.
    marker : str
        Marker name, used for logging only.
    predictors : list[str]
        Fixed-effect predictor names; FDR is applied across these.
    config : dict
        Parsed ``glmm_config.yaml``.
    marker_spec : dict
        This marker's entry from ``config["markers"]``.
    olre : bool
        Add an observation-level random effect (binomial markers only).
    mcc_method : str
        Multiple-comparisons method passed to ``multipletests``.
    mcc_alpha : float
        Significance threshold after correction.

    Returns
    -------
    pd.DataFrame
        Columns: predictor, estimate, std_error, t_value, z_value, p_value,
        conf_lower, conf_upper, p_fdr, significant_fdr, converged,
        dispersion, n_obs.
    """
    family = marker_spec["family"]

    if family == "binomial":
        needed = ["subject", marker_spec["success_col"],
                  marker_spec["total_col"]] + predictors
        model_data = data[[c for c in needed if c in data.columns]].dropna()
        model_data = build_binomial_response(
            model_data, marker_spec["success_col"], marker_spec["total_col"]
        )
        response_col = None
    else:
        needed = ["subject", marker_spec["response_col"]] + predictors
        model_data = data[[c for c in needed if c in data.columns]].dropna()
        response_col = marker_spec["response_col"]
        model_data = model_data[model_data[response_col] > 0]

    model_data = model_data.copy()
    model_data["subject"] = model_data["subject"].astype(str)

    spec = {
        "predictors": predictors,
        "family": family,
        "response_col": response_col,
        "olre": bool(olre),
        "optimizer": config["r"]["optimizer"],
        "maxfun": int(config["r"]["maxfun"]),
    }

    print(f"\n{'='*60}")
    print(f"GLMM: {marker}  |  family={family}  olre={olre}")
    print(f"N observations: {len(model_data)}  |  "
          f"N subjects: {model_data['subject'].nunique()}")
    print("="*60)

    coef_df, diag = _run_r(
        model_data, spec,
        config["r"]["rscript_path"],
        MODULE_DIR / config["r"]["script"],
    )

    results_df = coef_df.rename(columns={"z_value": "_z"})
    results_df["z_value"] = results_df["_z"]
    results_df["t_value"] = results_df["_z"]
    results_df = results_df.drop(columns=["_z"])

    predictor_rows = results_df[results_df["predictor"].isin(predictors)].copy()
    _, p_fdr, _, _ = multipletests(
        predictor_rows["p_value"].values, method=mcc_method
    )
    predictor_rows["p_fdr"] = p_fdr
    predictor_rows["significant_fdr"] = predictor_rows["p_fdr"] < mcc_alpha

    results_df = results_df.merge(
        predictor_rows[["predictor", "p_fdr", "significant_fdr"]],
        on="predictor", how="left",
    )
    results_df["converged"] = bool(diag["converged"])
    results_df["conv_message"] = str(diag["conv_message"])
    results_df["dispersion"] = float(diag["dispersion"])
    results_df["n_obs"] = int(diag["n_obs"])

    if not bool(diag["converged"]):
        print(f"  !! CONVERGENCE WARNING for {marker}: {diag['conv_message']}")
    print(f"  dispersion = {float(diag['dispersion']):.3f}")

    return results_df
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -v -o addopts=""
```

Expected: 8 passed. The round-trip test is the critical one — it proves the Python↔R boundary does not corrupt the model.

- [ ] **Step 7: Commit**

```bash
git add Behavior/Objective_Markers/glmm_fit.R \
        Behavior/Objective_Markers/glmm_backend.py \
        tests/test_glmm_objective_markers.py
git commit -m "feat(behavior): add lme4 glmer backend via Rscript subprocess

Subprocess rather than rpy2: conda.anaconda.org is blocked by the institutional
proxy and rpy2 fails to build against the conda R, but lme4 2.0.1 is already
present in the R/4.6.0 module. Round-trip test recovers a known log-odds
effect, validating the serialisation boundary end to end."
```

---

## Task 3: Real-data count validation and moderation backend

**Files:**
- Modify: `Behavior/Objective_Markers/glmm_backend.py`
- Modify: `tests/test_glmm_objective_markers.py`

**Interfaces:**
- Consumes: `fit_glmm`, `_run_r` from Task 2.
- Produces: `fit_moderation_glmm(data, marker, moderator, config, marker_spec) -> dict` with keys `marker, moderator, interaction_term, estimate, std_error, t_value, p_value, n_obs, n_subjects, converged` — the same contract as `fit_moderation_lmm`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_glmm_objective_markers.py`:

```python
from glmm_backend import fit_moderation_glmm  # noqa: E402

MARKER_CSVS = {
    "full_segment": REPO_ROOT / "results/Behavior/objective_markers/objective_markers_per_probe.csv",
    "n10": REPO_ROOT / "results/Behavior/objective_markers/objective_markers_per_probe_n10.csv",
}


@pytest.mark.parametrize("dataset", ["full_segment", "n10"])
def test_count_consistency_holds(dataset):
    """Binomial responses are only defined if no numerator exceeds its denominator."""
    df = pd.read_csv(MARKER_CSVS[dataset])
    assert (df["n_omissions"] <= df["n_go"]).all()
    assert (df["n_commissions"] <= df["n_nogo"]).all()
    assert (df["n_omissions"] + df["n_commissions"] <= df["n_trials_window"]).all()
    assert (df["n_go"] + df["n_nogo"] == df["n_trials_window"]).all()


def test_n10_zero_nogo_probes_are_dropped():
    """847 n10 probes have no no-go trials; they must leave the commission model."""
    df = pd.read_csv(MARKER_CSVS["n10"])
    n_zero = int((df["n_nogo"] == 0).sum())
    assert n_zero == 847
    out = build_binomial_response(df, "n_commissions", "n_nogo")
    assert len(out) == len(df) - n_zero


def test_fit_moderation_glmm_contract():
    cfg = load_glmm_config(CONFIG_PATH)
    df = _synthetic_binomial()
    rng = np.random.default_rng(7)
    df["valence"] = rng.normal(0.0, 1.0, len(df))
    marker_spec = {
        "family": "binomial",
        "success_col": "n_events",
        "total_col": "n_total",
    }
    out = fit_moderation_glmm(
        data=df, marker="synthetic", moderator="valence",
        config=cfg, marker_spec=marker_spec,
    )
    assert set(out) == {
        "marker", "moderator", "interaction_term", "estimate", "std_error",
        "t_value", "p_value", "n_obs", "n_subjects", "converged",
    }
    assert out["interaction_term"] == "onoff:valence"
    assert np.isfinite(out["estimate"])
```

- [ ] **Step 2: Run to verify they fail**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -k "moderation or count or n10" -v -o addopts=""
```

Expected: FAIL — `ImportError: cannot import name 'fit_moderation_glmm'`.

- [ ] **Step 3: Add the moderation function**

Append to `Behavior/Objective_Markers/glmm_backend.py`:

```python
def fit_moderation_glmm(
    data: pd.DataFrame,
    marker: str,
    moderator: str,
    config: dict,
    marker_spec: dict,
) -> dict:
    """Fit ``marker ~ onoff * moderator + (1|subject)`` as a GLMM.

    Mirrors the return contract of ``lmm_probe_dimensions.fit_moderation_lmm``
    so ``run_moderation_analysis`` can dispatch between backends without any
    change to its FDR or plotting logic.

    Parameters
    ----------
    data : pd.DataFrame
        Probe-level dataframe.
    marker : str
        Dependent variable name.
    moderator : str
        Moderator whose interaction with ``onoff`` is the test of interest.
    config : dict
        Parsed ``glmm_config.yaml``.
    marker_spec : dict
        This marker's entry from ``config["markers"]``.

    Returns
    -------
    dict
        Keys: marker, moderator, interaction_term, estimate, std_error,
        t_value, p_value, n_obs, n_subjects, converged.
    """
    interaction_term = f"onoff:{moderator}"
    results_df = fit_glmm(
        data=data,
        marker=marker,
        predictors=[f"onoff * {moderator}"],
        config=config,
        marker_spec=marker_spec,
        olre=False,
    )
    row = results_df[results_df["predictor"] == interaction_term]
    if len(row) == 0:
        raise RuntimeError(
            f"Interaction term {interaction_term!r} absent from glmer output "
            f"for marker {marker!r}. Terms present: "
            f"{list(results_df['predictor'])}"
        )
    row = row.iloc[0]
    return {
        "marker": marker,
        "moderator": moderator,
        "interaction_term": interaction_term,
        "estimate": float(row["estimate"]),
        "std_error": float(row["std_error"]),
        "t_value": float(row["t_value"]),
        "p_value": float(row["p_value"]),
        "n_obs": int(row["n_obs"]),
        "n_subjects": int(data["subject"].nunique()),
        "converged": bool(row["converged"]),
    }
```

Note: passing `"onoff * moderator"` as a single predictor string lets R expand
it into main effects plus the interaction. FDR across a single entry is a
no-op here; the real correction happens across all marker × moderator tests in
`run_moderation_analysis`, exactly as in the Gaussian track.

- [ ] **Step 4: Run tests to verify they pass**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -v -o addopts=""
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add Behavior/Objective_Markers/glmm_backend.py tests/test_glmm_objective_markers.py
git commit -m "feat(behavior): add GLMM moderation backend and real-data count checks

Count-consistency tests assert the binomial responses are well defined in both
datasets, and pin the 847 n10 probes with no no-go trials that must leave the
commission model."
```

---

## Task 4: Residual diagnostics

**Files:**
- Create: `Behavior/Objective_Markers/diagnostics.py`
- Modify: `tests/test_glmm_objective_markers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `binned_residuals(fitted, residuals, n_bins) -> pd.DataFrame` with columns `bin, x_mean, y_mean, se, n, outside_band`
  - `gaussian_residual_diagnostics(residuals, fitted) -> dict` with keys `skew, kurtosis, breusch_pagan_stat, breusch_pagan_p, n`
  - `save_diagnostic_plots(fitted, residuals, label, output_dir, n_bins) -> Path`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_glmm_objective_markers.py`:

```python
from diagnostics import (  # noqa: E402
    binned_residuals,
    gaussian_residual_diagnostics,
)


def test_binned_residuals_bin_count_and_band():
    rng = np.random.default_rng(3)
    fitted = rng.uniform(0, 1, 1000)
    resid = rng.normal(0, 1, 1000)
    out = binned_residuals(fitted, resid, n_bins=20)
    assert len(out) == 20
    assert out["n"].sum() == 1000
    # Well-behaved residuals: most bins inside the +/- 2 SE band
    assert out["outside_band"].mean() < 0.25


def test_gaussian_residual_diagnostics_flags_skew():
    rng = np.random.default_rng(11)
    fitted = rng.uniform(0, 1, 2000)
    skewed = rng.exponential(1.0, 2000)
    out = gaussian_residual_diagnostics(skewed, fitted)
    assert out["skew"] > 1.0
    assert out["n"] == 2000
    assert 0.0 <= out["breusch_pagan_p"] <= 1.0
```

- [ ] **Step 2: Run to verify they fail**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -k "binned or diagnostics" -v -o addopts=""
```

Expected: FAIL with `ModuleNotFoundError: No module named 'diagnostics'`.

- [ ] **Step 3: Write the implementation**

Create `Behavior/Objective_Markers/diagnostics.py`:

```python
"""Residual diagnostics for the objective-marker models.

This module exists because the original pipeline validated no distributional
assumption at all. It provides Gaussian-track diagnostics (QQ, residuals vs
fitted, skew/kurtosis, Breusch-Pagan) and binomial-track binned residual plots
(Gelman & Hill), which are the standard substitute for DHARMa's
simulation-based residuals — DHARMa is not installed and cannot be installed
because the conda channel is blocked by the institutional proxy.

With n ~ 2460, Shapiro-Wilk rejects on trivial deviations, so normality is
reported as descriptive summaries plus plots rather than a pass/fail test.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: E402

FIG_DPI = 300
SE_BAND_MULTIPLIER = 2.0


def binned_residuals(
    fitted: np.ndarray, residuals: np.ndarray, n_bins: int
) -> pd.DataFrame:
    """Gelman-Hill binned residual summary.

    Raw residuals from a binomial model are uninformative when plotted against
    fitted values because the response is discrete. Averaging within bins of
    the fitted value reveals systematic misfit that the raw plot hides.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values on the response scale.
    residuals : np.ndarray
        Residuals (Pearson or response scale).
    n_bins : int
        Number of equal-count bins.

    Returns
    -------
    pd.DataFrame
        Columns: bin, x_mean, y_mean, se, n, outside_band. ``outside_band`` is
        True where the bin mean falls outside +/- 2 SE, which under a
        well-specified model should happen for about 5% of bins.
    """
    df = pd.DataFrame({"fitted": np.asarray(fitted), "resid": np.asarray(residuals)})
    df["bin"] = pd.qcut(df["fitted"].rank(method="first"), n_bins, labels=False)

    agg = (
        df.groupby("bin")
        .agg(x_mean=("fitted", "mean"), y_mean=("resid", "mean"),
             sd=("resid", "std"), n=("resid", "size"))
        .reset_index()
    )
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    agg["outside_band"] = (
        agg["y_mean"].abs() > SE_BAND_MULTIPLIER * agg["se"]
    )
    return agg.drop(columns=["sd"])


def gaussian_residual_diagnostics(
    residuals: np.ndarray, fitted: np.ndarray
) -> dict:
    """Descriptive normality and homoscedasticity summaries.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals.
    fitted : np.ndarray
        Fitted values, used as the Breusch-Pagan regressor.

    Returns
    -------
    dict
        Keys: skew, kurtosis, breusch_pagan_stat, breusch_pagan_p, n.
    """
    residuals = np.asarray(residuals, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    exog = sm.add_constant(fitted)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, exog)
    return {
        "skew": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_p": float(bp_p),
        "n": int(len(residuals)),
    }


def save_diagnostic_plots(
    fitted: np.ndarray,
    residuals: np.ndarray,
    label: str,
    output_dir: Path,
    n_bins: int,
) -> Path:
    """Write a three-panel diagnostic figure: QQ, residuals vs fitted, binned.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values.
    residuals : np.ndarray
        Residuals.
    label : str
        Figure title and filename stem, e.g. ``"glmm_commission_rate"``.
    output_dir : Path
        Destination directory; created if absent.
    n_bins : int
        Bin count for the binned-residual panel.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Normal Q-Q")

    axes[1].scatter(fitted, residuals, s=6, alpha=0.25, linewidths=0)
    axes[1].axhline(0, color="black", lw=1.2, ls="--")
    axes[1].set_xlabel("Fitted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Fitted")

    binned = binned_residuals(fitted, residuals, n_bins)
    axes[2].errorbar(binned["x_mean"], binned["y_mean"],
                     yerr=SE_BAND_MULTIPLIER * binned["se"],
                     fmt="o", markersize=5, capsize=3)
    axes[2].axhline(0, color="black", lw=1.2, ls="--")
    axes[2].set_xlabel("Mean fitted value in bin")
    axes[2].set_ylabel("Mean residual")
    axes[2].set_title(
        f"Binned residuals ({int(binned['outside_band'].sum())}/"
        f"{len(binned)} outside ±2 SE)"
    )

    fig.suptitle(label, fontsize=13, fontweight="bold")
    out_path = output_dir / f"{label}_diagnostics.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -v -o addopts=""
```

Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add Behavior/Objective_Markers/diagnostics.py tests/test_glmm_objective_markers.py
git commit -m "feat(behavior): add residual diagnostics for objective-marker models

Closes the gap that motivated this work: the pipeline previously validated no
distributional assumption. Binned residuals (Gelman-Hill) substitute for
DHARMa, which cannot be installed on this network."
```

---

## Task 5: Three-track integration in the pipeline

**Files:**
- Modify: `Behavior/Objective_Markers/lmm_probe_dimensions.py`

**Interfaces:**
- Consumes: `fit_glmm`, `fit_moderation_glmm`, `load_glmm_config`, `empirical_logit`, `log_transform`, diagnostics helpers.
- Produces: `model_comparison.csv` per dataset.

- [ ] **Step 1: Add imports and config load**

In `Behavior/Objective_Markers/lmm_probe_dimensions.py`, after the existing
imports (around line 78), add:

```python
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glmm_backend import fit_glmm, fit_moderation_glmm
from response_transforms import empirical_logit, load_glmm_config, log_transform
from diagnostics import gaussian_residual_diagnostics, save_diagnostic_plots

GLMM_CONFIG = load_glmm_config()
```

- [ ] **Step 2: Redefine `total_errors` and add the legacy marker**

Replace lines 1836-1838 of `run_pipeline_for_dataset`:

```python
    df_markers["total_errors"] = (
        df_markers["omission_rate"] + df_markers["commission_rate"]
    )
```

with:

```python
    # New definition (all three tracks): a genuine proportion of all trials.
    # The old sum-of-two-rates mixed denominators (~27 go vs ~3 no-go trials)
    # and was a proportion of nothing. See the design spec, section 2.
    df_markers["_total_error_count"] = (
        df_markers["n_omissions"] + df_markers["n_commissions"]
    )
    df_markers["total_errors"] = (
        df_markers["_total_error_count"] / df_markers["n_trials_window"]
    )
    # Retained only for continuity with already-published Gaussian output.
    df_markers["total_errors_legacy"] = (
        df_markers["omission_rate"] + df_markers["commission_rate"]
    )
```

- [ ] **Step 3: Add the transformed-response builder**

Add this function immediately before `_fit_predictor_set` (around line 1696):

```python
def apply_response_transforms(
    df: pd.DataFrame, markers: list[str], glmm_config: dict
) -> pd.DataFrame:
    """Return a copy of *df* with each marker replaced by its transformed form.

    Used by the ``sensitivity_transformed`` track: ``log`` for ``rtcv`` and the
    Haldane-corrected empirical logit for the rate markers, which stays finite
    at the 65% of probes with zero events.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level dataframe with raw markers and count columns.
    markers : list[str]
        Markers to transform.
    glmm_config : dict
        Parsed ``glmm_config.yaml``.

    Returns
    -------
    pd.DataFrame
        Copy with transformed marker columns.
    """
    out = df.copy()
    correction = glmm_config["empirical_logit_correction"]
    for marker in markers:
        spec = glmm_config["transforms"][marker]
        if spec["type"] == "empirical_logit":
            valid = out[spec["total_col"]] > 0
            out.loc[~valid, marker] = np.nan
            out.loc[valid, marker] = empirical_logit(
                out.loc[valid, spec["success_col"]].values,
                out.loc[valid, spec["total_col"]].values,
                correction=correction,
            )
        elif spec["type"] == "log":
            valid = out[spec["response_col"]] > 0
            out.loc[~valid, marker] = np.nan
            out.loc[valid, marker] = log_transform(
                out.loc[valid, spec["response_col"]].values
            )
        else:
            raise ValueError(f"Unknown transform type: {spec['type']}")
    return out
```

- [ ] **Step 4: Add a `backend` parameter to `_fit_predictor_set`**

Change the signature at line 1696 from:

```python
def _fit_predictor_set(
    df: pd.DataFrame, set_name: str, set_cfg: dict, output_dir: Path,
) -> dict[str, pd.DataFrame]:
```

to:

```python
def _fit_predictor_set(
    df: pd.DataFrame, set_name: str, set_cfg: dict, output_dir: Path,
    backend: str = "gaussian", olre: bool = False,
    markers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
```

and replace the additive fitting loop (lines 1750-1763) with:

```python
    fit_markers = markers if markers is not None else OBJECTIVE_MARKERS
    results: dict[str, pd.DataFrame] = {}
    for marker in fit_markers:
        if backend == "glmm":
            results_df = fit_glmm(
                data=df_fit, marker=marker, predictors=predictors,
                config=GLMM_CONFIG,
                marker_spec=GLMM_CONFIG["markers"][marker],
                olre=olre, mcc_method=MCC_METHOD, mcc_alpha=MCC_ALPHA,
            )
        else:
            results_df = fit_lmm(
                data=df_fit, marker=marker,
                formula_template=set_cfg["formula_template"],
                method=LMM_METHOD, maxiter=LMM_MAXITER,
                predictors=predictors,
            )
        results[marker] = results_df
        csv_path = set_dir / f"{marker}_lmm_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results: {csv_path}")
```

Also replace the two `OBJECTIVE_MARKERS` references later in the function — in
the summary loop (line 1767 uses `results.items()`, already fine) and in the
moderation call (line 1796, `markers=OBJECTIVE_MARKERS`) — with `fit_markers`.

- [ ] **Step 5: Add the track loop and `model_comparison.csv`**

Replace lines 1896-1897 of `run_pipeline_for_dataset`:

```python
    for set_name, set_cfg in PREDICTOR_SETS.items():
        _fit_predictor_set(df, set_name, set_cfg, output_dir)
```

with:

```python
    track_results: dict[str, dict[str, pd.DataFrame]] = {}
    for set_name, set_cfg in PREDICTOR_SETS.items():
        # --- PRIMARY: GLMM ---------------------------------------------------
        glmm_cfg = dict(set_cfg, output_subdir="glmm")
        track_results["glmm"] = _fit_predictor_set(
            df, set_name, glmm_cfg, output_dir, backend="glmm",
        )

        # --- SENSITIVITY: OLRE variant (binomial markers only) ---------------
        olre_markers = [
            m for m in OBJECTIVE_MARKERS
            if GLMM_CONFIG["markers"][m].get("olre", False)
        ]
        olre_cfg = dict(set_cfg, output_subdir="sensitivity_olre")
        track_results["sensitivity_olre"] = _fit_predictor_set(
            df, set_name, olre_cfg, output_dir, backend="glmm",
            olre=True, markers=olre_markers,
        )

        # --- SENSITIVITY: Gaussian (existing model, plus legacy total_errors) -
        gauss_cfg = dict(set_cfg, output_subdir="sensitivity_gaussian")
        track_results["sensitivity_gaussian"] = _fit_predictor_set(
            df, set_name, gauss_cfg, output_dir, backend="gaussian",
            markers=OBJECTIVE_MARKERS + ["total_errors_legacy"],
        )

        # --- SENSITIVITY: transformed response -------------------------------
        df_transformed = apply_response_transforms(
            df, OBJECTIVE_MARKERS, GLMM_CONFIG
        )
        trans_cfg = dict(set_cfg, output_subdir="sensitivity_transformed")
        track_results["sensitivity_transformed"] = _fit_predictor_set(
            df_transformed, set_name, trans_cfg, output_dir,
            backend="gaussian",
        )

    comparison_path = output_dir / "model_comparison.csv"
    build_model_comparison(track_results, PREDICTORS).to_csv(
        comparison_path, index=False
    )
    print(f"\nModel comparison saved: {comparison_path}")
```

- [ ] **Step 6: Add the comparison builder**

Add immediately before `run_pipeline_for_dataset`:

```python
def build_model_comparison(
    track_results: dict[str, dict[str, pd.DataFrame]],
    predictors: list[str],
) -> pd.DataFrame:
    """Assemble the side-by-side specification comparison table.

    One row per marker x predictor, with the estimate, p-value and FDR
    significance from each track. This is the artefact that lets the manuscript
    state whether conclusions are robust to specification — or show exactly
    where they are not. ``total_errors_legacy`` is excluded because its
    different marker definition makes it non-comparable.

    Parameters
    ----------
    track_results : dict[str, dict[str, pd.DataFrame]]
        Mapping ``{track_name: {marker: fit results}}``.
    predictors : list[str]
        Predictors to include.

    Returns
    -------
    pd.DataFrame
        Long-to-wide comparison table.
    """
    rows: list[dict] = []
    for track, per_marker in track_results.items():
        for marker, results_df in per_marker.items():
            if marker == "total_errors_legacy":
                continue
            subset = results_df[results_df["predictor"].isin(predictors)]
            for _, r in subset.iterrows():
                rows.append({
                    "marker": marker,
                    "predictor": r["predictor"],
                    "track": track,
                    "estimate": r["estimate"],
                    "p_value": r["p_value"],
                    "p_fdr": r.get("p_fdr", np.nan),
                    "significant_fdr": r.get("significant_fdr", False),
                })
    long_df = pd.DataFrame(rows)
    wide = long_df.pivot_table(
        index=["marker", "predictor"], columns="track",
        values=["estimate", "p_fdr", "significant_fdr"],
        aggfunc="first",
    )
    wide.columns = [f"{stat}__{track}" for stat, track in wide.columns]
    return wide.reset_index()
```

- [ ] **Step 7: Verify the module imports cleanly**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -c "
import sys; sys.path.insert(0, 'Behavior/Objective_Markers')
import lmm_probe_dimensions as m
print('import OK'); print('markers:', m.OBJECTIVE_MARKERS)
print('glmm families:', {k: v['family'] for k, v in m.GLMM_CONFIG['markers'].items()})
"
```

Expected: `import OK`, the four markers, and the family map.

- [ ] **Step 8: Commit**

```bash
git add Behavior/Objective_Markers/lmm_probe_dimensions.py
git commit -m "feat(behavior): fit objective markers across GLMM and sensitivity tracks

GLMM becomes the primary model; Gaussian, transformed and OLRE fits are
pre-registered sensitivity analyses, all reported regardless of outcome so no
specification is selected after seeing the data. total_errors is redefined as a
genuine joint proportion; the old sum-of-rates is kept as total_errors_legacy
in the Gaussian track only."
```

---

## Task 6: End-to-end run and scientific validation

**Files:**
- Modify: `Behavior/Objective_Markers/lmm_probe_dimensions.py` (only if the run surfaces defects)

- [ ] **Step 1: Run the full pipeline**

```bash
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python \
  Behavior/Objective_Markers/lmm_probe_dimensions.py 2>&1 | tee /tmp/glmm_run.log
```

Expected: completes for both datasets. Watch for `!! CONVERGENCE WARNING`
lines — record them, do not suppress them.

- [ ] **Step 2: Check convergence and dispersion across every fit**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -c "
import pandas as pd, glob
rows = []
for f in glob.glob('results/Behavior/objective_markers/lmm_probe_dimensions/*/glmm/*_lmm_results.csv'):
    d = pd.read_csv(f)
    rows.append({'file': f.split('lmm_probe_dimensions/')[1],
                 'converged': bool(d['converged'].iloc[0]),
                 'dispersion': round(float(d['dispersion'].iloc[0]), 3)})
print(pd.DataFrame(rows).to_string(index=False))
"
```

Expected: a dispersion value for every model. Dispersion far above 1 indicates
overdispersion — this is *expected* and is exactly why the OLRE track is
fitted; record the values, do not act on them by switching the primary model.

- [ ] **Step 3: Verify the comparison table is populated**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -c "
import pandas as pd
for ds in ['full_segment', 'n10']:
    d = pd.read_csv(f'results/Behavior/objective_markers/lmm_probe_dimensions/{ds}/model_comparison.csv')
    sig = [c for c in d.columns if c.startswith('significant_fdr__')]
    print(f'=== {ds}: {len(d)} rows ===')
    print(d[['marker','predictor'] + sig].to_string(index=False))
"
```

Expected: one row per marker × predictor with a significance flag per track.

- [ ] **Step 4: Scientific plausibility review**

Confirm before declaring done:

1. `onoff` retains the same **sign** in the GLMM as in the Gaussian track for
   `omission_rate` and `rtcv`. A sign flip means a specification or link-scale
   bug, not a finding.
2. GLMM `omission_rate` intercept in log-odds is around
   `log(0.038 / 0.962) ≈ -3.2`, matching the 3.8% observed omission rate.
3. `n_obs` for `n10` `commission_rate` equals **1613** (2460 − 847).
4. Diagnostic figures exist for every marker in every track.

- [ ] **Step 5: Run the full test suite**

```bash
/network/iss/home/nicolas.bruno/miniforge3/envs/eeg/bin/python -m pytest \
  tests/test_glmm_objective_markers.py -v -o addopts=""
```

Expected: 14 passed.

- [ ] **Step 6: Commit results metadata**

```bash
git add -u
git status --short   # confirm no files >10MB and no result CSVs staged
git commit -m "fix(behavior): corrections surfaced by the end-to-end GLMM run"
```

If Step 1-4 surfaced no defects, skip this commit.

---

## Self-Review Notes

**Spec coverage:** §2 locked decisions → Tasks 1, 3, 5. §3 engine → Task 2.
§4 model spec, overdispersion rule, both sensitivity tracks → Tasks 1, 2, 5.
§5 architecture → Tasks 1-4 (plus the documented `response_transforms.py`
addition). §6 integration points → Task 5. §7 outputs → Task 5. §8 testing →
all five listed tests appear in Tasks 1-4. §9 risks: convergence recorded as a
column (Task 2 Step 5), R errors raised with stderr (Task 2 Step 5), module
path in config (Task 1 Step 3).

**Known gap, deliberate:** the spec's §5 mention of updating GLMM plot axis
labels to read *z* rather than *t* is not given its own task. The `z_value`
column ships in Task 2 so the CSVs are honest; changing the rendered axis text
touches `_draw_forest_panel`, which is shared with the Gaussian track, and is
better handled once the figures are being regenerated for the manuscript.
