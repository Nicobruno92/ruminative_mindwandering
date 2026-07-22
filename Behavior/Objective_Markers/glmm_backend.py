"""Python-R boundary for fitting GLMMs with lme4::glmer.

Serialises model data and a spec, invokes ``glmm_fit.R`` through an
absolute-path ``Rscript`` subprocess, and parses the result into **exactly the
tidy schema returned by** ``lmm_probe_dimensions.fit_lmm`` so that all existing
plotting code works unchanged.

See ``docs/superpowers/specs/2026-07-22-glmm-objective-markers-design.md``.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from response_transforms import build_binomial_response

# =============================================================================
# CONFIGURATION
# =============================================================================

MODULE_DIR = Path(__file__).resolve().parent

# =============================================================================
# R BOUNDARY
# =============================================================================


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


# =============================================================================
# PUBLIC API
# =============================================================================


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
        conv_message, dispersion, n_obs.
    """
    family = marker_spec["family"]
    # Interaction predictors ("onoff * valence") reference their component
    # columns, which must be present in the model frame.
    model_cols = sorted({
        token
        for predictor in predictors
        for token in predictor.replace("*", " ").replace(":", " ").split()
    })

    if family == "binomial":
        needed = ["subject", marker_spec["success_col"],
                  marker_spec["total_col"]] + model_cols
        model_data = data[[c for c in needed if c in data.columns]].dropna()
        model_data = build_binomial_response(
            model_data, marker_spec["success_col"], marker_spec["total_col"]
        )
        response_col = None
    else:
        needed = ["subject", marker_spec["response_col"]] + model_cols
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

    results_df = coef_df.copy()
    results_df["t_value"] = results_df["z_value"]

    predictor_rows = results_df[results_df["predictor"].isin(predictors)].copy()
    if len(predictor_rows) == 0:
        # Interaction specs expand in R, so the requested names may not appear
        # verbatim; correct across every non-intercept term instead.
        predictor_rows = results_df[
            results_df["predictor"] != "Intercept"
        ].copy()
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
