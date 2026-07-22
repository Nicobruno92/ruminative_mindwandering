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

from diagnostics import gaussian_residual_diagnostics, save_diagnostic_plots
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
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Invoke the R fitting script and return coefficients, diagnostics, residuals.

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
    resid_df : pd.DataFrame
        Per-observation ``fitted`` and Pearson ``residual``.

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
        resid_file = tmp_path / "resid.csv"

        model_data.to_csv(data_file, index=False)
        spec_file.write_text(json.dumps(spec))

        proc = subprocess.run(
            [
                rscript_path, str(script_path),
                "--data", str(data_file),
                "--spec", str(spec_file),
                "--out-coef", str(coef_file),
                "--out-diag", str(diag_file),
                "--out-resid", str(resid_file),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"glmm_fit.R failed (exit {proc.returncode}).\n"
                f"--- stderr ---\n{proc.stderr}\n"
                f"--- stdout ---\n{proc.stdout}"
            )
        return (
            pd.read_csv(coef_file),
            pd.read_csv(diag_file).iloc[0],
            pd.read_csv(resid_file),
        )


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
    diagnostics_dir: Path | str | None = None,
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
    diagnostics_dir : Path | str | None
        When given, write residual diagnostic plots and a summary CSV there.

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

    coef_df, diag, resid_df = _run_r(
        model_data, spec,
        config["r"]["rscript_path"],
        MODULE_DIR / config["r"]["script"],
    )

    if diagnostics_dir is not None:
        label = f"glmm_olre_{marker}" if olre else f"glmm_{marker}"
        save_diagnostic_plots(
            fitted=resid_df["fitted"].values,
            residuals=resid_df["residual"].values,
            label=label,
            output_dir=Path(diagnostics_dir),
            n_bins=int(config["diagnostics"]["n_residual_bins"]),
        )
        summary = gaussian_residual_diagnostics(
            resid_df["residual"].values, resid_df["fitted"].values
        )
        summary.update({
            "model": label, "family": family,
            "dispersion": float(diag["dispersion"]),
            "converged": bool(diag["converged"]),
        })
        pd.DataFrame([summary]).to_csv(
            Path(diagnostics_dir) / f"{label}_residual_summary.csv", index=False
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

    Raises
    ------
    RuntimeError
        If the interaction term is absent from the glmer output.
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
