import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import textwrap

# =============================================================================
# CONFIGURATION - Modify these variables if paths change
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

DEMOGRAPHICS_DIR = RESULTS_DIR / "demographics"
DEMOGRAPHICS_STATS_CSV = DEMOGRAPHICS_DIR / "group_comparison_demographics_psychometrics.csv"
DEMOGRAPHICS_PLOTS_DIR = DEMOGRAPHICS_DIR / "raincloud_plots"

BEHAVIOR_DIR = RESULTS_DIR / "Behavior"
PROBE_DATA_DIR = BEHAVIOR_DIR / "probe_data"
BDI_SPLIT_DIR = BEHAVIOR_DIR / "bdi_split"
MEDIATION_DIR = BEHAVIOR_DIR / "mediation_analysis"

# Common helpers for safe loading of images/HTML

def safe_image(path: Path, caption: str = "") -> None:
    """Display an image if it exists and is valid, otherwise show an info message.

    This wrapper avoids hard crashes when a file exists but is not a valid image
    (e.g., very small PNGs or text files saved with a .png extension).
    """
    if not path.exists():
        st.info(f"Image not found: {path}")
        return

    try:
        # width="stretch" replaces deprecated use_container_width=True
        st.image(str(path), caption=caption, width="stretch")
    except Exception:
        # If PIL/Streamlit cannot interpret the file as an image, show a message
        st.info(f"File exists but could not be loaded as an image: {path}")


def safe_html_plot(path: Path, caption: str = "") -> None:
    """Embed an HTML plot (e.g., Plotly) if it exists and is readable.

    This is used for interactive scatter plots saved as standalone HTML files.
    """
    if not path.exists():
        st.info(f"HTML plot not found: {path}")
        return

    try:
        html = path.read_text(encoding="utf-8")
        if caption:
            st.markdown(f"**{caption}**")
        components.html(html, height=500, scrolling=True)
    except Exception:
        st.info(f"File exists but could not be loaded as HTML: {path}")


def safe_table(path: Path, title: str = "Table") -> None:
    """Display a table if CSV exists, otherwise show a warning."""
    if path.exists():
        st.subheader(title)
        df = pd.read_csv(path)
        st.dataframe(df)
    else:
        st.info(f"CSV not found: {path}")


def load_bic_from_metrics(metrics_path: Path) -> str:
    """Return BIC from a metrics CSV as a formatted string, or '?' if unavailable.

    This is used to keep the BIC values shown in table titles in sync with the
    underlying LMM outputs. If the file or column is missing, a fallback "?" is
    returned so that the UI still renders.
    """
    if not metrics_path.exists():
        return "?"

    try:
        df = pd.read_csv(metrics_path)
        if "bic" not in df.columns:
            return "?"
        bic_value = float(df.loc[0, "bic"])
        return f"{bic_value:,.0f}"
    except Exception:
        return "?"


def show_model_comparison_table(
    metrics_dir: Path, dep_name: str, title: str, group_kind: str | None = None
) -> None:
    """Display a BIC-based model comparison table for a family of LMMs.

    Parameters
    ----------
    metrics_dir : Path
        Directory containing one or more ``*_metrics.csv`` files produced by the
        LMM scripts.
    dep_name : str
        Dependent variable name (e.g. "valence", "time", "PC1"). Only metrics
        files whose stem ends with this name are included.
    title : str
        Title shown above the comparison table in the dashboard.
    group_kind : {"all_data", "ie_blocks", None}, optional
        If provided, restrict the table to models fitted on the full dataset
        ("all_data") or only on Inclusion/Exclusion blocks ("ie_blocks"). If
        ``None``, all models are included in a single table.
    """
    if not metrics_dir.exists():
        st.info(f"Metrics directory not found: {metrics_dir}")
        return

    # Mapping from short model names (used in filenames) to RHS formulas and
    # whether they are fitted on all data or only on inclusion/exclusion blocks.
    model_rhs: dict[str, str] = {
        # Multidimensional & PCA: group / time-on-task models (all data)
        "group_effect": "group",
        "time_on_task_effect": "time_on_task",
        "group_time_additive": "group + time_on_task",
        "group_time_interaction": "group * time_on_task",
        # Inclusion/Exclusion main and interaction effects
        "inclusion_exclusion_effect": "inclusion_exclusion",
        "group_ie_interaction": "group * inclusion_exclusion",
        # Distance / intervention models (IE blocks)
        "ie_intervention_distance": "inclusion_exclusion + probe_number",
        "group_ie_plus_distance": "group * inclusion_exclusion + probe_number",
        "ie_distance_interaction": "inclusion_exclusion * probe_number",
        "three_way_with_distance": "group * inclusion_exclusion * probe_number",
        "group_distance_plus_ie": "group * probe_number + inclusion_exclusion",
        # PCA-specific IE × time-on-task models
        "ie_time_additive": "inclusion_exclusion + time_on_task",
        "full_model_with_time": "group * inclusion_exclusion + time_on_task",
        "ie_time_interaction": "inclusion_exclusion * time_on_task",
        "three_way_interaction": "group * inclusion_exclusion * time_on_task",
        "group_time_int_plus_ie": "group * time_on_task + inclusion_exclusion",
    }

    model_group: dict[str, str] = {
        # Models using all available (off-task) data across sessions
        "group_effect": "all_data",
        "time_on_task_effect": "all_data",
        "group_time_additive": "all_data",
        "group_time_interaction": "all_data",
        # Models restricted to inclusion/exclusion blocks
        "inclusion_exclusion_effect": "ie_blocks",
        "group_ie_interaction": "ie_blocks",
        "ie_intervention_distance": "ie_blocks",
        "group_ie_plus_distance": "ie_blocks",
        "ie_distance_interaction": "ie_blocks",
        "three_way_with_distance": "ie_blocks",
        "group_distance_plus_ie": "ie_blocks",
        "ie_time_additive": "ie_blocks",
        "full_model_with_time": "ie_blocks",
        "ie_time_interaction": "ie_blocks",
        "three_way_interaction": "ie_blocks",
        "group_time_int_plus_ie": "ie_blocks",
    }

    def _classify_model(name: str) -> tuple[str, str]:
        """Return (group_kind, rhs_formula) for a given short model name."""

        # Prefer explicit mapping when available
        rhs = model_rhs.get(name)
        group_kind = model_group.get(name)

        # Fallback heuristics for any additional models
        if group_kind is None:
            if (
                "inclusion_exclusion" in name
                or "ie_" in name
                or name.startswith("ie")
                or "distance" in name
                or "three_way" in name
            ):
                group_kind = "ie_blocks"
            else:
                group_kind = "all_data"

        if rhs is None:
            # As a conservative fallback, keep the short name as RHS so that the
            # table still renders; this should be rare.
            rhs = name

        return group_kind, rhs

    rows: list[dict[str, object]] = []
    for metrics_path in sorted(metrics_dir.glob("*_metrics.csv")):
        stem = metrics_path.stem  # e.g. 'group_time_interaction_valence_metrics'
        if stem.endswith("_metrics"):
            stem = stem[: -len("_metrics")]

        # Accept either '<model>_<dep_name>' or '<model>_<dep_name>_normalized'
        if stem.endswith(f"_{dep_name}_normalized"):
            model_name = stem[: -len(f"_{dep_name}_normalized")]
        elif stem.endswith(f"_{dep_name}"):
            model_name = stem[: -len(f"_{dep_name}")]
        else:
            continue

        try:
            df_metrics = pd.read_csv(metrics_path)
        except Exception:
            continue
        if df_metrics.empty or "bic" not in df_metrics.columns:
            continue

        try:
            bic_val = float(df_metrics.loc[0, "bic"])
        except Exception:
            continue

        # Classify this model into a family ("all_data" vs "ie_blocks") and
        # retrieve its RHS formula. Use a separate local name (model_family) so
        # that we do not shadow the function argument ``group_kind`` used for
        # filtering below.
        model_family, rhs = _classify_model(model_name)
        equation = f"{dep_name} ~ {rhs}"

        rows.append(
            {
                "model_name": model_name,
                "group_kind": model_family,
                "equation": equation,
                "bic": bic_val,
            }
        )

    if not rows:
        st.info(f"No LMM metrics with BIC found for '{dep_name}' in {metrics_dir}")
        return

    df_models = pd.DataFrame(rows)

    def _highlight_best(row: pd.Series) -> list[str]:
        if row.get("delta_bic", 1.0) == 0:
            style = "background-color: #264653; color: white; font-weight: bold"
            return [style] * len(row)
        return [""] * len(row)

    st.subheader(title)

    # Optionally restrict to one family of models
    if group_kind is not None:
        df_sub = df_models[df_models["group_kind"] == group_kind].copy()
    else:
        df_sub = df_models.copy()

    if df_sub.empty:
        st.info("No models found for the requested family.")
        return

    best_bic = df_sub["bic"].min()
    df_sub["delta_bic"] = df_sub["bic"] - best_bic

    styled = (
        df_sub[["equation", "bic", "delta_bic"]]
        .sort_values("bic")
        .style.format({"bic": "{:,.0f}", "delta_bic": "{:,.1f}"})
        .apply(_highlight_best, axis=1)
    )

    # width="stretch" replaces deprecated use_container_width=True
    st.dataframe(styled, width="stretch")


def show_pairwise_if_significant(
    results_path: Path,
    pairwise_path: Path,
    title: str,
    group_prefix: str = "group_bdi3",
    alpha: float = 0.05,
) -> None:
    """Show BDI-split pairwise group comparisons only if group effects are present.

    Parameters
    ----------
    results_path : Path
        CSV with LMM fixed-effect results for the group model.
    pairwise_path : Path
        CSV with pairwise t-tests between BDI-split groups.
    title : str
        Title for the displayed table.
    group_prefix : str
        Prefix used to identify group-related predictors in the LMM results.
    alpha : float
        Significance threshold on the uncorrected *p*-values.
    """
    if not results_path.exists() or not pairwise_path.exists():
        return

    try:
        df_res = pd.read_csv(results_path)
    except Exception:
        return

    if "predictor" not in df_res.columns or "p_value" not in df_res.columns:
        return

    mask = df_res["predictor"].astype(str).str.contains(group_prefix)
    if not mask.any():
        return

    if not (df_res.loc[mask, "p_value"] < alpha).any():
        return

    safe_table(pairwise_path, title=title)


# =============================================================================
# PAGE RENDERING FUNCTIONS
# =============================================================================

# 1) Demographics rainclouds + group comparisons

def page_demographics() -> None:
    st.header("1. Demographics & Psychometrics")

    st.markdown(
        textwrap.dedent(
            """
            This page summarizes how the two groups differ in basic demographics and
            psychometric scales derived from the baseline questionnaires.
            """
        )
    )

    # Plots
    st.subheader("Raincloud plots by domain")
    safe_image(DEMOGRAPHICS_PLOTS_DIR / "raincloud_grid_demographics.png", "Demographic variables")
    safe_image(DEMOGRAPHICS_PLOTS_DIR / "raincloud_grid_psychometrics.png", "Psychometric scales")

    # Stats table
    safe_table(DEMOGRAPHICS_STATS_CSV, title="Group comparison statistics (demographics & psychometrics)")

    st.subheader("Interpretation")
    st.markdown(
        textwrap.dedent(
            """
            These comparisons show whether Controls and the Risk of Depression (RoD)
            group differ in age, gender distribution, and a range of psychometric
            scales (BDI, RRS, MWQ, self-esteem, CTQ, etc.). Significant variables in
            the raincloud plots are marked with an asterisk, and the table provides
            exact effect sizes (Cohen's *d*) and *p*-values for each test.

            In this sample, RoD participants score markedly higher on depressive
            symptoms and rumination: for instance, BDI is roughly **4–5 times
            higher** in RoD than Controls (mean ≈ 11.8 vs 2.7; *p* ≈ .001; *d* ≈
            −1.9), and total RRS scores are also strongly elevated (mean ≈ 47.6
            vs 33.0; *p* < .001; *d* ≈ −1.5). Trait mind-wandering (MWQ) follows
            a similar pattern (mean ≈ 19.7 vs 11.5; *p* < .001; *d* ≈ −1.7),
            whereas self-esteem is lower in RoD (mean ≈ 29.2 vs 34.6; *p* ≈ .014;
            *d* ≈ 1.1). Childhood adversity (CTQ subscales) tends to be higher in
            RoD but with more modest and often non-significant effects.

            Overall, these background measures confirm that the RoD group carries
            a substantially higher **affective and cognitive load** (symptoms,
            rumination, low self-esteem), which provides important context for
            interpreting the task-based mind-wandering and mediation results on
            the following pages.
            """
        )
    )


# 2) Probe Analysis – Multidimensional (valence, time, self/other, confidence)

def page_probe_multidim() -> None:
    st.header("2. Multidimensional Probe Analysis (Mind-Wandering Phenomenology)")

    st.markdown(
        textwrap.dedent(
            """
            This page summarizes the multidimensional linear mixed‑model
            analyses of mind-wandering episodes along four phenomenological
            dimensions: **valence**, **time**, **self/other** and **confidence**.

            **Analysis overview**  
            We fitted several linear mixed-effects models to each dimension,
            comparing the effects of Group (Controls vs RoD), Time-on-Task and
            Cyberball Inclusion/Exclusion. Model comparison was used to identify
            which factors best explain variation in phenomenology for each
            dimension. This allows us to ask whether group differences, temporal
            trends, social context, or their interactions drive the observed
            patterns in mind-wandering experience.

            Analyses are restricted to **off-task states**, defined as probes
            with ON/OFF < 50, to isolate genuine mind-wandering rather than
            task-focused cognition.

            **Analysis overview**  
            For each dimension we fitted several linear mixed-effects models
            including combinations of Group (Controls vs RoD), Time-on-Task and
            Cyberball Inclusion/Exclusion. Model comparison (via BIC and
            significance of key terms) was used to identify, for each
            dimension, whether group differences, temporal trends, social
            context, or their interactions best explain variation in
            phenomenology.
            """
        )
    )
    multidim_dir = (
        PROBE_DATA_DIR
        / "lmm_analysis_multidim"
        / "onoff_lt50"
        / "no_baseline"
        / "no_baseline"
    )

    st.subheader("Comprehensive analysis: Valence")
    valence_dir = multidim_dir / "valence"
    safe_image(
        valence_dir / "valence_comprehensive_analysis.png",
        "Valence dimension in off-task episodes",
    )
    st.markdown("### Valence – models using full dataset (all sessions)")
    show_model_comparison_table(
        valence_dir,
        "valence",
        "Model comparison – Valence (full dataset, off-task)",
        group_kind="all_data",
    )
    bic_val_group = load_bic_from_metrics(
        valence_dir / "group_effect_valence_metrics.csv"
    )
    safe_table(
        valence_dir / "group_effect_valence_results.csv",
        title=(
            f"LMM results – Valence (Group model, all SART sessions; BIC≈{bic_val_group})"
        ),
    )
    st.markdown("### Valence – models restricted to Inclusion/Exclusion blocks")
    bic_val_ie = load_bic_from_metrics(
        valence_dir / "inclusion_exclusion_effect_valence_normalized_metrics.csv"
    )
    show_model_comparison_table(
        valence_dir,
        "valence",
        "Model comparison – Valence (Inclusion/Exclusion blocks, off-task)",
        group_kind="ie_blocks",
    )
    safe_table(
        valence_dir / "inclusion_exclusion_effect_valence_normalized_results.csv",
        title=(
            "LMM results – Valence (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_val_ie})"
        ),
    )
    st.caption(
        "Valence: a simple Group effect model best explains off-task valence. "
        "Controls report moderately positive thoughts (mean ≈ 59), whereas RoD "
        "participants report much more negative content (mean ≈ 48), a "
        "difference of roughly 11 points, with the largest gap during "
        "Cyberball exclusion."
    )

    st.subheader("Comprehensive analysis: Time")
    time_dir = multidim_dir / "time"
    safe_image(
        time_dir / "time_comprehensive_analysis.png",
        "Time dimension (past–future) in off-task episodes",
    )
    bic_time_group = load_bic_from_metrics(
        time_dir / "group_effect_time_metrics.csv"
    )
    st.markdown("### Time – models using full dataset (all sessions)")
    show_model_comparison_table(
        time_dir,
        "time",
        "Model comparison – Time (full dataset, off-task)",
        group_kind="all_data",
    )
    safe_table(
        time_dir / "group_effect_time_results.csv",
        title=(
            f"LMM results – Time (Group model, all SART sessions; BIC≈{bic_time_group})"
        ),
    )
    st.markdown("### Time – models restricted to Inclusion/Exclusion blocks")
    bic_time_ie = load_bic_from_metrics(
        time_dir / "inclusion_exclusion_effect_time_normalized_metrics.csv"
    )
    show_model_comparison_table(
        time_dir,
        "time",
        "Model comparison – Time (Inclusion/Exclusion blocks, off-task)",
        group_kind="ie_blocks",
    )
    safe_table(
        time_dir / "inclusion_exclusion_effect_time_normalized_results.csv",
        title=(
            "LMM results – Time (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_time_ie})"
        ),
    )
    st.caption(
        "Time: both groups show a gradual drift toward more past-oriented "
        "thoughts over the session (time-on-task effect). RoD participants "
        "start slightly more past-focused and remain so throughout, with only "
        "modest modulation by Cyberball."
    )

    st.subheader("Comprehensive analysis: Self/Other")
    selfother_dir = multidim_dir / "selfother"
    safe_image(
        selfother_dir / "selfother_comprehensive_analysis.png",
        "Self/Other focus in off-task episodes",
    )
    bic_self_group = load_bic_from_metrics(
        selfother_dir / "group_effect_selfother_metrics.csv"
    )
    st.markdown("### Self/Other – models using full dataset (all sessions)")
    show_model_comparison_table(
        selfother_dir,
        "selfother",
        "Model comparison – Self/Other (full dataset, off-task)",
        group_kind="all_data",
    )
    safe_table(
        selfother_dir / "group_effect_selfother_results.csv",
        title=(
            "LMM results – Self/Other (Group model, all SART sessions; "
            f"BIC≈{bic_self_group})"
        ),
    )
    st.markdown("### Self/Other – models restricted to Inclusion/Exclusion blocks")
    bic_self_ie = load_bic_from_metrics(
        selfother_dir / "inclusion_exclusion_effect_selfother_normalized_metrics.csv"
    )
    show_model_comparison_table(
        selfother_dir,
        "selfother",
        "Model comparison – Self/Other (Inclusion/Exclusion blocks, off-task)",
        group_kind="ie_blocks",
    )
    safe_table(
        selfother_dir / "inclusion_exclusion_effect_selfother_normalized_results.csv",
        title=(
            "LMM results – Self/Other (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_self_ie})"
        ),
    )
    st.caption(
        "Self/Other: mind-wandering is predominantly self-focused in both "
        "groups, with a general time-on-task trend toward more self-referential "
        "content. RoD participants tend to be slightly more self-focused, "
        "especially in exclusion blocks, but differences are smaller than for "
        "valence."
    )

    st.subheader("Comprehensive analysis: Confidence")
    confidence_dir = multidim_dir / "confidence"
    safe_image(
        confidence_dir / "confidence_comprehensive_analysis.png",
        "Confidence ratings in off-task episodes",
    )
    bic_conf_group = load_bic_from_metrics(
        confidence_dir / "group_effect_confidence_metrics.csv"
    )
    st.markdown("### Confidence – models using full dataset (all sessions)")
    show_model_comparison_table(
        confidence_dir,
        "confidence",
        "Model comparison – Confidence (full dataset, off-task)",
        group_kind="all_data",
    )
    safe_table(
        confidence_dir / "group_effect_confidence_results.csv",
        title=(
            "LMM results – Confidence (Group model, all SART sessions; "
            f"BIC≈{bic_conf_group})"
        ),
    )
    st.markdown("### Confidence – models restricted to Inclusion/Exclusion blocks")
    bic_conf_ie = load_bic_from_metrics(
        confidence_dir / "inclusion_exclusion_effect_confidence_normalized_metrics.csv"
    )
    show_model_comparison_table(
        confidence_dir,
        "confidence",
        "Model comparison – Confidence (Inclusion/Exclusion blocks, off-task)",
        group_kind="ie_blocks",
    )
    safe_table(
        confidence_dir / "inclusion_exclusion_effect_confidence_normalized_results.csv",
        title=(
            "LMM results – Confidence (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_conf_ie})"
        ),
    )
    st.subheader("Interpretation")
    st.markdown(
        textwrap.dedent(
            """
            **Summary across dimensions**  
            - **Valence**: RoD participants consistently report more negative
              thoughts than Controls (≈ 11 points lower on average), especially
              during Cyberball exclusion, indicating a robust negative bias in
              off-task experience.
            - **Time**: both groups drift toward more past-oriented thoughts over
              time, with RoD starting slightly more past-focused and remaining so.
            - **Self/Other**: mind-wandering is predominantly self-focused in
              both groups. RoD participants are marginally more self-focused, but
              differences are smaller than for valence.
            - **Confidence**: RoD participants show lower and more variable
              confidence, particularly under exclusion, whereas Controls maintain
              high confidence throughout.

            Together, these models suggest that risk for depression is expressed
            not only in **how often** participants mind-wander, but also in the
            **quality of off-task experience**: more negative, slightly more
            past- and self-focused, and accompanied by reduced confidence.
            """
        )
    )
    st.caption(
        "Confidence: Controls maintain relatively high confidence in their "
        "ratings across the session, whereas RoD participants show lower and "
        "more variable confidence, with a noticeable drop during Cyberball "
        "exclusion and only partial recovery during inclusion."
    )


# 3) Probe Analysis – ON/OFF (clean LMMs + key plots)

def page_probe_onoff() -> None:
    st.header("3. Probe Analysis – ON/OFF (Sustained Attention & Cyberball)")

    st.markdown(
        textwrap.dedent(
            """
            This page summarizes the ON/OFF analyses that index **sustained
            attention** and its modulation by **social inclusion/exclusion**.

            **Analysis overview**  
            We fitted a family of linear mixed-effects models to the ON/OFF
            ratings, comparing the effects of Group, Time-on-Task and Cyberball
            Inclusion/Exclusion as well as their interactions. Model comparison
            focused on which combinations of these factors best explain variance
            in ON/OFF, yielding a time-on-task model with a strong Group × Time
            interaction and, separately, a Cyberball model with a robust Group ×
            Inclusion/Exclusion interaction.

            The ON/OFF models therefore address three questions:

            1. Do Controls and Risk‑of‑Depression (RoD) participants differ in
               overall ON/OFF levels?
            2. Does Cyberball inclusion vs exclusion modulate ON/OFF ratings
               within each group?
            3. How do ON/OFF ratings change across the 60 probes (time‑on‑task)
               as a function of group?

            Models that include Cyberball use only the SART blocks immediately
            surrounding the social intervention, whereas the time‑on‑task models
            use all four SART runs across the session. Best models are selected
            **within** each subset, so that Cyberball-related conclusions are
            based on the social-manipulation blocks and time-related conclusions
            on the full longitudinal dataset.
            """
        )
    )
    onoff_plots_dir = PROBE_DATA_DIR / "lmm_plots_onoff"

    st.subheader("Overall ON/OFF effects")
    safe_image(
        onoff_plots_dir / "onoff_comprehensive_analysis.png",
        "Comprehensive ON/OFF analysis (group, time-on-task, inclusion/exclusion)",
    )
    safe_table(
        onoff_plots_dir / "onoff_descriptive_statistics.csv",
        title="ON/OFF descriptive statistics (all probes)",
    )
    # LMM tables for the two key best models
    onoff_lmm_dir = PROBE_DATA_DIR / "lmm_analysis_onoff"
    bic_time = load_bic_from_metrics(
        onoff_lmm_dir / "group_time_interaction_onoff_metrics.csv"
    )
    bic_ie = load_bic_from_metrics(
        onoff_lmm_dir / "group_ie_interaction_onoff_normalized_metrics.csv"
    )
    safe_table(
        onoff_lmm_dir / "group_time_interaction_onoff_results.csv",
        title=f"LMM results – ON/OFF ~ Group × Time-on-Task (BIC≈{bic_time})",
    )
    safe_table(
        onoff_lmm_dir / "group_ie_interaction_onoff_normalized_results.csv",
        title=f"LMM results – ON/OFF ~ Group × Inclusion/Exclusion (BIC≈{bic_ie})",
    )

    st.subheader("Time-on-task trajectories")
    safe_image(
        onoff_plots_dir / "onoff_time_on_task_analysis.png",
        "ON/OFF trajectories across probes by group",
    )
    st.caption(
        "Time-on-task: both groups show increasing mind-wandering as the session "
        "progresses, but the Risk-of-Depression group deteriorates more steeply, "
        "leading to a substantial group difference in task engagement by the end "
        "of the experiment."
    )


# 4) PCA Analysis

def page_pca() -> None:
    st.header("4. PCA of Thought Content")

    st.markdown(
        textwrap.dedent(
            """
            This page summarizes the principal components analysis (PCA) of the
            phenomenological ratings obtained at each thought probe (valence,
            temporal focus and self/other). The aim is to reduce these correlated
            dimensions to a small set of latent components (PC1–PC3) that capture
            the main structure of ongoing experience and can then be entered as
            predictors in the mixed models.

            **Analysis overview**  
            We applied PCA to the covariance between valence, time and self/other
            ratings, and then fitted linear mixed models to each component,
            comparing the effects of Group, Time-on-Task and Cyberball
            Inclusion/Exclusion as well as their interactions. This allows us to
            ask whether latent thought patterns (e.g. a valence/rumination
            component) differ between groups and how they are shaped by fatigue
            and social context.
            """
        )
    )
    st.subheader("Overall PCA summary")
    safe_image(
        PROBE_DATA_DIR / "pca_scree_plot.png",
        "Scree plot: proportion of variance explained by each component",
    )
    st.caption(
        "Stats: PC1≈42.1% var, PC2≈33.6%, PC3≈24.3% (three components explain "
        "~100% of variance)."
    )
    safe_image(
        PROBE_DATA_DIR / "pca_biplot.png",
        "Biplot: loadings of valence, time and self/other on PC1–PC2",
    )
    st.caption(
        "Loadings: PC1 loads strongly on Valence (~0.80) and also on Self/Other "
        "(~0.57) and Time (~0.55); PC2 loads on Time (~0.72) and negatively on "
        "Self/Other (~−0.70), with near‑zero Valence loading."
    )
    safe_image(
        PROBE_DATA_DIR / "pca_correlation_heatmap.png",
        "Correlation matrix between original dimensions and components",
    )
    st.caption(
        "Probe-level correlations: PC1–Valence r≈0.74, PC1–Time r≈0.49, "
        "PC1–Self/Other r≈0.58; PC2–Time r≈0.71, PC2–Self/Other r≈−0.67; "
        "PC3–Valence r≈0.56, PC3–Time r≈−0.37, PC3–Self/Other r≈−0.42."
    )

    st.subheader("Group, inclusion/exclusion and time-on-task effects on PCs")
    pca_lmm_dir = PROBE_DATA_DIR / "lmm_analysis_pca"
    safe_table(
        pca_lmm_dir / "descriptive_statistics.csv",
        "Descriptive statistics of PCA scores (PC1–PC3)",
    )
    safe_image(
        pca_lmm_dir / "PC1_comprehensive_analysis.png",
        "PC1 – comprehensive LMM summary (group, inclusion/exclusion, time-on-task)",
    )
    st.caption(
        "PC1 (rumination axis): best model Group×Inclusion/Exclusion (Sart2 & Sart4), "
        "BIC≈1,297; RoD vs Controls in exclusion β≈−0.80, p≈.003; Inclusion main "
        "effect β≈−0.18 (ns); Group×Inclusion β≈+0.69, p<.001 (inclusion "
        "substantially reduces rumination in RoD)."
    )
    st.markdown(
        textwrap.dedent(
            """
            **PC1 – Best models and interpretation**  
            PC1 captures a **valence/rumination axis**, loading strongly on
            negative valence and, to a lesser extent, on self-focused and
            temporally extended thoughts. The best-fitting model includes a
            **Group × Inclusion/Exclusion interaction**. In exclusion blocks, RoD
            participants show markedly higher PC1 scores than Controls
            (β≈−0.80 for RoD vs Controls in baseline-corrected units, *p*≈.003),
            consistent with more negative, ruminative content. During inclusion,
            PC1 scores decrease selectively in RoD (interaction β≈+0.69,
            *p*<.001), bringing them much closer to Controls.

            Thus, PC1 shows that **social context strongly modulates latent
            rumination-like patterns** in RoD participants: exclusion amplifies,
            whereas inclusion attenuates, these negative/self-focused modes of
            thinking.
            """
        )
    )
    # PC1 – model comparison and best models
    st.markdown("### PC1 – models using full dataset (all sessions)")
    show_model_comparison_table(
        pca_lmm_dir,
        "PC1",
        "Model comparison – PC1 (full dataset)",
        group_kind="all_data",
    )
    bic_pc1_group = load_bic_from_metrics(
        pca_lmm_dir / "group_effect_PC1_metrics.csv"
    )
    safe_table(
        pca_lmm_dir / "group_effect_PC1_results.csv",
        title=(
            "LMM results – PC1 (Group model across all sessions; "
            f"BIC≈{bic_pc1_group})"
        ),
    )
    st.markdown("### PC1 – models restricted to Inclusion/Exclusion blocks")
    bic_pc1_ie = load_bic_from_metrics(
        pca_lmm_dir / "group_ie_interaction_PC1_metrics.csv"
    )
    show_model_comparison_table(
        pca_lmm_dir,
        "PC1",
        "Model comparison – PC1 (Inclusion/Exclusion blocks)",
        group_kind="ie_blocks",
    )
    safe_table(
        pca_lmm_dir / "group_ie_interaction_PC1_results.csv",
        title=(
            "LMM results – PC1 (Group × Inclusion/Exclusion model; "
            f"BIC≈{bic_pc1_ie})"
        ),
    )
    safe_image(
        pca_lmm_dir / "PC2_comprehensive_analysis.png",
        "PC2 – comprehensive LMM summary (group, inclusion/exclusion, time-on-task)",
    )
    st.caption(
        "PC2 (temporal–social axis): best model Group×Time-on-Task (all sessions), "
        "BIC≈2,402; Time β≈0.00 (ns) for Controls; Group×Time β≈−0.012, p≈.001 "
        "(RoD drifts toward lower PC2 scores—more past- and other-focused—over "
        "time). No clear Cyberball effects."
    )
    st.markdown(
        textwrap.dedent(
            """
            **PC2 – Best models and interpretation**  
            PC2 reflects a **temporal–social axis**, contrasting more future- and
            other-focused thoughts with more past- and self-focused ones. The
            best model includes a **Group × Time-on-Task interaction** across all
            sessions. For Controls, PC2 remains relatively stable over time,
            whereas RoD participants show a significant negative drift in PC2
            (Group × Time β≈−0.012, *p*≈.001).

            This indicates that, as the experiment progresses, RoD participants'
            thoughts become increasingly **past-oriented and other-focused**
            relative to Controls, consistent with a gradual shift toward more
            ruminative, socially tinged content rather than task-related future
            planning. Cyberball has no strong additional impact on PC2 beyond
            this temporal trend.
            """
        )
    )
    # PC2 – model comparison and best models
    st.markdown("### PC2 – models using full dataset (all sessions)")
    show_model_comparison_table(
        pca_lmm_dir,
        "PC2",
        "Model comparison – PC2 (full dataset)",
        group_kind="all_data",
    )
    bic_pc2_group = load_bic_from_metrics(
        pca_lmm_dir / "group_time_interaction_PC2_metrics.csv"
    )
    safe_table(
        pca_lmm_dir / "group_time_interaction_PC2_results.csv",
        title=(
            "LMM results – PC2 (Group × Time-on-Task model; "
            f"BIC≈{bic_pc2_group})"
        ),
    )
    st.markdown("### PC2 – models restricted to Inclusion/Exclusion blocks")
    bic_pc2_ie = load_bic_from_metrics(
        pca_lmm_dir / "inclusion_exclusion_effect_PC2_normalized_metrics.csv"
    )
    show_model_comparison_table(
        pca_lmm_dir,
        "PC2",
        "Model comparison – PC2 (Inclusion/Exclusion blocks)",
        group_kind="ie_blocks",
    )
    safe_table(
        pca_lmm_dir / "inclusion_exclusion_effect_PC2_normalized_results.csv",
        title=(
            "LMM results – PC2 (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_pc2_ie})"
        ),
    )
    safe_image(
        pca_lmm_dir / "PC3_comprehensive_analysis.png",
        "PC3 – comprehensive LMM summary (group, inclusion/exclusion, time-on-task)",
    )
    st.caption(
        "PC3 (residual component): Cyberball model with Inclusion/Exclusion only, "
        "BIC≈1,115; Inclusion β≈+0.24, p≈.001 (both groups increase PC3 during "
        "inclusion). Across all sessions, a simple Group effect model, BIC≈2,092; "
        "RoD vs Controls β≈−0.35, p≈.024, indicating a modest overall deficit."
    )
    st.markdown(
        textwrap.dedent(
            """
            **PC3 – Best models and interpretation**  
            PC3 captures residual variance not explained by the first two
            components, but still shows systematic effects. A simple **group
            effect** model across all sessions indicates slightly lower PC3
            scores in RoD than Controls (β≈−0.35, *p*≈.024), suggesting a modest
            overall deficit on this residual dimension.

            When focusing on Cyberball blocks, a model with **Inclusion/Exclusion
            only** fits best: inclusion reliably increases PC3 scores in both
            groups (β≈+0.24, *p*≈.001), consistent with a general improvement in
            this residual component under socially supportive conditions. These
            effects are smaller than those seen for PC1 but indicate that even
            secondary aspects of thought are sensitive to the social context.
            """
        )
    )
    # PC3 – model comparison and best models
    st.markdown("### PC3 – models using full dataset (all sessions)")
    show_model_comparison_table(
        pca_lmm_dir,
        "PC3",
        "Model comparison – PC3 (full dataset)",
        group_kind="all_data",
    )
    bic_pc3_group = load_bic_from_metrics(
        pca_lmm_dir / "group_effect_PC3_metrics.csv"
    )
    safe_table(
        pca_lmm_dir / "group_effect_PC3_results.csv",
        title=(
            "LMM results – PC3 (Group model across all sessions; "
            f"BIC≈{bic_pc3_group})"
        ),
    )
    st.markdown("### PC3 – models restricted to Inclusion/Exclusion blocks")
    bic_pc3_ie = load_bic_from_metrics(
        pca_lmm_dir / "inclusion_exclusion_effect_PC3_normalized_metrics.csv"
    )
    show_model_comparison_table(
        pca_lmm_dir,
        "PC3",
        "Model comparison – PC3 (Inclusion/Exclusion blocks)",
        group_kind="ie_blocks",
    )
    safe_table(
        pca_lmm_dir / "inclusion_exclusion_effect_PC3_normalized_results.csv",
        title=(
            "LMM results – PC3 (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_pc3_ie})"
        ),
    )

    st.subheader("Interpretation")
    st.markdown(
        textwrap.dedent(
            """
            - The **scree plot** shows that the first two components (PC1 and PC2)
            capture most of the variance, so they already summarize the main
            structure of the thought-content space.
            - In the **biplot**, arrows indicate how the original dimensions
            project onto these components. For example, if *valence* loads strongly
            on PC1, that axis can be interpreted as a continuum from more positive
            to more negative thoughts.
            - The other dimensions (time, self/other) typically organize PC2 and
            PC3, separating experiences that are more past- vs future-oriented or
            more self- vs other-focused.

            These components are then used in the mixed models to ask whether
            certain **latent patterns of experience** (e.g. negative,
            past-oriented, self-focused thoughts) are more common in the
            Risk-of-Depression group or are modulated by Cyberball.
            """
        )
    )


# 5) Correlation & Partial Correlation Analyses

def page_correlations() -> None:
    st.header("5. Correlations Between Mind-Wandering, PCs and Psychometrics")

    st.markdown(
        textwrap.dedent(
            """
            This page summarizes the correlation analyses that link three
            levels of information:

            - **Probe-level variables** (ON/OFF, valence, time, self/other,
              confidence),
            - **PCA components** (PC1–PC3),
            - **Psychometric scales** (BDI, RRS, MWQ, FNE, self-esteem, CTQ,
              ARSQ, SRIS, etc.). The matrices show both simple Spearman
              correlations and partial correlations (controlling for other
              thought dimensions).

            **Analysis overview**  
            For each pair of variables, we computed Spearman correlations at the
            probe level and corrected *p*-values using FDR across rows. We also
            estimated partial correlations that remove shared variance with other
            dimensions (e.g. ON/OFF or PC scores) to isolate more specific
            links. This reveals which aspects of ongoing experience and which
            latent components are most closely tied to trait measures of mood
            and cognition.
            """
        )
    )
    st.subheader("Probe-level correlations with clinical scales")
    corr_dir = PROBE_DATA_DIR / "final_correlation_analysis"
    safe_image(
        corr_dir / "final_correlation_heatmap.png",
        "Spearman correlations between probe-level variables/PCs and clinical scales",
    )
    safe_table(
        corr_dir / "final_correlation_results.csv",
        "Final correlation results",
    )

    partial_corr_dir = PROBE_DATA_DIR / "partial_correlation_analysis"
    if partial_corr_dir.exists():
        st.subheader("Partial correlations (controlling for other dimensions)")
        safe_image(
            partial_corr_dir / "partial_correlation_heatmap.png",
            "Partial correlation heatmap",
        )
        st.caption(
            "Partial correlations: 107 coefficients estimated while controlling "
            "for other dimensions; a subset remain significant after FDR, "
            "highlighting specific links between thought patterns and clinical "
            "scales beyond shared variance."
        )
        safe_table(
            partial_corr_dir / "partial_correlation_results.csv",
            "Partial correlation results",
        )
    else:
        st.info(
            "Partial correlation outputs not found. Please generate the "
            "partial-correlation results before using this page.",
        )

    st.subheader("How to read these matrices")
    st.markdown(
        textwrap.dedent(
            """
            - The **raw correlation heatmap** shows which aspects of experience
            (e.g. more negative or more self-focused thoughts) are associated with
            each clinical scale. This is useful for seeing the overall pattern of
            relationships.
            - The **partial correlation heatmap** isolates more specific
            relationships, controlling for other dimensions. For example, it lets
            you ask whether the link between PC1 and BDI remains once the average
            ON/OFF level is controlled for.

            In reports, you can highlight which thought patterns (or PCA
            components) are most tightly related to depressive symptoms or
            rumination, beyond simply spending more or less time off-task.
            """
        )
    )


# 6) BDI-Split Analysis

def page_bdi_split() -> None:
    st.header("6. BDI-Split Analysis Within Controls")

    st.markdown(
        textwrap.dedent(
            """
            This analysis takes the Control group and splits it into two
            subgroups based on the median BDI score within Controls:

            - **Controls – Low BDI**
            - **Controls – High BDI**

            The **Risk-of-Depression (RoD)** group is kept as a third level, and
            the multidimensional analyses (ON/OFF, valence, time, self/other,
            confidence and PCA components) are repeated across these three groups.
            This allows us to test whether the effects observed when comparing
            Controls vs RoD actually follow a graded pattern across increasing
            depressive symptom load.

            **Analysis overview**  
            We repeated the multidimensional analyses (ON/OFF, valence, time,
            self/other, confidence and PCA components) across the three groups
            (Low-BDI Controls, High-BDI Controls, RoD), comparing the effects of
            Group, Time-on-Task and Cyberball Inclusion/Exclusion as well as
            their interactions. This allows us to ask whether the observed
            patterns in mind-wandering experience follow a continuous gradient
            across increasing depressive symptom load.
            """
        )
    )
    st.subheader("Main figure: ON/OFF (full range)")
    safe_image(
        BDI_SPLIT_DIR
        / "lmm_plots_multidim"
        / "onoff_full_range"
        / "onoff"
        / "onoff_comprehensive_analysis.png",
        "ON/OFF over the full response range (three groups)",
    )
    # LMM table for ON/OFF (full range)
    bdi_onoff_lmm_dir = BDI_SPLIT_DIR / "lmm_analysis_multidim" / "onoff_full_range"
    bic_bdi_onoff = load_bic_from_metrics(
        bdi_onoff_lmm_dir / "onoff" / "group_effect_onoff_metrics.csv"
    )
    bic_bdi_onoff_ie = load_bic_from_metrics(
        bdi_onoff_lmm_dir / "onoff" / "inclusion_exclusion_effect_onoff_normalized_metrics.csv"
    )
    st.markdown("### ON/OFF – model comparison")
    show_model_comparison_table(
        bdi_onoff_lmm_dir / "onoff",
        "onoff",
        "Model comparison – ON/OFF (full range, all tested LMMs)",
    )
    st.markdown("#### Best model: three-level group effect (full range)")
    safe_table(
        bdi_onoff_lmm_dir / "onoff" / "group_effect_onoff_results.csv",
        title=(
            "LMM results – ON/OFF (full-range three-level group effect, "
            f"BIC≈{bic_bdi_onoff})"
        ),
    )
    show_pairwise_if_significant(
        bdi_onoff_lmm_dir / "onoff" / "group_effect_onoff_results.csv",
        bdi_onoff_lmm_dir / "onoff" / "pairwise_group_comparisons_onoff_overall.csv",
        title="Pairwise group comparisons – ON/OFF (BDI-split groups)",
    )
    st.markdown("#### Best model: Inclusion/Exclusion blocks (baseline-corrected)")
    safe_table(
        bdi_onoff_lmm_dir
        / "onoff"
        / "inclusion_exclusion_effect_onoff_normalized_results.csv",
        title=(
            "LMM results – ON/OFF (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_bdi_onoff_ie})"
        ),
    )
    st.caption(
        "ON/OFF (full range): Low-BDI Controls show the highest task engagement, "
        "High-BDI Controls are intermediate, and RoD participants display the "
        "lowest engagement, especially under exclusion. The best full-range "
        "three-level group model (BIC≈21,441) captures this graded pattern in "
        "sustained attention as depressive symptoms increase."
    )

    st.subheader("Main figure: Valence (off-task only)")
    safe_image(
        BDI_SPLIT_DIR
        / "lmm_plots_multidim"
        / "onoff_lt50"
        / "valence"
        / "valence_comprehensive_analysis.png",
        "Valence of off-task thoughts in the three subgroups",
    )
    # LMM table for Valence (off-task only)
    bdi_lt50_lmm_dir = BDI_SPLIT_DIR / "lmm_analysis_multidim" / "onoff_lt50"
    bic_bdi_val = load_bic_from_metrics(
        bdi_lt50_lmm_dir / "valence" / "group_effect_valence_metrics.csv"
    )
    bic_bdi_val_ie = load_bic_from_metrics(
        bdi_lt50_lmm_dir
        / "valence"
        / "inclusion_exclusion_effect_valence_normalized_metrics.csv"
    )
    st.markdown("### Valence – model comparison (off-task)")
    show_model_comparison_table(
        bdi_lt50_lmm_dir / "valence",
        "valence",
        "Model comparison – Valence (off-task, all tested LMMs)",
    )
    st.markdown("#### Best model: three-level group effect (off-task)")
    safe_table(
        bdi_lt50_lmm_dir / "valence" / "group_effect_valence_results.csv",
        title=(
            "LMM results – Valence (off-task three-level group effect, "
            f"BIC≈{bic_bdi_val})"
        ),
    )
    show_pairwise_if_significant(
        bdi_lt50_lmm_dir / "valence" / "group_effect_valence_results.csv",
        bdi_lt50_lmm_dir / "valence" / "pairwise_group_comparisons_valence_overall.csv",
        title="Pairwise group comparisons – Valence (BDI-split groups)",
    )
    st.markdown("#### Best model: Inclusion/Exclusion blocks (baseline-corrected)")
    safe_table(
        bdi_lt50_lmm_dir
        / "valence"
        / "inclusion_exclusion_effect_valence_normalized_results.csv",
        title=(
            "LMM results – Valence (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_bdi_val_ie})"
        ),
    )
    st.caption(
        "Valence (off-task only): for probes with ON/OFF < 50, Low-BDI Controls "
        "report the most positive thoughts, High-BDI Controls are more neutral, "
        "and RoD participants show the most negative content, especially during "
        "exclusion. The corresponding three-level group model (BIC≈7,150) "
        "quantifies this graded increase in negative off-task thought content "
        "with higher depressive symptoms."
    )

    st.subheader("Multidimensional dimensions (off-task only)")
    safe_image(
        BDI_SPLIT_DIR
        / "lmm_plots_multidim"
        / "onoff_lt50"
        / "time"
        / "time_comprehensive_analysis.png",
        "Time dimension in off-task episodes (BDI-split)",
    )
    time_bdi_dir = (
        BDI_SPLIT_DIR
        / "lmm_analysis_multidim"
        / "onoff_lt50"
        / "time"
    )
    safe_table(
        time_bdi_dir / "time_descriptive_statistics.csv",
        title="Time – descriptive statistics (BDI-split)",
    )
    bic_time_bdi_group = load_bic_from_metrics(
        time_bdi_dir / "group_effect_time_metrics.csv"
    )
    bic_time_bdi_ie = load_bic_from_metrics(
        time_bdi_dir / "inclusion_exclusion_effect_time_normalized_metrics.csv"
    )
    st.markdown("### Time – model comparison (off-task)")
    show_model_comparison_table(
        time_bdi_dir,
        "time",
        "Model comparison – Time (off-task, all tested LMMs)",
    )
    st.markdown("#### Best model: three-level group effect (off-task)")
    safe_table(
        time_bdi_dir / "group_effect_time_results.csv",
        title=(
            "LMM results – Time (three-level group effect, "
            f"BIC≈{bic_time_bdi_group})"
        ),
    )
    show_pairwise_if_significant(
        time_bdi_dir / "group_effect_time_results.csv",
        time_bdi_dir / "pairwise_group_comparisons_time_overall.csv",
        title="Pairwise group comparisons – Time (BDI-split groups)",
    )
    st.markdown("#### Best model: Inclusion/Exclusion blocks (baseline-corrected)")
    safe_table(
        time_bdi_dir / "inclusion_exclusion_effect_time_normalized_results.csv",
        title=(
            "LMM results – Time (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_time_bdi_ie})"
        ),
    )
    st.caption(
        "Time (off-task): group differences are modest, but RoD participants "
        "tend to report slightly more past-oriented thoughts overall, "
        "especially during exclusion, whereas Low- and High-BDI Controls hover "
        "closer to the mid-range."
    )
    safe_image(
        BDI_SPLIT_DIR
        / "lmm_plots_multidim"
        / "onoff_lt50"
        / "selfother"
        / "selfother_comprehensive_analysis.png",
        "Self/Other dimension in off-task episodes (BDI-split)",
    )
    self_bdi_dir = (
        BDI_SPLIT_DIR
        / "lmm_analysis_multidim"
        / "onoff_lt50"
        / "selfother"
    )
    safe_table(
        self_bdi_dir / "selfother_descriptive_statistics.csv",
        title="Self/Other – descriptive statistics (BDI-split)",
    )
    bic_self_bdi_group = load_bic_from_metrics(
        self_bdi_dir / "group_effect_selfother_metrics.csv"
    )
    bic_self_bdi_ie = load_bic_from_metrics(
        self_bdi_dir / "inclusion_exclusion_effect_selfother_normalized_metrics.csv"
    )
    st.markdown("### Self/Other – model comparison (off-task)")
    show_model_comparison_table(
        self_bdi_dir,
        "selfother",
        "Model comparison – Self/Other (off-task, all tested LMMs)",
    )
    st.markdown("#### Best model: three-level group effect (off-task)")
    safe_table(
        self_bdi_dir / "group_effect_selfother_results.csv",
        title=(
            "LMM results – Self/Other (three-level group effect, "
            f"BIC≈{bic_self_bdi_group})"
        ),
    )
    show_pairwise_if_significant(
        self_bdi_dir / "group_effect_selfother_results.csv",
        self_bdi_dir / "pairwise_group_comparisons_selfother_overall.csv",
        title="Pairwise group comparisons – Self/Other (BDI-split groups)",
    )
    st.markdown("#### Best model: Inclusion/Exclusion blocks (baseline-corrected)")
    safe_table(
        self_bdi_dir / "inclusion_exclusion_effect_selfother_normalized_results.csv",
        title=(
            "LMM results – Self/Other (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_self_bdi_ie})"
        ),
    )
    st.caption(
        "Self/Other (off-task): all three groups show predominantly "
        "self-focused mind-wandering. RoD participants remain the most "
        "self-focused, particularly during exclusion, whereas Low- and "
        "High-BDI Controls are slightly closer to the self/other midpoint."
    )
    safe_image(
        BDI_SPLIT_DIR
        / "lmm_plots_multidim"
        / "onoff_lt50"
        / "confidence"
        / "confidence_comprehensive_analysis.png",
        "Confidence dimension in off-task episodes (BDI-split)",
    )
    conf_bdi_dir = (
        BDI_SPLIT_DIR
        / "lmm_analysis_multidim"
        / "onoff_lt50"
        / "confidence"
    )
    safe_table(
        conf_bdi_dir / "confidence_descriptive_statistics.csv",
        title="Confidence – descriptive statistics (BDI-split)",
    )
    bic_conf_bdi_group = load_bic_from_metrics(
        conf_bdi_dir / "group_effect_confidence_metrics.csv"
    )
    bic_conf_bdi_ie = load_bic_from_metrics(
        conf_bdi_dir / "inclusion_exclusion_effect_confidence_normalized_metrics.csv"
    )
    st.markdown("### Confidence – model comparison (off-task)")
    show_model_comparison_table(
        conf_bdi_dir,
        "confidence",
        "Model comparison – Confidence (off-task, all tested LMMs)",
    )
    st.markdown("#### Best model: three-level group effect (off-task)")
    safe_table(
        conf_bdi_dir / "group_effect_confidence_results.csv",
        title=(
            "LMM results – Confidence (three-level group effect, "
            f"BIC≈{bic_conf_bdi_group})"
        ),
    )
    show_pairwise_if_significant(
        conf_bdi_dir / "group_effect_confidence_results.csv",
        conf_bdi_dir / "pairwise_group_comparisons_confidence_overall.csv",
        title="Pairwise group comparisons – Confidence (BDI-split groups)",
    )
    st.markdown("#### Best model: Inclusion/Exclusion blocks (baseline-corrected)")
    safe_table(
        conf_bdi_dir / "inclusion_exclusion_effect_confidence_normalized_results.csv",
        title=(
            "LMM results – Confidence (Inclusion/Exclusion model, baseline-corrected; "
            f"BIC≈{bic_conf_bdi_ie})"
        ),
    )
    st.caption(
        "Confidence (off-task): Low-BDI Controls report the highest confidence "
        "in their probe ratings, High-BDI Controls are intermediate, and RoD "
        "participants show generally lower and more variable confidence, "
        "particularly under inclusion where their scores are clearly reduced."
    )

    st.subheader("PCA components by BDI-split group")
    for pc in ("PC1", "PC2", "PC3"):
        img = (
            BDI_SPLIT_DIR
            / "lmm_plots_multidim"
            / "pca_components"
            / pc
            / f"{pc}_comprehensive_analysis.png"
        )
        safe_image(img, f"{pc}: thought-content pattern by BDI subgroup")

        # LMM group and inclusion/exclusion models per component
        pc_lmm_dir = (
            BDI_SPLIT_DIR
            / "lmm_analysis_multidim"
            / "pca_components"
            / pc
        )
        bic_group = load_bic_from_metrics(
            pc_lmm_dir / f"group_effect_{pc}_metrics.csv"
        )
        bic_ie = load_bic_from_metrics(
            pc_lmm_dir
            / f"inclusion_exclusion_effect_{pc}_normalized_metrics.csv"
        )
        safe_table(
            pc_lmm_dir / f"group_effect_{pc}_results.csv",
            title=(
                f"LMM results – {pc} (three-level group effect, BIC≈{bic_group})"
            ),
        )
        safe_table(
            pc_lmm_dir / f"inclusion_exclusion_effect_{pc}_normalized_results.csv",
            title=(
                f"LMM results – {pc} (Inclusion/Exclusion model, baseline-corrected; "
                f"BIC≈{bic_ie})"
            ),
        )
        show_model_comparison_table(
            pc_lmm_dir,
            pc,
            f"Model comparison – {pc} (off-task, all tested LMMs)",
        )
        show_pairwise_if_significant(
            pc_lmm_dir / f"group_effect_{pc}_results.csv",
            pc_lmm_dir / f"pairwise_group_comparisons_{pc}_overall.csv",
            title=f"Pairwise group comparisons – {pc} (BDI-split groups)",
        )

    st.subheader("Interpretation")
    st.markdown(
        textwrap.dedent(
            """
            Across ON/OFF, multidimensional dimensions and PCA components, the
            BDI-split analyses show a **graded pattern** rather than a simple
            dichotomy between Controls and RoD. Low-BDI Controls have the best
            sustained attention, the most positive and future-oriented thoughts,
            and the highest confidence. High-BDI Controls occupy an intermediate
            position on most measures, and RoD participants show the lowest
            engagement, most negative and self-focused content, and the lowest
            confidence.

            This gradient suggests that many of the effects seen when comparing
            Controls vs RoD are already present, in attenuated form, within the
            Control group as depressive symptoms increase. The corresponding
            group-effect models for PC1–PC3 (BICs ≈ 2,316, 2,391 and 2,098,
            respectively) quantify this pattern in the latent space: higher
            symptom load is associated with more negative and self-focused
            components and modest shifts on the residual component, even before
            crossing the threshold into the RoD group.
            """
        )
    )


# Cyberball Moderated Mediation Results Directory
CYBERBALL_MEDIATION_DIR = MEDIATION_DIR / "cyberball_moderated_mediation"


# 7–9) Mediation Analyses (Forward, Reverse, Comparative)

def page_mediation() -> None:
    st.header("Mediation Analyses (Mood ↔ Thought Content)")

    st.markdown(
        textwrap.dedent(
            """
            This section provides a visual summary of the multilevel mediation
            models that link **group**, **mood** and **thought content**. Only the
            most informative elements are shown:

            - The causal diagrams (DAGs) for each model.
            - The *combined* figures that integrate direct and indirect effects.

            **Analysis overview**  
            We estimated multilevel mediation models to examine the relationships
            between group, mood and thought content. In the forward models, group
            predicts block-level mood, which then predicts thought content
            (dimensions and PCA scores). In the reverse models, group predicts
            thought content, which in turn predicts subsequent mood. We focus on
            the size and confidence intervals of the indirect effects (*a×b*) to
            determine whether mood statistically mediates the group differences
            in thought content or whether thought patterns mediate subsequent
            mood changes.
            """
        )
    )

    # Forward mediation
    st.subheader("Forward mediation: Group → Mood → Thought Content")
    dag_forward = MEDIATION_DIR / "DAGs" / "DAG_forward.png"
    safe_image(dag_forward, "DAG of the forward mediation model (Group → Mood → Thoughts)")

    forward_combined = (
        MEDIATION_DIR
        / "multilevel_mediation_comparative"
        / "forward_mediation_combined.png"
    )
    safe_image(forward_combined, "Combined figure for the forward mediation")

    st.markdown(
        textwrap.dedent(
            """
            In this model, group (Control vs RoD) predicts **block-level mood**;
            that mood in turn predicts the average thought content (valence,
            ON/OFF, PCA1, etc.).

            Quantitatively, the group→mood path is negative (Path *a* ≈ −2.0,
            indicating that RoD participants report ≈ 2 units worse mood than
            Controls). This lower mood then predicts more negative and
            self-focused thoughts and lower confidence. The **indirect effects**
            (*a×b*) are significant for **valence** (≈ −5.9, 95% CI [−11.8,
            −1.5]), **self/other** (≈ −7.3, CI [−14.2, −2.1]), **confidence**
            (≈ −3.4, CI [−8.0, −0.1]) and **PCA1** (≈ −0.35, CI roughly
            [−0.68, −0.10]), but not for **time** or **ON/OFF**.

            These results support a forward mediation pattern in which RoD
            participants’ more negative **mood state** partly explains why their
            mind-wandering is more negative, more self-focused, less confident
            and higher on the PCA "rumination" component, over and above any
            direct group differences in thought content.
            """
        )
    )

    # Reverse mediation
    st.subheader("Reverse mediation: Group → Thoughts → Mood change")
    dag_reverse = MEDIATION_DIR / "DAGs" / "DAG_reverse.png"
    safe_image(dag_reverse, "DAG of the reverse mediation model (Group → Thoughts → Mood)")

    reverse_dir = MEDIATION_DIR / "reverse_mediation"
    safe_image(
        reverse_dir / "reverse_mediation_combined.png",
        "Combined figure for the reverse mediation",
    )

    st.markdown(
        textwrap.dedent(
            """
            Here the direction is flipped: group first predicts **thought
            content** (e.g. more negative and self-focused thinking in RoD), and
            those features of mind-wandering then predict **changes in mood** in
            the following block.

            In these reverse models, the estimated indirect effects (*a×b*) are
            generally small (typically |ab| < 0.2) and their 95% confidence
            intervals include zero across all combinations of thought dimension
            (ON/OFF, valence, time, self/other, confidence, PCA1) and mood
            scales. None of the reverse mediation paths reaches conventional
            significance after accounting for baseline mood and time.

            This pattern suggests that, in this dataset, we find **clear
            evidence** that mood helps explain group differences in thought
            content (forward mediation), but **no strong evidence** that
            maladaptive thoughts in one block reliably drive subsequent mood
            changes once baseline mood and temporal trends are controlled. Any
            "vicious cycle" between thoughts and mood would therefore be subtle
            and not robustly captured by these block-level measures.
            """
        )
    )


######################################################################
# MAIN APP
######################################################################

# 8) Cyberball Moderated Mediation (Hayes Model 7)

def page_cyberball_moderated_mediation() -> None:
    st.header("8. Cyberball Moderated Mediation (Hayes Model 7)")

    st.markdown(
        textwrap.dedent(
            """
            This analysis tests whether the **indirect effect** of Cyberball
            condition (Inclusion vs Exclusion) on thought content via mood is
            **moderated by group** (Controls vs Risk of Depression).

            **Core Hypothesis**  
            Social exclusion worsens mood in everyone, but the Risk Group may
            have an "amplifier" on this connection—they react more intensely
            to exclusion, leading to a larger indirect effect on subsequent
            thought content.

            **Model Specification (Hayes Model 7)**
            - **X**: Cyberball Condition (0=Inclusion, 1=Exclusion)
            - **M**: Mood (EVA scales, post-Cyberball)
            - **Y**: Thought Dimensions (valence, time, self/other, ON/OFF, confidence)
            - **W**: Group (0=Controls, 1=Risk of Depression)
            - **Covariate**: Baseline Mood (pre-Cyberball)

            **Path Structure**
            - **Path a (moderated)**: Condition → Mood, with Group × Condition interaction
            - **Path b**: Mood → Thoughts
            - **Path c'**: Direct effect of Condition on Thoughts

            **Indirect Effects**
            - *IE_Controls* = a_base × b (indirect effect for Controls)
            - *IE_Risk* = (a_base + a_int) × b (indirect effect for Risk Group)
            - *Index of Moderated Mediation* = a_int × b (difference between groups)
            """
        )
    )

    # DAG
    st.subheader("Conceptual Model (DAG)")
    dag_cyberball = MEDIATION_DIR / "DAGs" / "DAG_cyberball_moderated.png"
    safe_image(
        dag_cyberball,
        "Moderated Mediation: Cyberball Condition → Mood → Thoughts (Group moderates Path a)"
    )
    st.markdown(
        textwrap.dedent(
            """
            **How to read this DAG**
            - The **red dashed arrow** from Group to Path a represents the
              moderation: "Exclusion affects mood in everyone, but the Risk
              Group has an amplifier on that connection."
            - The **Baseline Mood** (dotted gray arrow) controls for emotional
              inertia, ensuring we measure the *change* due to Cyberball.
            - Path c' (direct effect) is also potentially moderated, testing
              whether exclusion directly biases thoughts beyond mood changes.
            """
        )
    )

    # Combined Figure
    st.subheader("Results: Moderated Mediation Analysis")
    combined_fig = CYBERBALL_MEDIATION_DIR / "plots" / "cyberball_moderated_mediation_combined.png"
    safe_image(
        combined_fig,
        "Combined moderated mediation results: Indirect effects by group, Path A, Path B, and Index of Moderated Mediation"
    )

    st.markdown(
        textwrap.dedent(
            """
            **Figure Interpretation**

            1. **Top Panel (Forest Plot)**: Shows indirect effects (a × b) for
               Controls (blue circles) and Risk Group (red squares) across
               thought dimensions. Filled markers indicate significant effects
               (95% CI excludes zero).

            2. **Middle Left (Path A)**: Effect of exclusion on mood, separately
               for Controls and Risk. If the Risk Group shows a larger negative
               effect, this supports the "emotional amplifier" hypothesis.

            3. **Middle Right (Path B Heatmap)**: Effect of mood on thoughts.
               Asterisks indicate significant paths. This shows which mood
               scales most strongly predict which thought dimensions.

            4. **Index of Moderated Mediation**: The key test of our hypothesis.
               A significant positive Index means the Risk Group has a *stronger*
               indirect effect than Controls (i.e., exclusion → worse mood →
               more negative thoughts is amplified in the Risk Group).

            5. **Bottom (Summary Bar Chart)**: Average effects across mood scales,
               providing a quick visual comparison of indirect effects and the
               moderation index.
            """
        )
    )

    # Results Table
    st.subheader("Detailed Results")
    results_csv = CYBERBALL_MEDIATION_DIR / "cyberball_moderated_mediation_results.csv"
    if results_csv.exists():
        df_results = pd.read_csv(results_csv)

        # Summary statistics
        n_sig_index = df_results["index_mm_sig"].sum()
        n_sig_risk = df_results["ie_risk_sig"].sum()
        n_sig_ctrl = df_results["ie_controls_sig"].sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Significant Index MM", f"{n_sig_index}/{len(df_results)}")
        with col2:
            st.metric("Significant IE (Risk)", f"{n_sig_risk}/{len(df_results)}")
        with col3:
            st.metric("Significant IE (Controls)", f"{n_sig_ctrl}/{len(df_results)}")

        # Show significant Index of Moderated Mediation
        sig_index = df_results[df_results["index_mm_sig"]].sort_values(
            "index_mm", ascending=False
        )
        if not sig_index.empty:
            st.markdown("**Significant Index of Moderated Mediation (Group difference):**")
            for _, row in sig_index.iterrows():
                direction = "stronger" if row["index_mm"] > 0 else "weaker"
                st.markdown(
                    f"- **{row['mood_scale']} → {row['thought_dim']}**: "
                    f"Index = {row['index_mm']:.3f} "
                    f"[{row['index_mm_ci_low']:.3f}, {row['index_mm_ci_high']:.3f}] — "
                    f"Risk Group has {direction} indirect effect"
                )

        # Full table
        st.markdown("**Full Results Table**")
        st.dataframe(
            df_results[
                [
                    "mood_scale", "thought_dim",
                    "ie_controls", "ie_controls_ci_low", "ie_controls_ci_high", "ie_controls_sig",
                    "ie_risk", "ie_risk_ci_low", "ie_risk_ci_high", "ie_risk_sig",
                    "index_mm", "index_mm_ci_low", "index_mm_ci_high", "index_mm_sig",
                ]
            ].round(4)
        )
    else:
        st.info(
            f"Results file not found: {results_csv}. "
            "Please run the cyberball_moderated_mediation.py script first."
        )

    # Interpretation
    st.subheader("Interpretation")
    st.markdown(
        textwrap.dedent(
            """
            **Key Questions Addressed**

            1. **Is there mediation in the Risk Group?**
               Check if *IE_Risk* confidence intervals exclude zero. If yes,
               exclusion → worse mood → more negative/self-focused thoughts
               is a significant pathway for the Risk Group.

            2. **Is the mediation different between groups?**
               Check if the *Index of Moderated Mediation* excludes zero.
               A significant positive Index means the Risk Group shows a
               stronger indirect effect than Controls.

            3. **What drives the moderation?**
               - If Path a (Condition → Mood) is larger for Risk, they have
                 greater *emotional reactivity* to exclusion.
               - If Path b (Mood → Thoughts) is similar across groups, the
                 moderation is primarily driven by differential mood response.

            **Clinical Implications**
            If the Risk Group shows amplified emotional reactivity to social
            exclusion (larger Path a), this suggests that individuals at risk
            for depression may be particularly vulnerable to social rejection,
            with downstream effects on thought content. This supports models
            of depression emphasizing heightened sensitivity to social threat
            and negative cognitive biases following mood perturbation.
            """
        )
    )


######################################################################
# MAIN APP
######################################################################

PAGES: List[str] = [
    "Overview (Text Summary)",
    "1. Demographics & Psychometrics",
    "2. Probe Analysis – ON/OFF",
    "3. Multidimensional Probe Analysis",
    "4. PCA Analysis",
    "5. Correlation Analyses",
    "6. BDI-Split Analysis",
    "7. Mediation Analyses",
    "8. Cyberball Moderated Mediation",
]


def main() -> None:
    st.set_page_config(page_title="CyberSART Behavioral Dashboard", layout="wide")

    # Global CSS override: remove grey background "boxes" around all text blocks
    st.markdown(
        """
        <style>
        /* Generic markdown containers (old + new Streamlit DOM) */
        div.stMarkdown,
        [data-testid="stMarkdownContainer"] {
            background-color: transparent !important;
            padding: 0 !important;
            box-shadow: none !important;
        }

        /* Ensure child elements of markdown (paragraphs, lists) also have no box */
        div.stMarkdown > p,
        div.stMarkdown > ul,
        div.stMarkdown > ol {
            background-color: transparent !important;
            box-shadow: none !important;
        }

        /* Generic vertical blocks that often wrap text */
        div[data-testid="stVerticalBlock"] {
            background-color: transparent !important;
            box-shadow: none !important;
        }

        /* Any preformatted blocks, in case some text is still rendered as <pre> */
        [data-testid="stMarkdownContainer"] pre {
            background-color: transparent !important;
            border-radius: 0 !important;
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Behavioral Results Dashboard")
    selection = st.sidebar.radio("Select section", PAGES, index=0)

    st.sidebar.markdown(
        """---
        **Sections**  
        - Demographics & psychometrics  
        - Behavioral task and probe analyses
        """
    )

    if selection == "Overview (Text Summary)":
        st.title("CyberSART Behavioral Results – Overview")
        st.markdown(
            "This dashboard integrates all main behavioral analyses to help you "
            "report results to your supervisors. Each page corresponds to a "
            "specific analysis and result set in the project, following the exact "
            "order you requested.\n\n"
            "**Key questions and main answers:**\n\n"
            "1. Do Controls and Risk-of-Depression (RoD) participants differ in "
            "their **baseline characteristics**?  → Yes. RoD participants show "
            "substantially higher depressive symptoms and rumination (BDI, RRS, "
            "MWQ) and lower self-esteem, while basic demographics are broadly "
            "comparable.\n"
            "2. How does **sustained attention** (ON/OFF) evolve over time, and how "
            "is it modulated by **Cyberball inclusion/exclusion**?  → ON/OFF "
            "ratings decline across the session in both groups, but the RoD group "
            "shows a steeper deterioration (strong Group × Time interaction). "
            "During Cyberball, exclusion is associated with higher mind-wandering "
            "in RoD, whereas inclusion partially restores engagement; Controls "
            "are much less affected.\n"
            "3. How does the **phenomenology of mind-wandering** (valence, temporal "
            "focus, self/other, confidence) differ between groups?  → RoD "
            "participants report more negative thoughts, become progressively "
            "more past-focused over time and show a stronger drop in confidence, "
            "especially under exclusion. Self/other focus mainly reflects a "
            "general time-on-task trend toward self-referential thought in both "
            "groups.\n"
            "4. How can complex thought patterns be summarized via **PCA** and "
            "linked to trait measures?  → Three components capture most of the "
            "variance: a valence/rumination axis (PC1), a temporal–social axis "
            "(PC2) and a residual component (PC3). PC1 and related negative "
            "thought patterns show strong associations with depressive symptoms "
            "and rumination scales, and are modulated by Cyberball in the RoD "
            "group.\n"
            "5. How do **mood and thought content** interact in multilevel "
            "**mediation models**?  → Forward models support that RoD status leads "
            "to more negative mood, which in turn partly explains more negative, "
            "self-focused and low-confidence thoughts and higher scores on the "
            "PCA rumination component. Reverse models do not show robust evidence "
            "that thought content in one block reliably drives subsequent mood "
            "changes once baseline mood and time are controlled.\n\n"
            "Use the sidebar to navigate through each analysis. Each page combines "
            "the key figures with a short statistical summary and interpretation "
            "that you can reuse in slides or manuscripts."
        )

    elif selection == "1. Demographics & Psychometrics":
        page_demographics()
        
    elif selection == "2. Probe Analysis – ON/OFF":
        page_probe_onoff()

    elif selection == "3. Multidimensional Probe Analysis":
        page_probe_multidim()

    elif selection == "4. PCA Analysis":
        page_pca()

    elif selection == "5. Correlation Analyses":
        page_correlations()

    elif selection == "6. BDI-Split Analysis":
        page_bdi_split()

    elif selection == "7. Mediation Analyses":
        page_mediation()

    elif selection == "8. Cyberball Moderated Mediation":
        page_cyberball_moderated_mediation()


if __name__ == "__main__":
    main()
