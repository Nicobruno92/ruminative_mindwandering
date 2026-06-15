"""
Mind-Wandering Classification Pipeline — ML Utilities.

Cross-validation, feature selection, model building, and metric computation.
Supports LOSO (Leave-One-Subject-Out) cross-validation only.
Models: RandomForest, XGBoost, LogisticRegression.

Project: depressed_mindwandering
"""

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked")
warnings.filterwarnings("ignore", message="resource_tracker:.*FileNotFoundError.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import trim_mean
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, chi2
)
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_curve, recall_score, precision_score,
    average_precision_score, matthews_corrcoef, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectorMixin

try:
    from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    SMOTE = SVMSMOTE = ADASYN = SMOTETomek = ImbPipeline = None

try:
    from mrmr import mrmr_classif
    HAS_MRMR = True
except ImportError:
    HAS_MRMR = False
    mrmr_classif = None

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None


# =============================================================================
# LMM FEATURE SELECTION
# =============================================================================

def _fit_single_lmm_feature_decoding(
    feat_idx: int, feature_col: np.ndarray,
    y_arr: np.ndarray, groups_arr: np.ndarray
) -> float:
    """
    Fit LMM decoding model: Label ~ Feature + (1|Subject).

    Uses Linear Probability Model approximation (binary outcome as continuous).

    Parameters
    ----------
    feat_idx : int
        Index of the feature (used for parallelization).
    feature_col : np.ndarray
        Feature values for all samples.
    y_arr : np.ndarray
        Binary target variable.
    groups_arr : np.ndarray
        Subject group labels.

    Returns
    -------
    float
        P-value for the feature effect, or 1.0 if model fails to converge.
    """
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    if np.std(feature_col) == 0:
        return 1.0

    exog = sm.add_constant(feature_col)
    mlm = MixedLM(endog=y_arr, exog=exog, groups=groups_arr)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*Covariance.*')
        warnings.filterwarnings('ignore', message='.*convergence.*')
        result = mlm.fit(reml=False, method='lbfgs', maxiter=200, disp=False)

    return result.pvalues[1]


def _fit_single_lmm_feature_encoding(
    feat_idx: int, feature_col: np.ndarray,
    y_arr: np.ndarray, groups_arr: np.ndarray
) -> float:
    """
    Fit LMM encoding model: Feature ~ Label + (1|Subject).

    Uses Gaussian family (continuous feature as outcome). Better convergence
    than the decoding model because the outcome is continuous.

    Parameters
    ----------
    feat_idx : int
        Index of the feature (used for parallelization).
    feature_col : np.ndarray
        Feature values for all samples (dependent variable).
    y_arr : np.ndarray
        Binary label (independent variable).
    groups_arr : np.ndarray
        Subject group labels.

    Returns
    -------
    float
        P-value for the label effect, or 1.0 if model fails to converge.
    """
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    if np.std(feature_col) == 0:
        return 1.0

    exog = sm.add_constant(y_arr)
    mlm = MixedLM(endog=feature_col, exog=exog, groups=groups_arr)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*Covariance.*')
        warnings.filterwarnings('ignore', message='.*convergence.*')
        result = mlm.fit(reml=False, method='lbfgs', maxiter=200, disp=False)

    return result.pvalues[1]


def _lmm_prefilter_indices(X_arr: np.ndarray, y_arr: np.ndarray,
                            n_candidates: int) -> np.ndarray:
    """
    Fast f_classif pre-filter: return indices of the top n_candidates features
    by F-statistic. Used to reduce the LMM call count before the full LMM run.
    """
    from sklearn.feature_selection import f_classif
    f_stats, _ = f_classif(X_arr, y_arr)
    f_stats = np.nan_to_num(f_stats, nan=0.0)
    return np.argsort(f_stats)[::-1][:n_candidates]


class LMMDecodingFeatureSelector(BaseEstimator, SelectorMixin):
    """
    Feature selector using LMM decoding model: Label ~ Feature + (1|Subject).

    Parameters
    ----------
    k : int
        Number of top features to select (lowest p-values).
    n_jobs : int
        Number of parallel jobs.
    prefilter_factor : int
        If > 0, first run f_classif and keep only ``prefilter_factor * k``
        candidates before fitting LMMs. Reduces LMM calls by
        N_features / (prefilter_factor * k). Default 0 = disabled.
    """

    def __init__(self, k: int = 20, n_jobs: int = -1, prefilter_factor: int = 0):
        self.k = k
        self.n_jobs = n_jobs
        self.prefilter_factor = prefilter_factor
        self.pvalues_ = None
        self.selected_features_mask_ = None
        self.n_features_in_ = None

    def fit(self, X, y, groups=None):
        """Fit the LMM decoding feature selector."""
        from joblib import Parallel, delayed

        if groups is None:
            raise ValueError(
                "LMMDecodingFeatureSelector requires 'groups' for mixed-effects modeling. "
                "Pass groups via fit_params: {'feature_selection__groups': groups}"
            )

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
            self.feature_names_ = X.columns.tolist()
        else:
            X_arr = np.asarray(X)
            self.feature_names_ = [f"feature_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y).ravel()
        groups_arr = np.asarray(groups).ravel()
        self.n_features_in_ = X_arr.shape[1]
        n_features = X_arr.shape[1]

        # Optional fast pre-filter: run f_classif first, then LMM on top candidates
        if self.prefilter_factor > 0 and n_features > self.k:
            n_candidates = min(self.prefilter_factor * self.k, n_features)
            candidate_idx = _lmm_prefilter_indices(X_arr, y_arr, n_candidates)
            X_candidates = X_arr[:, candidate_idx]
        else:
            candidate_idx = np.arange(n_features)
            X_candidates = X_arr

        pvalues_candidates = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(_fit_single_lmm_feature_decoding)(
                i, X_candidates[:, i], y_arr, groups_arr
            )
            for i in range(X_candidates.shape[1])
        )

        # Map back to full feature space (non-candidates get p=1.0)
        full_pvalues = np.ones(n_features)
        full_pvalues[candidate_idx] = pvalues_candidates

        self.pvalues_ = full_pvalues
        actual_k = min(self.k, n_features)
        top_k_indices = np.argsort(self.pvalues_)[:actual_k]
        self.selected_features_mask_ = np.zeros(n_features, dtype=bool)
        self.selected_features_mask_[top_k_indices] = True
        return self

    def _get_support_mask(self):
        if self.selected_features_mask_ is None:
            raise ValueError("LMMDecodingFeatureSelector has not been fitted yet.")
        return self.selected_features_mask_


class LMMEncodingFeatureSelector(BaseEstimator, SelectorMixin):
    """
    Feature selector using LMM encoding model: Feature ~ Label + (1|Subject).

    Better convergence than decoding because the outcome is continuous.

    Parameters
    ----------
    k : int
        Number of top features to select (lowest p-values).
    n_jobs : int
        Number of parallel jobs.
    prefilter_factor : int
        If > 0, first run f_classif and keep only ``prefilter_factor * k``
        candidates before fitting LMMs. Reduces LMM calls by
        N_features / (prefilter_factor * k). Default 0 = disabled.
    """

    def __init__(self, k: int = 20, n_jobs: int = -1, prefilter_factor: int = 0):
        self.k = k
        self.n_jobs = n_jobs
        self.prefilter_factor = prefilter_factor
        self.pvalues_ = None
        self.selected_features_mask_ = None
        self.n_features_in_ = None

    def fit(self, X, y, groups=None):
        """Fit the LMM encoding feature selector."""
        from joblib import Parallel, delayed

        if groups is None:
            raise ValueError(
                "LMMEncodingFeatureSelector requires 'groups' for mixed-effects modeling. "
                "Pass groups via fit_params: {'feature_selection__groups': groups}"
            )

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
            self.feature_names_ = X.columns.tolist()
        else:
            X_arr = np.asarray(X)
            self.feature_names_ = [f"feature_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y).ravel()
        groups_arr = np.asarray(groups).ravel()
        self.n_features_in_ = X_arr.shape[1]
        n_features = X_arr.shape[1]

        # Optional fast pre-filter: run f_classif first, then LMM on top candidates
        if self.prefilter_factor > 0 and n_features > self.k:
            n_candidates = min(self.prefilter_factor * self.k, n_features)
            candidate_idx = _lmm_prefilter_indices(X_arr, y_arr, n_candidates)
            X_candidates = X_arr[:, candidate_idx]
        else:
            candidate_idx = np.arange(n_features)
            X_candidates = X_arr

        pvalues_candidates = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(_fit_single_lmm_feature_encoding)(
                i, X_candidates[:, i], y_arr, groups_arr
            )
            for i in range(X_candidates.shape[1])
        )

        # Map back to full feature space (non-candidates get p=1.0)
        full_pvalues = np.ones(n_features)
        full_pvalues[candidate_idx] = pvalues_candidates

        self.pvalues_ = full_pvalues
        actual_k = min(self.k, n_features)
        top_k_indices = np.argsort(self.pvalues_)[:actual_k]
        self.selected_features_mask_ = np.zeros(n_features, dtype=bool)
        self.selected_features_mask_[top_k_indices] = True
        return self

    def _get_support_mask(self):
        if self.selected_features_mask_ is None:
            raise ValueError("LMMEncodingFeatureSelector has not been fitted yet.")
        return self.selected_features_mask_


# Alias for backwards compatibility
LMMFeatureSelector = LMMDecodingFeatureSelector


class MRMRFeatureSelector(BaseEstimator, SelectorMixin):
    """
    Feature selector using MRMR (Minimum Redundancy Maximum Relevance).

    Parameters
    ----------
    k : int
        Number of features to select.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self.selected_features_ = None
        self.support_mask_ = None

    def fit(self, X, y, groups=None):
        """Fit MRMR feature selector."""
        if not HAS_MRMR:
            raise ImportError("mrmr is required. Install with: pip install mrmr_selection")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # mrmr_classif uses boolean Series indexing internally, which fails when
        # X and y have non-contiguous indices (always the case after LOSO slicing).
        X = X.reset_index(drop=True)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)

        y_int = y.astype(int)
        k_actual = min(self.k, X.shape[1])
        selected_features = mrmr_classif(X=X, y=y_int, K=k_actual)
        self.selected_features_ = selected_features
        self.support_mask_ = np.array([col in selected_features for col in X.columns])
        return self

    def _get_support_mask(self):
        return self.support_mask_



# =============================================================================
# CONFIDENCE-BASED SAMPLE WEIGHTING
# =============================================================================

def compute_within_subject_confidence_weights(
    confidence: np.ndarray,
    groups: np.ndarray,
    w_min: float = 0.1,
    normalization: str = "within_subject",
) -> np.ndarray:
    """
    Map per-trial probe confidence to per-trial training sample weights.

    Confidence (a probe-report metacognitive rating, 0-100 scale) is used as a
    proxy for label reliability: higher-confidence trials receive more weight in
    the classifier's loss. Normalisation is performed independently within each
    subject so that the weights reflect *trial-level* reliability rather than
    between-subject differences in baseline confidence.

    Parameters
    ----------
    confidence : np.ndarray
        Raw confidence values (0-100), aligned to the training samples.
    groups : np.ndarray
        Subject IDs, aligned to ``confidence``. Normalisation is per group.
    w_min : float
        Floor weight assigned to a subject's lowest-confidence trial(s). Values
        in (0, 1]; ``w_min < 1`` down-weights but never fully removes low-
        confidence trials. The maximum-confidence trial always maps to 1.0.
    normalization : str
        Currently only ``"within_subject"`` (min-max within each subject) is
        supported.

    Returns
    -------
    np.ndarray
        Sample weights in ``[w_min, 1]``, same length and order as ``confidence``.
        A subject whose confidence is constant (zero range) maps to all-ones,
        i.e. a neutral weighting for that subject.
    """
    if normalization != "within_subject":
        raise ValueError(
            f"Unknown confidence weight normalization: '{normalization}'. "
            f"Supported: 'within_subject'."
        )

    confidence = np.asarray(confidence, dtype=float).ravel()
    groups = np.asarray(groups).ravel()
    if confidence.shape[0] != groups.shape[0]:
        raise ValueError(
            f"confidence ({confidence.shape[0]}) and groups ({groups.shape[0]}) "
            f"must have the same length."
        )

    weights = np.ones_like(confidence, dtype=float)
    for subj in np.unique(groups):
        mask = groups == subj
        c_subj = confidence[mask]
        c_min, c_max = c_subj.min(), c_subj.max()
        c_range = c_max - c_min
        if c_range == 0:
            # Constant confidence -> neutral (all ones) for this subject.
            weights[mask] = 1.0
        else:
            weights[mask] = w_min + (1.0 - w_min) * (c_subj - c_min) / c_range

    return weights


# =============================================================================
# OVERSAMPLING
# =============================================================================

def get_oversampler(method: str, k_neighbors: int = 5, random_state: int = 42,
                    m_neighbors: int = 10):
    """
    Instantiate an oversampler by method name.

    Parameters
    ----------
    method : str
        One of "SMOTE", "SVMSMOTE", "ADASYN", "SMOTETomek".
    k_neighbors : int
        Number of neighbors for SMOTE-based methods.
    random_state : int
        Random seed.
    m_neighbors : int
        SVMSMOTE-specific: neighbors used for danger/noise detection (default 10).
        Must be < n_samples_fit; caller is responsible for capping it.

    Returns
    -------
    Configured oversampler instance.
    """
    if not HAS_IMBLEARN:
        raise ImportError("imbalanced-learn is required. Install with: pip install imbalanced-learn")

    if method == "SMOTE":
        return SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    elif method == "SVMSMOTE":
        return SVMSMOTE(random_state=random_state, k_neighbors=k_neighbors,
                        m_neighbors=m_neighbors)
    elif method == "ADASYN":
        return ADASYN(random_state=random_state, n_neighbors=k_neighbors)
    elif method == "SMOTETomek":
        return SMOTETomek(
            random_state=random_state,
            smote=SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        )
    else:
        raise ValueError(f"Unknown oversampling method: {method}. "
                         f"Choose from: SMOTE, SVMSMOTE, ADASYN, SMOTETomek")


def _interpolate_weights_after_resample(X_original, X_balanced, weights_original):
    """
    Assign a sample weight to every resampled row via 1-NN on the originals.

    Each output row inherits the weight of its single nearest original sample in
    feature space. Original rows are unchanged by SMOTE, so they match themselves
    (distance 0) and keep their exact weight; synthetic rows inherit the weight of
    the original sample they were interpolated closest to. This carries label
    reliability into synthetic samples without perturbing SMOTE's feature-space
    geometry (the weight is never added as a feature) and is independent of the
    order in which the resampler returns rows.

    Parameters
    ----------
    X_original : pd.DataFrame or np.ndarray
        The subject's feature matrix before resampling.
    X_balanced : pd.DataFrame or np.ndarray
        The subject's feature matrix after resampling.
    weights_original : np.ndarray
        Weights aligned to ``X_original``.

    Returns
    -------
    np.ndarray
        Weights aligned to ``X_balanced``, bounded by the original weight range.
    """
    from sklearn.neighbors import NearestNeighbors
    X_orig_arr = np.asarray(X_original, dtype=float)
    X_bal_arr = np.asarray(X_balanced, dtype=float)
    nn = NearestNeighbors(n_neighbors=1).fit(X_orig_arr)
    _, idx = nn.kneighbors(X_bal_arr)
    return np.asarray(weights_original, dtype=float)[idx.ravel()]


def apply_within_subject_oversampling(
    X, y, groups,
    method: str = "SMOTE", k_neighbors: int = 5, random_state: int = 42,
    return_groups: bool = False,
    weights=None, return_weights: bool = False,
):
    """
    Apply oversampling within each subject in the training set.

    Each subject's samples are independently balanced, then concatenated.
    Subjects with too few minority samples for SMOTE use a reduced k_neighbors.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    groups : np.ndarray
        Subject IDs.
    method : str
        Oversampling method.
    k_neighbors : int
        Number of neighbors for SMOTE-based methods.
    random_state : int
        Random seed.
    return_groups : bool
        If True, also return a groups array aligned to the expanded (X, y).
        Synthetic samples are assigned to the same subject as their source.
        Required when using LMM feature selectors after within-subject SMOTE,
        because the groups array must match the expanded training set size.
    weights : np.ndarray, optional
        Per-sample weights aligned to (X, y, groups). When provided together with
        ``return_weights=True``, an expanded weight array is returned in which
        synthetic samples inherit the weight of their nearest original sample (see
        :func:`_interpolate_weights_after_resample`).
    return_weights : bool
        If True, also return the expanded weight array. Requires ``weights``.

    Returns
    -------
    tuple
        (X_balanced, y_balanced), optionally extended with groups_balanced
        (if return_groups) and/or weights_balanced (if return_weights), in that
        order.
    """
    if return_weights and weights is None:
        raise ValueError("return_weights=True requires 'weights' to be provided.")

    weights_arr = None if weights is None else np.asarray(weights, dtype=float).ravel()

    unique_subjects = np.unique(groups)
    balanced_X_list = []
    balanced_y_list = []
    balanced_groups_list = []
    balanced_weights_list = []

    for subj in unique_subjects:
        subj_mask = groups == subj
        X_subj = X.loc[subj_mask].copy() if isinstance(X, pd.DataFrame) else X[subj_mask].copy()
        y_subj = y[subj_mask] if isinstance(y, np.ndarray) else y.loc[subj_mask].values
        w_subj = None if weights_arr is None else weights_arr[subj_mask]

        unique_classes, counts = np.unique(y_subj, return_counts=True)

        # Skip resampling if only one class
        if len(unique_classes) < 2:
            balanced_X_list.append(X_subj)
            balanced_y_list.append(y_subj)
            balanced_groups_list.append(np.full(len(y_subj), subj))
            if w_subj is not None:
                balanced_weights_list.append(w_subj)
            continue

        min_count = min(counts)

        # Cap k_neighbors and SVMSMOTE's m_neighbors to what the minority class can support.
        # SVMSMOTE internally needs m_neighbors + 1 neighbors, so cap at min_count - 1.
        actual_k = min(k_neighbors, min_count - 1)
        actual_m = min(10, min_count - 1)  # SVMSMOTE m_neighbors default is 10
        if actual_k < 1:
            balanced_X_list.append(X_subj)
            balanced_y_list.append(y_subj)
            balanced_groups_list.append(np.full(len(y_subj), subj))
            if w_subj is not None:
                balanced_weights_list.append(w_subj)
            continue

        oversampler = get_oversampler(method, k_neighbors=actual_k, random_state=random_state,
                                      m_neighbors=actual_m)
        X_subj_balanced, y_subj_balanced = oversampler.fit_resample(X_subj, y_subj)
        balanced_X_list.append(X_subj_balanced)
        balanced_y_list.append(y_subj_balanced)
        # Synthetic samples belong to the same subject (used by LMM selectors)
        balanced_groups_list.append(np.full(len(y_subj_balanced), subj))
        if w_subj is not None:
            balanced_weights_list.append(
                _interpolate_weights_after_resample(X_subj, X_subj_balanced, w_subj)
            )

    X_balanced = (pd.concat(balanced_X_list, ignore_index=True)
                  if isinstance(X, pd.DataFrame)
                  else np.vstack(balanced_X_list))
    y_balanced = np.concatenate(balanced_y_list)

    result = [X_balanced, y_balanced]
    if return_groups:
        result.append(np.concatenate(balanced_groups_list))
    if return_weights:
        result.append(np.concatenate(balanced_weights_list))
    return tuple(result)


# =============================================================================
# SCALING
# =============================================================================

def _apply_fold_scaling(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    groups_train: pd.Series, groups_test: pd.Series,
    mode: str, scaler_type: str = 'standard'
) -> tuple:
    """
    Apply scaling inside a CV fold.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train and test feature matrices.
    groups_train, groups_test : pd.Series
        Subject labels for train and test.
    mode : str
        'global': fit scaler on train, transform both.
        'within': scale each participant independently.
    scaler_type : str
        'standard' or 'robust'.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled)
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return X_train, X_test

    ScalerClass = RobustScaler if scaler_type == 'robust' else StandardScaler
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if mode == 'global':
        scaler = ScalerClass()
        X_train_scaled.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    elif mode == 'within':
        # Scale each training participant by their own stats.
        for participant in groups_train.unique():
            mask = groups_train == participant
            scaler = ScalerClass()
            X_train_scaled.loc[mask, numeric_cols] = scaler.fit_transform(
                X_train.loc[mask, numeric_cols]
            )
        # Scale the held-out (test) participant by their own stats as well.
        # This is consistent with the within-person normalization applied to
        # training participants: every subject's features are expressed in units
        # of that subject's own mean / std.  Because the test subject's data is
        # never seen during training, fitting the scaler on test samples does
        # NOT constitute leakage — it simply puts the held-out features on the
        # same within-person scale that the model was trained to expect.
        for participant in groups_test.unique():
            mask = groups_test == participant
            subj_scaler = ScalerClass()
            X_test_scaled.loc[mask, numeric_cols] = subj_scaler.fit_transform(
                X_test.loc[mask, numeric_cols]
            )

    return X_train_scaled, X_test_scaled


# =============================================================================
# PIPELINE BUILDING
# =============================================================================

def build_model_pipeline(
    X,
    model_type: str = 'rf',
    use_smote: bool = True,
    oversampling_method: str = 'SMOTE',
    random_state: int = None,
    class_weight=None,
    scale_pos_weight=None,
    k: int = 20,
    rf_params: dict = None,
    xgb_params: dict = None,
    lr_params: dict = None,
    ocsvm_params: dict = None,
    iforest_params: dict = None,
    feature_selection_method: str = 'f_classif',
    scaler: str = 'none',
    y=None,
    lmm_n_jobs: int = -1,
    lmm_prefilter_factor: int = 0,
    use_pca: bool = False,
    pca_n_components: int = None,
    pca_type: str = 'standard',
    pca_kernel: str = 'rbf',
    smote_k_neighbors: int = 5,
):
    """
    Build an sklearn Pipeline for RF, XGBoost, or LogisticRegression.

    Pipeline order: preprocessor → scaler → feature_selection → pca → smote → clf.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (used for column inspection only).
    model_type : str
        One of 'rf', 'xgb', 'lr', 'ocsvm', 'iforest'.
    use_smote : bool
        Whether to add oversampling step.
    oversampling_method : str
        SMOTE variant name.
    random_state : int
        Random seed.
    class_weight : str or dict
        Class weight specification for the classifier.
    scale_pos_weight : float or 'auto'
        For XGBoost imbalance handling.
    k : int
        Number of features to select.
    rf_params, xgb_params, lr_params, ocsvm_params, iforest_params : dict
        Model-specific hyperparameter overrides.
    feature_selection_method : str
        One of 'mrmr', 'lmm', 'lmm_encoding', 'lmm_decoding', 'f_classif',
        'mutual_info_classif', 'chi2', 'none'.
    scaler : str
        One of 'standard', 'minmax', 'robust', 'none'.
    y : array-like, optional
        Needed for dynamic XGBoost weighting and MRMR.
    lmm_n_jobs : int
        Parallel jobs for LMM feature selection.
    use_pca : bool
        Whether to apply PCA after feature selection.
    pca_n_components : int, optional
        PCA components.
    pca_type : str
        'standard' or 'kernel'.
    pca_kernel : str
        Kernel for KernelPCA.

    Returns
    -------
    Pipeline or ImbPipeline
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[('num', 'passthrough', numeric_features)]
    )
    steps = [('preprocessor', preprocessor)]

    is_oneclass = model_type in ('ocsvm', 'iforest')

    # Scaling (before feature selection)
    if scaler == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaler == 'robust':
        steps.append(('scaler', RobustScaler()))

    # Feature selection (skipped for one-class models — supervised selectors require multiple classes)
    if feature_selection_method != 'none' and not is_oneclass:
        actual_k = len(numeric_features) if k == 'all' else min(int(k), len(numeric_features))

        if feature_selection_method in ('lmm', 'lmm_decoding'):
            feature_selector = LMMDecodingFeatureSelector(
                k=actual_k, n_jobs=lmm_n_jobs, prefilter_factor=lmm_prefilter_factor)
        elif feature_selection_method == 'lmm_encoding':
            feature_selector = LMMEncodingFeatureSelector(
                k=actual_k, n_jobs=lmm_n_jobs, prefilter_factor=lmm_prefilter_factor)
        elif feature_selection_method == 'mrmr':
            feature_selector = MRMRFeatureSelector(k=actual_k)
        elif feature_selection_method == 'f_classif':
            feature_selector = SelectKBest(f_classif, k=actual_k)
        elif feature_selection_method == 'mutual_info_classif':
            feature_selector = SelectKBest(mutual_info_classif, k=actual_k)
        elif feature_selection_method == 'chi2':
            feature_selector = SelectKBest(chi2, k=actual_k)
        else:
            feature_selector = SelectKBest(f_classif, k=actual_k)

        steps.append(('feature_selection', feature_selector))

    # PCA (optional, after feature selection — skipped for one-class models)
    if use_pca and not is_oneclass:
        pca_transformer = (
            KernelPCA(n_components=pca_n_components, kernel=pca_kernel, random_state=random_state)
            if pca_type == 'kernel'
            else PCA(n_components=pca_n_components, random_state=random_state)
        )
        steps.append(('pca', pca_transformer))

    # Oversampling (after feature selection — skipped for one-class models)
    if use_smote and not is_oneclass:
        if not HAS_IMBLEARN:
            raise ImportError("imbalanced-learn is required. Install with: pip install imbalanced-learn")
        steps.append(('smote', get_oversampler(oversampling_method, k_neighbors=smote_k_neighbors,
                                               random_state=random_state)))

    # Classifier
    if model_type == 'rf':
        rf_defaults = {
            'n_estimators': 100,
            'max_depth': 40,
            'min_samples_split': 15,
            'min_samples_leaf': 15,
            'max_features': 0.5,
            'random_state': random_state,
            'class_weight': class_weight if class_weight is not None else 'balanced',
            'n_jobs': -1,
            'bootstrap': True,
        }
        if rf_params:
            rf_defaults.update({k: v for k, v in rf_params.items() if v is not None})
        steps.append(('clf', RandomForestClassifier(**rf_defaults)))

    elif model_type == 'xgb':
        if not HAS_XGBOOST:
            raise ImportError("model_type='xgb' requires 'xgboost'. Install with: pip install xgboost")

        # Dynamic class weight
        if scale_pos_weight in ('auto', None) and y is not None:
            y_temp = pd.Series(y)
            n_neg = (y_temp == 0).sum()
            n_pos = (y_temp == 1).sum()
            xgb_scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        elif isinstance(scale_pos_weight, (int, float)):
            xgb_scale_pos_weight = float(scale_pos_weight)
        else:
            xgb_scale_pos_weight = 1.0

        xgb_defaults = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 4,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'scale_pos_weight': xgb_scale_pos_weight,
            'gamma': 1,
            'seed': random_state,
            'n_jobs': -1,
        }
        if xgb_params:
            cleaned = {k: v for k, v in xgb_params.items()
                       if v is not None and not (k == 'scale_pos_weight' and v == 'auto')}
            xgb_defaults.update(cleaned)
        steps.append(('clf', xgb.XGBClassifier(**xgb_defaults)))

    elif model_type == 'lr':
        lr_defaults = {
            'penalty': 'elasticnet',
            'solver': 'saga',
            'class_weight': 'balanced',
            'C': 0.1,
            'l1_ratio': 0.5,
            'max_iter': 2000,
            'random_state': random_state,
            'n_jobs': -1,
        }
        if lr_params:
            cleaned = {k: v for k, v in lr_params.items() if v is not None}
            lr_defaults.update(cleaned)
        steps.append(('clf', LogisticRegression(**lr_defaults)))

    elif model_type == 'ocsvm':
        from sklearn.svm import OneClassSVM
        ocsvm_defaults = {
            'nu': 0.1,
            'kernel': 'rbf',
            'gamma': 'scale',
        }
        if ocsvm_params:
            ocsvm_defaults.update({k: v for k, v in ocsvm_params.items() if v is not None})
        steps.append(('clf', OneClassSVM(**ocsvm_defaults)))

    elif model_type == 'iforest':
        from sklearn.ensemble import IsolationForest
        iforest_defaults = {
            'n_estimators': 100,
            'contamination': 'auto',
            'random_state': random_state,
            'n_jobs': -1,
        }
        if iforest_params:
            iforest_defaults.update({k: v for k, v in iforest_params.items() if v is not None})
        steps.append(('clf', IsolationForest(**iforest_defaults)))

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from: 'rf', 'xgb', 'lr', 'ocsvm', 'iforest'")

    return ImbPipeline(steps) if (use_smote and not is_oneclass) else Pipeline(steps)


# =============================================================================
# CROSS-VALIDATION (LOSO ONLY)
# =============================================================================

from joblib import Parallel, delayed

def _process_cv_fold_loso(
    fold_idx, train_idx, test_idx, X, y, groups, pipeline, scale_by_participant, scaler,
    use_smote, oversampling_scope, oversampling_method, random_state, smote_k_neighbors: int = 5,
    sample_weights=None,
):
    train_subjects = set(groups.iloc[train_idx])
    test_subjects = set(groups.iloc[test_idx])
    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        raise ValueError(f"CRITICAL: Subject leakage in fold {fold_idx}: {overlap}")

    held_out_subject = list(test_subjects)[0]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    # Per-trial training weights (e.g. confidence-based). Only the training fold
    # is weighted; test metrics remain unweighted and comparable across runs.
    w_train = None if sample_weights is None else np.asarray(sample_weights)[train_idx]

    if scale_by_participant and scale_by_participant != 'none':
        X_train, X_test = _apply_fold_scaling(
            X_train, X_test, groups_train, groups_test, scale_by_participant, scaler
        )

    # Within-subject oversampling: keep an aligned groups array so that LMM
    # feature selectors (which need groups) receive a vector of the same length
    # as the expanded training set.  Synthetic samples are labelled with the
    # same subject ID as the real samples they were interpolated from. Sample
    # weights, when present, are interpolated onto synthetic samples in the same
    # call so they stay aligned to the expanded training set.
    if use_smote and oversampling_scope == 'within':
        oversampled = apply_within_subject_oversampling(
            X_train, y_train, groups_train.values,
            method=oversampling_method, k_neighbors=smote_k_neighbors, random_state=random_state,
            return_groups=True,
            weights=w_train, return_weights=(w_train is not None),
        )
        if w_train is None:
            X_train, y_train, groups_train_expanded = oversampled
        else:
            X_train, y_train, groups_train_expanded, w_train = oversampled
    else:
        groups_train_expanded = groups_train

    # Use clone from sklearn.base
    from sklearn.base import clone
    fold_pipeline = clone(pipeline)
    fit_params = {}
    feature_selector_step = fold_pipeline.named_steps.get('feature_selection', None)
    if type(feature_selector_step).__name__ in ('LMMDecodingFeatureSelector', 'LMMEncodingFeatureSelector'):
        # Use the (possibly expanded) groups so size matches X_train after SMOTE
        fit_params['feature_selection__groups'] = (
            groups_train_expanded
            if isinstance(groups_train_expanded, np.ndarray)
            else groups_train_expanded.values
        )

    if w_train is not None:
        fit_params['clf__sample_weight'] = np.asarray(w_train, dtype=float)

    fold_pipeline.fit(X_train, y_train, **fit_params)

    y_pred = fold_pipeline.predict(X_test)
    y_score = None
    auc = 0.5
    fpr, tpr = None, None
    auprc = 0.0
    precision_curve, recall_curve = None, None

    if hasattr(fold_pipeline, 'predict_proba'):
        proba = fold_pipeline.predict_proba(X_test)
        y_score = proba[:, 1]
        if len(np.unique(y_test)) > 1:
            from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
            auc = roc_auc_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auprc = average_precision_score(y_test, y_score)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)

    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    mcc = matthews_corrcoef(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    loso_subject_metric = {
        'subject': held_out_subject,
        'fold_idx': fold_idx,
        'auc': auc,
        'auprc': auprc,
        'mcc': mcc,
        'balanced_accuracy': bal_acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'n_samples': len(y_test),
        'n_positive': int(y_test.sum()),
        'n_negative': int(len(y_test) - y_test.sum()),
    }

    import pandas as pd
    label_counts = pd.Series(y_test).value_counts(normalize=True).to_dict()
    fold_detail = {
        'fold_idx': fold_idx,
        'subject': held_out_subject,
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_score.tolist() if y_score is not None else [],
        'test_indices': test_idx.tolist(),
        'label_percentages': label_counts,
    }

    importances = np.zeros(X.shape[1])
    fs_step = fold_pipeline.named_steps.get('feature_selection', None)
    clf_step = fold_pipeline.named_steps.get('clf', None)
    if fs_step is not None and clf_step is not None and hasattr(clf_step, 'feature_importances_'):
        selected_indices = fs_step.get_support(indices=True)
        importances_val = clf_step.feature_importances_
        importances[selected_indices] = importances_val

    return {
        'fold_idx': fold_idx,
        'auc': auc,
        'auprc': auprc,
        'mcc': mcc,
        'bal_acc': bal_acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'prec_curve': precision_curve,
        'rec_curve': recall_curve,
        'importances': importances,
        'fold_detail': fold_detail,
        'loso_subject_metric': loso_subject_metric,
        'estimator': fold_pipeline
    }

def _process_cv_fold_loso_oneclass(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    pipeline,
    scale_by_participant: str,
    scaler: str,
    oneclass_target: str,
    random_state: int,
) -> dict:
    """Process a single LOSO fold using a one-class (outlier detection) classifier.

    Trains on the target class only from the N-1 training subjects, evaluates
    on the held-out subject. Decision function scores are signed so that higher
    values indicate the minority class.

    Parameters
    ----------
    fold_idx : int
        Index of the current fold.
    train_idx, test_idx : np.ndarray
        Integer position indices for train/test split.
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Binary target labels (0/1).
    groups : pd.Series
        Subject IDs (defines LOSO folds).
    pipeline : sklearn Pipeline
        Unfitted pipeline with a one-class classifier as the final step.
    scale_by_participant : str
        Participant-level scaling strategy ('none', 'global', 'within').
    scaler : str
        Scaler type used in per-participant scaling.
    oneclass_target : str
        ``"minority"`` → train on class 1; inlier (+1) = minority.
        ``"majority"`` → train on class 0; outlier (-1) = minority.
    random_state : int
        Kept for API consistency; unused here.

    Returns
    -------
    dict
        Fold-level metrics compatible with ``_process_cv_fold_loso``.
    """
    train_subjects = set(groups.iloc[train_idx])
    test_subjects = set(groups.iloc[test_idx])
    overlap = train_subjects.intersection(test_subjects)
    if overlap:
        raise ValueError(f"CRITICAL: Subject leakage in fold {fold_idx}: {overlap}")

    held_out_subject = list(test_subjects)[0]
    X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    if scale_by_participant and scale_by_participant != 'none':
        X_train_full, X_test = _apply_fold_scaling(
            X_train_full, X_test, groups_train, groups_test, scale_by_participant, scaler
        )

    counts = y_train_full.value_counts()
    if oneclass_target == "minority":
        train_class_label = counts.idxmin()
        score_sign = 1.0    # higher score → more inlier → more minority (class 1)
        inlier_maps_to = 1
    else:
        train_class_label = counts.idxmax()
        score_sign = -1.0   # lower score → more outlier → more minority (class 1)
        inlier_maps_to = 0

    X_train_oneclass = X_train_full[y_train_full == train_class_label].reset_index(drop=True)

    from sklearn.base import clone
    fold_pipeline = clone(pipeline)
    fold_pipeline.fit(X_train_oneclass)

    raw_pred = fold_pipeline.predict(X_test)          # +1 = inlier, -1 = outlier
    y_pred = np.where(raw_pred == 1, inlier_maps_to, 1 - inlier_maps_to)

    y_score = None
    auc = 0.5
    fpr, tpr = None, None
    auprc = 0.0
    precision_curve, recall_curve = None, None

    if hasattr(fold_pipeline, 'decision_function'):
        raw_scores = fold_pipeline.decision_function(X_test)
        y_score = score_sign * raw_scores
        if len(np.unique(y_test)) > 1:
            from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
            auc = roc_auc_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auprc = average_precision_score(y_test, y_score)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)

    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    mcc = matthews_corrcoef(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    loso_subject_metric = {
        'subject': held_out_subject,
        'fold_idx': fold_idx,
        'auc': auc,
        'auprc': auprc,
        'mcc': mcc,
        'balanced_accuracy': bal_acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'n_samples': len(y_test),
        'n_positive': int(y_test.sum()),
        'n_negative': int(len(y_test) - y_test.sum()),
    }

    label_counts = pd.Series(y_test).value_counts(normalize=True).to_dict()
    fold_detail = {
        'fold_idx': fold_idx,
        'subject': held_out_subject,
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_score.tolist() if y_score is not None else [],
        'test_indices': test_idx.tolist(),
        'label_percentages': label_counts,
    }

    return {
        'fold_idx': fold_idx,
        'auc': auc,
        'auprc': auprc,
        'mcc': mcc,
        'bal_acc': bal_acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'prec_curve': precision_curve,
        'rec_curve': recall_curve,
        'importances': np.zeros(X.shape[1]),
        'fold_detail': fold_detail,
        'loso_subject_metric': loso_subject_metric,
        'estimator': fold_pipeline,
    }


def _process_cv_fold_within_oneclass(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: pd.DataFrame,
    y: pd.Series,
    pipeline,
    oneclass_target: str,
    random_state: int,
) -> dict:
    """Process a single CV fold using a one-class (outlier detection) classifier.

    Trains the pipeline on a single class only, then predicts on the full test set.
    Decision function scores are signed so that higher values indicate the minority class.

    Parameters
    ----------
    fold_idx : int
        Index of the current CV fold.
    train_idx : np.ndarray
        Integer position indices for the training split.
    test_idx : np.ndarray
        Integer position indices for the test split.
    X : pd.DataFrame
        Full feature matrix for this subject.
    y : pd.Series
        Binary target labels (0/1) for this subject.
    pipeline : sklearn Pipeline
        Unfitted pipeline whose final step is a one-class classifier
        (OneClassSVM or IsolationForest).
    oneclass_target : str
        ``"minority"`` → train on class 1 (minority/MB); inlier (+1) = class 1.
        ``"majority"`` → train on class 0 (majority); outlier (-1) = class 1.
    random_state : int
        Kept for API consistency; unused here.

    Returns
    -------
    dict
        Fold-level performance metrics compatible with ``_process_cv_fold_within``.
    """
    X_train_full = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train_full = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    counts = y_train_full.value_counts()
    if oneclass_target == "minority":
        train_class_label = counts.idxmin()
        score_sign = 1.0    # higher decision score → more inlier → more minority (class 1)
        inlier_maps_to = 1
    else:
        train_class_label = counts.idxmax()
        score_sign = -1.0   # lower decision score → more outlier → more minority (class 1)
        inlier_maps_to = 0

    X_train_oneclass = X_train_full[y_train_full == train_class_label].reset_index(drop=True)

    from sklearn.base import clone
    fold_pipeline = clone(pipeline)
    fold_pipeline.fit(X_train_oneclass)

    raw_pred = fold_pipeline.predict(X_test)          # +1 = inlier, -1 = outlier
    y_pred = np.where(raw_pred == 1, inlier_maps_to, 1 - inlier_maps_to)

    y_score = None
    auc = 0.5
    fpr, tpr = None, None
    auprc = 0.0
    precision_curve, recall_curve = None, None

    if hasattr(fold_pipeline, 'decision_function'):
        raw_scores = fold_pipeline.decision_function(X_test)
        y_score = score_sign * raw_scores
        if len(np.unique(y_test)) > 1:
            from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
            auc = roc_auc_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auprc = average_precision_score(y_test, y_score)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)

    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    mcc = matthews_corrcoef(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    label_counts = pd.Series(y_test).value_counts(normalize=True).to_dict()
    fold_detail = {
        'fold_idx': fold_idx,
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_score.tolist() if y_score is not None else [],
        'test_indices': test_idx.tolist(),
        'label_percentages': label_counts,
    }

    return {
        'fold_idx': fold_idx,
        'auc': auc,
        'auprc': auprc,
        'mcc': mcc,
        'bal_acc': bal_acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'prec_curve': precision_curve,
        'rec_curve': recall_curve,
        'importances': np.zeros(X.shape[1]),
        'fold_detail': fold_detail,
        'estimator': fold_pipeline,
    }


def run_model_pipeline_cv(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_type: str = 'rf',
    use_smote: bool = True, oversampling_method: str = 'SMOTE', oversampling_scope: str = 'global',
    fixed_random_state: int = None, class_weight=None, scale_pos_weight=None, k: int = 20,
    rf_params: dict = None, xgb_params: dict = None, lr_params: dict = None,
    ocsvm_params: dict = None, iforest_params: dict = None,
    oneclass_target: str = "minority",
    feature_selection_method: str = 'mrmr', scaler: str = 'standard', scale_by_participant: str = 'none',
    lmm_n_jobs: int = 1, lmm_prefilter_factor: int = 0,
    use_pca: bool = False, pca_n_components: int = None,
    pca_type: str = 'standard', pca_kernel: str = 'rbf', cv_n_jobs: int = -1,
    smote_k_neighbors: int = 5, sample_weights=None,
) -> pd.DataFrame:

    random_state = fixed_random_state if fixed_random_state is not None else 42
    is_oneclass = model_type in ('ocsvm', 'iforest')
    pipeline_scaler = 'none' if (scale_by_participant and scale_by_participant != 'none') else scaler

    # Confidence-based sample weighting is only defined for supervised models and
    # for oversampling that happens outside the estimator (scope != 'global'),
    # where weights can be interpolated onto synthetic samples and passed to the
    # classifier. Fail explicitly rather than silently dropping the weights.
    if sample_weights is not None:
        if is_oneclass:
            raise ValueError(
                f"sample_weights is not supported for one-class model "
                f"'{model_type}': these models do not use labels."
            )
        if use_smote and oversampling_scope == 'global':
            raise ValueError(
                "sample_weights requires oversampling_scope != 'global'. "
                "With global SMOTE the resampler lives inside the pipeline and "
                "cannot propagate per-sample weights. Use oversampling_scope: "
                "'within' (or disable SMOTE)."
            )
        if len(sample_weights) != len(y):
            raise ValueError(
                f"sample_weights length ({len(sample_weights)}) must match the "
                f"number of samples ({len(y)})."
            )

    # Fix to 1 for inner threads to avoid oversubscription
    if rf_params is None: rf_params = {}
    if xgb_params is None: xgb_params = {}
    if lr_params is None: lr_params = {}
    if ocsvm_params is None: ocsvm_params = {}
    if iforest_params is None: iforest_params = {}
    rf_params['n_jobs'] = 1
    xgb_params['n_jobs'] = 1
    lr_params['n_jobs'] = 1
    iforest_params['n_jobs'] = 1

    pipeline = build_model_pipeline(
        X, model_type=model_type, use_smote=(use_smote and oversampling_scope == 'global'),
        oversampling_method=oversampling_method, y=y, random_state=random_state,
        class_weight=class_weight, scale_pos_weight=scale_pos_weight, k=k,
        rf_params=rf_params, xgb_params=xgb_params, lr_params=lr_params,
        ocsvm_params=ocsvm_params, iforest_params=iforest_params,
        feature_selection_method=feature_selection_method, scaler=pipeline_scaler,
        lmm_n_jobs=lmm_n_jobs, lmm_prefilter_factor=lmm_prefilter_factor,
        use_pca=use_pca, pca_n_components=pca_n_components,
        pca_type=pca_type, pca_kernel=pca_kernel, smote_k_neighbors=smote_k_neighbors,
    )

    from sklearn.model_selection import LeaveOneGroupOut
    cv = LeaveOneGroupOut()
    cv_splits = list(cv.split(X, y, groups))

    # Parallel over folds
    if is_oneclass:
        results = Parallel(n_jobs=cv_n_jobs, backend='loky')(
            delayed(_process_cv_fold_loso_oneclass)(
                fold_idx, train_idx, test_idx, X, y, groups, pipeline,
                scale_by_participant, scaler, oneclass_target, random_state
            )
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits)
        )
    else:
        results = Parallel(n_jobs=cv_n_jobs, backend='loky')(
            delayed(_process_cv_fold_loso)(
                fold_idx, train_idx, test_idx, X, y, groups, pipeline, scale_by_participant, scaler,
                use_smote, oversampling_scope, oversampling_method, random_state, smote_k_neighbors,
                sample_weights
            )
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits)
        )

    result_dict = {
        'mean_auc': np.mean([r['auc'] for r in results]),
        'std_auc': np.std([r['auc'] for r in results]),
        'mean_auprc': np.mean([r['auprc'] for r in results]),
        'std_auprc': np.std([r['auprc'] for r in results]),
        'mean_mcc': np.mean([r['mcc'] for r in results]),
        'std_mcc': np.std([r['mcc'] for r in results]),
        'mean_balanced_accuracy': np.mean([r['bal_acc'] for r in results]),
        'std_balanced_accuracy': np.std([r['bal_acc'] for r in results]),
        'mean_precision': np.mean([r['prec'] for r in results]),
        'std_precision': np.std([r['prec'] for r in results]),
        'mean_recall': np.mean([r['rec'] for r in results]),
        'std_recall': np.std([r['rec'] for r in results]),
        'mean_f1': np.mean([r['f1'] for r in results]),
        'std_f1': np.std([r['f1'] for r in results]),
        'fold_aucs': [r['auc'] for r in results],
        'fold_auprcs': [r['auprc'] for r in results],
        'fold_mccs': [r['mcc'] for r in results],
        'fold_bal_accs': [r['bal_acc'] for r in results],
        'fold_cms': [r['cm'] for r in results],
        'fold_fprs': [r['fpr'] for r in results if r['fpr'] is not None],
        'fold_tprs': [r['tpr'] for r in results if r['tpr'] is not None],
        'fold_precisions_curve': [r['prec_curve'] for r in results if r['prec_curve'] is not None],
        'fold_recalls_curve': [r['rec_curve'] for r in results if r['rec_curve'] is not None],
        'feature_importances': np.mean([r['importances'] for r in results], axis=0),
        'fold_details': [r['fold_detail'] for r in results],
        'cv_splits': cv_splits,
        'estimators': [r['estimator'] for r in results],
        'loso_subject_metrics': [r['loso_subject_metric'] for r in results],
    }
    return pd.DataFrame([result_dict])


# =============================================================================
# CROSS-VALIDATION (WITHIN-SUBJECT)
# =============================================================================

def _process_cv_fold_within(
    fold_idx, train_idx, test_idx, X, y, pipeline, use_smote, oversampling_scope,
    oversampling_method, random_state, y_raw=None, groups=None, smote_k_neighbors: int = 5
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    if y_train.nunique() < 2:
        return {
            'fold_idx': fold_idx,
            'skipped': True,
            'skip_reason': 'single_class_y_train',
            'importances': np.zeros(X.shape[1]),
        }

    from sklearn.base import clone
    fold_pipeline = clone(pipeline)

    if use_smote and oversampling_scope == 'within':
        # Scale first (fitting on training data only), then SMOTE in normalized
        # feature space so k-NN distances are not distorted by raw magnitudes.
        pre_names = {'preprocessor', 'scaler'}
        X_train_t = X_train.copy()
        X_test_t = X_test.copy()
        for name, step in fold_pipeline.steps:
            if name in pre_names:
                arr_tr = step.fit_transform(X_train_t, y_train)
                arr_te = step.transform(X_test_t)
                X_train_t = pd.DataFrame(arr_tr, columns=X_train.columns)
                X_test_t = pd.DataFrame(arr_te, columns=X_train.columns)

        X_train_t, y_train = apply_within_subject_oversampling(
            X_train_t, y_train, np.zeros(len(y_train)),
            method=oversampling_method, k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )

        remaining_steps = [(n, s) for n, s in fold_pipeline.steps if n not in pre_names]
        fit_params = {}
        fs_obj = next((s for n, s in remaining_steps if n == 'feature_selection'), None)
        if fs_obj is not None and type(fs_obj).__name__ in (
            'LMMDecodingFeatureSelector', 'LMMEncodingFeatureSelector'
        ):
            if groups is not None:
                g_train = groups.iloc[train_idx] if isinstance(groups, pd.Series) else groups[train_idx]
                fit_params['feature_selection__groups'] = (
                    g_train.values if hasattr(g_train, 'values') else g_train
                )
        fold_pipeline = Pipeline(remaining_steps)
        fold_pipeline.fit(X_train_t, y_train, **fit_params)
        X_test = X_test_t
    else:
        # LMM selectors require groups passed via fit_params.
        fit_params = {}
        feature_selector_step = fold_pipeline.named_steps.get('feature_selection', None)
        if feature_selector_step is not None and type(feature_selector_step).__name__ in (
            'LMMDecodingFeatureSelector', 'LMMEncodingFeatureSelector'
        ):
            if groups is not None:
                groups_train = groups.iloc[train_idx] if isinstance(groups, pd.Series) else groups[train_idx]
                fit_params['feature_selection__groups'] = (
                    groups_train.values if hasattr(groups_train, 'values') else groups_train
                )
        fold_pipeline.fit(X_train, y_train, **fit_params)

    y_pred = fold_pipeline.predict(X_test)
    y_score = None
    auc = 0.5
    fpr, tpr = None, None
    auprc = 0.0
    precision_curve, recall_curve = None, None

    if hasattr(fold_pipeline, 'predict_proba'):
        proba = fold_pipeline.predict_proba(X_test)
        y_score = proba[:, 1]
        if len(np.unique(y_test)) > 1:
            from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
            auc = roc_auc_score(y_test, y_score)
            fpr, tpr, _ = roc_curve(y_test, y_score)
            auprc = average_precision_score(y_test, y_score)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)

    from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    mcc = matthews_corrcoef(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    label_counts = pd.Series(y_test).value_counts(normalize=True).to_dict()
    fold_detail = {
        'fold_idx': fold_idx,
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'y_proba': y_score.tolist() if y_score is not None else [],
        'test_indices': test_idx.tolist(),
        'label_percentages': label_counts,
    }

    importances = np.zeros(X.shape[1])
    fs_step = fold_pipeline.named_steps.get('feature_selection', None)
    clf_step = fold_pipeline.named_steps.get('clf', None)
    if fs_step is not None and clf_step is not None and hasattr(clf_step, 'feature_importances_'):
        selected_indices = fs_step.get_support(indices=True)
        importances_val = clf_step.feature_importances_
        importances[selected_indices] = importances_val

    return {
        'fold_idx': fold_idx,
        'auc': auc,
        'auprc': auprc,
        'mcc': mcc,
        'bal_acc': bal_acc,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'cm': cm,
        'fpr': fpr,
        'tpr': tpr,
        'prec_curve': precision_curve,
        'rec_curve': recall_curve,
        'importances': importances,
        'fold_detail': fold_detail,
        'estimator': fold_pipeline
    }

def run_within_subject_cv(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series = None, cv_strategy: str = 'stratified_kfold',
    cv_folds: int = 5, model_type: str = 'rf', use_smote: bool = True, oversampling_method: str = 'SMOTE',
    oversampling_scope: str = 'within', fixed_random_state: int = None, class_weight=None,
    scale_pos_weight=None, k: int = 20, rf_params: dict = None, xgb_params: dict = None,
    lr_params: dict = None, ocsvm_params: dict = None, iforest_params: dict = None,
    oneclass_target: str = "minority",
    feature_selection_method: str = 'mrmr', scaler: str = 'standard',
    use_pca: bool = False, pca_n_components: int = None, pca_type: str = 'standard',
    pca_kernel: str = 'rbf', cv_n_jobs: int = -1,
    lmm_n_jobs: int = 1,
    lmm_prefilter_factor: int = 0,
    y_raw: "pd.Series | None" = None,
    smote_k_neighbors: int = 5,
) -> pd.DataFrame:

    random_state = fixed_random_state if fixed_random_state is not None else 42
    is_oneclass = model_type in ('ocsvm', 'iforest')

    # Fix to 1 for inner threads to avoid oversubscription
    if rf_params is None: rf_params = {}
    if xgb_params is None: xgb_params = {}
    if lr_params is None: lr_params = {}
    if ocsvm_params is None: ocsvm_params = {}
    if iforest_params is None: iforest_params = {}
    rf_params['n_jobs'] = 1
    xgb_params['n_jobs'] = 1
    lr_params['n_jobs'] = 1
    iforest_params['n_jobs'] = 1

    pipeline = build_model_pipeline(
        X, model_type=model_type, use_smote=(use_smote and oversampling_scope == 'global'),
        oversampling_method=oversampling_method, y=y, random_state=random_state, class_weight=class_weight,
        scale_pos_weight=scale_pos_weight, k=k, rf_params=rf_params, xgb_params=xgb_params,
        lr_params=lr_params, ocsvm_params=ocsvm_params, iforest_params=iforest_params,
        feature_selection_method=feature_selection_method, scaler=scaler,
        lmm_n_jobs=lmm_n_jobs, lmm_prefilter_factor=lmm_prefilter_factor,
        use_pca=use_pca, pca_n_components=pca_n_components, pca_type=pca_type, pca_kernel=pca_kernel,
        smote_k_neighbors=smote_k_neighbors,
    )

    # Guard: StratifiedKFold requires at least n_splits samples in every class.
    # After the neutral-zone gap exclusion, some subjects (especially in permutations)
    # may fall below this threshold — return empty so the caller skips them.
    min_class_count = int(pd.Series(y).value_counts().min())
    if min_class_count < cv_folds:
        return pd.DataFrame()

    from sklearn.model_selection import StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
    if cv_strategy == 'stratified_kfold':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_splits = list(cv.split(X, y))
    elif cv_strategy == 'group_kfold':
        if groups is None: raise ValueError("groups array required for group_kfold cv_strategy")
        cv = GroupKFold(n_splits=cv_folds)
        cv_splits = list(cv.split(X, y, groups))
    elif cv_strategy == 'repeated_stratified_kfold':
        cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=3, random_state=random_state)
        cv_splits = list(cv.split(X, y))
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    if is_oneclass:
        results = Parallel(n_jobs=cv_n_jobs, backend='loky')(
            delayed(_process_cv_fold_within_oneclass)(
                fold_idx, train_idx, test_idx, X, y, pipeline, oneclass_target, random_state
            )
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits)
        )
    else:
        results = Parallel(n_jobs=cv_n_jobs, backend='loky')(
            delayed(_process_cv_fold_within)(
                fold_idx, train_idx, test_idx, X, y, pipeline, use_smote, oversampling_scope,
                oversampling_method, random_state, y_raw, groups, smote_k_neighbors
            )
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits)
        )

    # Drop any folds that were skipped (e.g. single-class y_train after
    # training-median rebinarization). If every fold was skipped, return an
    # empty DataFrame so the caller treats this subject as filtered-out.
    skipped = [r for r in results if r.get('skipped', False)]
    results = [r for r in results if not r.get('skipped', False)]
    if skipped:
        print(
            f"  [within-CV] Skipped {len(skipped)}/{len(skipped)+len(results)} "
            f"fold(s): {skipped[0].get('skip_reason', 'unknown')}"
        )
    if not results:
        return pd.DataFrame()

    result_dict = {
        'mean_auc': np.mean([r['auc'] for r in results]),
        'std_auc': np.std([r['auc'] for r in results]),
        'mean_auprc': np.mean([r['auprc'] for r in results]),
        'std_auprc': np.std([r['auprc'] for r in results]),
        'mean_mcc': np.mean([r['mcc'] for r in results]),
        'std_mcc': np.std([r['mcc'] for r in results]),
        'mean_balanced_accuracy': np.mean([r['bal_acc'] for r in results]),
        'std_balanced_accuracy': np.std([r['bal_acc'] for r in results]),
        'mean_precision': np.mean([r['prec'] for r in results]),
        'std_precision': np.std([r['prec'] for r in results]),
        'mean_recall': np.mean([r['rec'] for r in results]),
        'std_recall': np.std([r['rec'] for r in results]),
        'mean_f1': np.mean([r['f1'] for r in results]),
        'std_f1': np.std([r['f1'] for r in results]),
        'fold_aucs': [r['auc'] for r in results],
        'fold_auprcs': [r['auprc'] for r in results],
        'fold_mccs': [r['mcc'] for r in results],
        'fold_bal_accs': [r['bal_acc'] for r in results],
        'fold_cms': [r['cm'] for r in results],
        'fold_fprs': [r['fpr'] for r in results if r['fpr'] is not None],
        'fold_tprs': [r['tpr'] for r in results if r['tpr'] is not None],
        'fold_precisions_curve': [r['prec_curve'] for r in results if r['prec_curve'] is not None],
        'fold_recalls_curve': [r['rec_curve'] for r in results if r['rec_curve'] is not None],
        'feature_importances': np.mean([r['importances'] for r in results], axis=0),
        'fold_details': [r['fold_detail'] for r in results],
        'cv_splits': cv_splits,
        'estimators': [r['estimator'] for r in results],
    }
    return pd.DataFrame([result_dict])


# =============================================================================
# SHAP
# =============================================================================

def compute_shap_values_for_pipeline(
    pipeline, X: pd.DataFrame, feature_names
) -> np.ndarray:
    """
    Compute SHAP values for a fitted pipeline (RF or XGB).

    Applies the pipeline's scaler step (if present) before feature selection
    and SHAP computation so that the classifier receives data on the same scale
    it was trained on.  Unselected features are assigned SHAP value of zero so
    the output shape always matches the full feature set.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        Fitted pipeline with optional 'scaler', 'feature_selection', 'clf' steps.
    X : pd.DataFrame
        Feature matrix for the test fold (may be unscaled).
    feature_names : Index
        Full list of feature names (before selection).

    Returns
    -------
    np.ndarray
        SHAP values of shape (n_samples, n_features).
    """
    if not HAS_SHAP:
        raise ImportError("shap is required. Install with: pip install shap")

    scaler           = pipeline.named_steps.get('scaler', None)
    feature_selector = pipeline.named_steps.get('feature_selection', None)
    clf              = pipeline.named_steps.get('clf', None)

    # Apply scaler so the clf receives data on the same scale it was trained on.
    if scaler is not None:
        X_arr = scaler.transform(X)
        X = pd.DataFrame(X_arr, columns=X.columns, index=X.index)

    if feature_selector is not None:
        selected_mask = feature_selector.get_support()
        selected_features = [f for f, sel in zip(feature_names, selected_mask) if sel]
        X_selected = X[selected_features]
    else:
        selected_mask = np.ones(len(feature_names), dtype=bool)
        X_selected = X

    if clf is None or not hasattr(clf, 'predict_proba'):
        raise ValueError("SHAP computation requires a classifier with predict_proba (RF or XGB).")

    explainer = shap.Explainer(clf, X_selected)
    # RF predict_proba SHAP values are reconstructed from per-tree margin
    # contributions; floating-point summation order can make the additivity
    # check fail by a small margin even when the decomposition is correct.
    shap_vals = explainer(X_selected, check_additivity=False).values

    # Handle 3D output (n_samples, n_features, 2) — take positive class
    if shap_vals.ndim == 3 and shap_vals.shape[2] == 2:
        shap_vals = shap_vals[..., 1]

    # Map back to full feature space
    if feature_selector is not None:
        full_shap = np.zeros((len(X_selected), len(feature_names)))
        full_shap[:, selected_mask] = shap_vals
        return full_shap

    return shap_vals