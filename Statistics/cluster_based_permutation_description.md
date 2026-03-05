# Methods

## Statistical Analysis

All statistical analyses were performed using a custom pipeline implemented in Python, leveraging the *MNE-Python* (Gramfort et al., 2013), *Statsmodels* (Seabold & Perktold, 2010), and *SciPy* (Virtanen et al., 2020) libraries. The analysis aimed to identify neurophysiological markers associated with mind-wandering and other subjective states (e.g., valence, confidence) while accounting for the hierarchical structure of the data (trials nested within subjects).

### Data Preprocessing and Normalization

Prior to statistical modeling, data were filtered to exclude subjects or trials failing quality assurance checks. To account for inter-individual variability in baseline physiological power, a subject-level Z-score normalization was applied to the neural features (power spectra, connectivity metrics). Continuous behavioral predictors (e.g., mind-wandering intensity, confidence ratings) were similarly Z-scored within each subject to standardize the scale of effects and facilitate model convergence.

### Linear Mixed Models (LMM)

To assess the relationship between neural features and subjective states, we employed a mass-univariate Linear Mixed Model (LMM) approach. For each EEG channel and frequency band (or connectivity feature), a separate LMM was fitted. The general model specification included the neural feature as the dependent variable, the subjective state of interest (e.g., on-task vs. mind-wandering) as a fixed effect, and nuisance covariates (e.g., time-on-task, trial index) as additional fixed effects. A random intercept was included for each subject to account for the non-independence of repeated measures:

$$Y_{ij} = \beta_0 + \beta_1 X_{ij} + \sum_{k} \gamma_k C_{kij} + u_j + \epsilon_{ij}$$

where $Y_{ij}$ is the neural feature for subject $j$ at trial $i$, $X_{ij}$ is the predictor of interest, $C_{kij}$ are covariates, $u_j \sim \mathcal{N}(0, \sigma_u^2)$ is the subject-specific random intercept, and $\epsilon_{ij} \sim \mathcal{N}(0, \sigma_\epsilon^2)$ is the residual error. Models were estimated using Restricted Maximum Likelihood (REML) with Powell's method for optimization to ensure robust convergence.

### Spatial Cluster-Based Permutation Testing

To correct for multiple comparisons across the topographic array of channels, we implemented a non-parametric cluster-based permutation test adapted for LMMs (Maris & Oostenveld, 2007). This procedure identifies spatially contiguous clusters of significant effects without relying on Gaussian assumptions for the test statistic distribution.

1.  **Cluster Formation**: T-statistics were computed for the predictor of interest at all channels. Channels with absolute t-values exceeding a standard threshold ($|t| > 2.5$) were selected. Spatially adjacent selected channels were grouped into clusters based on the specific EEG montage adjacency matrix (defined by Euclidean distance).

2.  **Cluster Statistic**: For each identified cluster, a cluster-level statistic was calculated as the sum of the t-values of its constituent channels.

3.  **Null Distribution**: To assess statistical significance, a null distribution of cluster statistics was generated via permutation. We employed the **Freedman-Lane procedure** (Freedman & Lane, 1983) to handle nuisance covariates strictly. This involved:
    *   Fitting a reduced model (excluding the predictor of interest) to the data.
    *   Computing the residuals of this reduced model.
    *   Permuting these residuals within each subject (to preserve within-subject correlation structure).
    *   Adding the permuted residuals back to the fitted values of the reduced model to create a surrogate dataset satisfying the null hypothesis.
    *   Re-running the full LMM analysis on this surrogate dataset and extracting the maximum cluster statistic.
    This process was repeated (typically $N=1000$ permutations) to build the empirical null distribution.

4.  **Significance Testing**: The observed cluster statistics were compared against the null distribution. Clusters with a Monte Carlo p-value $< 0.05$ (two-tailed) were considered statistically significant.

### Multiple Features Correction

When testing multiple distinct neural markers (e.g., multiple frequency bands or connectivity metrics) simultaneously, we applied the Benjamini-Hochberg False Discovery Rate (FDR) correction to the cluster-level p-values to control the family-wise error rate across the set of tested hypotheses.

### Software and Reproducibility

The analysis pipeline was containerized and executed on a high-performance computing cluster. Key software dependencies included *Python* (v3.9+), *MNE-Python* for channel adjacency and topology handling, *Statsmodels* for mixed effects modeling, and *SciPy* for sparse matrix operations and statistical computations. The exact code structure, including the `Statistics` module for pipeline orchestration, ensures full reproducibility of the reported results.
