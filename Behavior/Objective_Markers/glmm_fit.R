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

data_path  <- get_arg("--data")
spec_path  <- get_arg("--spec")
coef_path  <- get_arg("--out-coef")
diag_path  <- get_arg("--out-diag")
resid_path <- get_arg("--out-resid")

spec <- fromJSON(spec_path)
d <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)
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

# Optimizer messages alone do NOT catch a singular fit, where a random-effect
# variance collapses to ~0. That is the real failure mode for the OLRE models,
# so record it explicitly along with each random-effect SD.
re_sd <- as.data.frame(VarCorr(m))
re_sd_str <- paste(sprintf("%s=%.4f", re_sd$grp, re_sd$sdcor), collapse = "; ")

# glmer does not always populate optinfo$derivs (it is absent for some
# optimizer paths), so guard the gradient rather than assume it exists.
grad <- m@optinfo$derivs$gradient
max_grad <- if (is.null(grad)) NA_real_ else max(abs(grad))

diag_df <- data.frame(
  converged    = is.null(conv_msgs) || length(conv_msgs) == 0,
  conv_message = if (is.null(conv_msgs) || length(conv_msgs) == 0) ""
                 else paste(conv_msgs, collapse = "; "),
  singular     = isSingular(m),
  re_sd        = re_sd_str,
  max_grad     = max_grad,
  dispersion   = sum(rp^2) / df.residual(m),
  n_obs        = nrow(d),
  n_subjects   = nlevels(droplevels(d$subject)),
  stringsAsFactors = FALSE
)
write.csv(diag_df, diag_path, row.names = FALSE)

# Per-observation fitted values and Pearson residuals, so the Python side can
# draw QQ / residual-vs-fitted / binned-residual diagnostics. Binned residuals
# (Gelman-Hill) are the standard substitute for DHARMa here, which cannot be
# installed because the conda channel is blocked on this network.
if (!is.na(resid_path) && nzchar(resid_path)) {
  resid_df <- data.frame(
    fitted   = fitted(m),
    residual = rp,
    stringsAsFactors = FALSE
  )
  write.csv(resid_df, resid_path, row.names = FALSE)
}
