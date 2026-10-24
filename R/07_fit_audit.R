# Preserve posterior summaries and fitting assumptions alongside model objects.
write_fit_audit <- function(fit, cfg, prefix) {
  write.csv(fit$model$summary.fixed, paste0(prefix, "_fixed_effects.csv"), row.names = TRUE)
  write.csv(fit$model$summary.hyperpar, paste0(prefix, "_hyperparameters.csv"), row.names = TRUE)
  write.csv(fit$diagnostics, paste0(prefix, "_diagnostics.csv"), row.names = FALSE)
  saveRDS(list(variables = fit$vars, mean = fit$design$mean, sd = fit$design$sd),
          paste0(prefix, "_preprocessing.rds"))
  write_json(list(
    response = "log1p(density)", inverse = "expm1(linear_predictor)",
    fixed_effect_mean = cfg$fixed_effect_mean,
    fixed_effect_precision = cfg$fixed_effect_precision,
    intercept_mean = cfg$intercept_mean,
    intercept_precision = cfg$intercept_precision,
    gaussian_precision_hyper = cfg$gaussian_precision_hyper,
    spde_alpha = cfg$alpha,
    spde_coordinate_divisor = cfg$spde_coordinate_divisor,
    theta_parameterisation = c("log_tau", "log_kappa"),
    theta_prior_mean = c(0, 0), theta_prior_precision = diag(2),
    negative_prediction_policy = cfg$negative_prediction_policy,
    training_support = fit$design$support_audit,
    baseline_feature_policy = cfg$baseline_feature_policy,
    mode_status = fit$model$mode$mode.status,
    note = "Hyperparameter labels/scales are retained from the actual INLA build."
  ), paste0(prefix, "_assumptions.json"))
}
