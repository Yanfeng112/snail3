# Base-R regression checks: no INLA/sf model is fitted by this script.
# Run explicitly with: Rscript tests/test_R_core.R
source("R/00_common.R")
source("R/04_bootstrap_prediction.R")
expect_error <- function(expression) {
  failed <- FALSE
  tryCatch(force(expression), error = function(error) { failed <<- TRUE })
  stopifnot(failed)
}
cfg <- list(covariates = list(x = list(transform = "identity")),
            unseen_predictor_policy = "error", design_rank_policy = "error")
train <- data.frame(x = c(1, 2, 3))
target <- data.frame(x = c(100, 101))
design <- prepare_design(train, target, "x", cfg)
stopifnot(identical(unname(design$mean), 2), identical(unname(design$sd), 1))
stopifnot(isTRUE(all.equal(design$new$x, c(98, 99))))
expect_error(prepare_design(data.frame(x = c(1, 1, 1)), target, "x", cfg))
factors <- list(covariates = list(vegetation = list(levels = c("Other", "Reed"))),
                unseen_predictor_policy = "error", design_rank_policy = "report")
expect_error(prepare_design(data.frame(vegetation = rep("Other", 3)),
                           data.frame(vegetation = "Reed"), "vegetation", factors))
collision <- list(covariates = list(vegetation = list(levels = c("0", "1")),
                                   vegetation_L2 = list(transform = "identity")))
values <- data.frame(vegetation = c("0", "1"), vegetation_L2 = c(0, 1))
expect_error(prepare_design(values, values, names(collision$covariates), collision))
stopifnot(isTRUE(all.equal(inverse_rmse_weights(c(0, 0, 1)), c(.5, .5, 0))))
weights <- inverse_rmse_weights(c(1e-300, 1))
stopifnot(all(is.finite(weights)), abs(sum(weights) - 1) < 1e-15)
expect_error(inverse_rmse_weights(c(NA_real_, 1)))

# Verify the prediction writer preserves a value just below a decimal class boundary.
path <- tempfile(fileext = ".csv")
value <- 0.0005 - .Machine$double.eps * 0.0005
write_prediction_csv(data.frame(predicted = value), path)
stopifnot(identical(read.csv(path)$predicted, value))
unlink(path)

cat("Base-R contract checks passed. No INLA or TensorFlow fit was performed.\n")
