# Keep mean-of-evaluation and pooled metrics distinct in all exported tables.
summary_cv <- function(cfg) {
  root <- file.path(cfg$output, "cv")
  data <- read.csv(file.path(root, "all_paired_predictions.csv"),
                   colClasses = c(survey_id = "character"), stringsAsFactors = FALSE)
  validate_cv_coverage(data, cfg)
  keys <- c("region", "repeat_id", "fold", "model", "set")
  parts <- split(data, interaction(data[, keys], drop = TRUE, lex.order = TRUE))
  regional <- do.call(rbind, lapply(parts, function(x) {
    cbind(x[1, keys, drop = FALSE], metric(x$observed, x$predicted))
  }))
  write.csv(regional, file.path(root, "regional_evaluation_metrics.csv"), row.names = FALSE)
  regional_mean <- aggregate(cbind(bias, mae, rmse) ~ region + model + set,
                              regional, mean)
  write.csv(regional_mean, file.path(root, "regional_mean_of_100_evaluations.csv"),
            row.names = FALSE)
  groups <- split(regional, interaction(
    regional[, c("repeat_id", "fold", "model", "set")], drop = TRUE
  ))
  overall <- do.call(rbind, lapply(groups, function(x) {
    data.frame(x[1, c("repeat_id", "fold", "model", "set")], n = sum(x$n),
               bias = sum(x$n * x$bias) / sum(x$n),
               mae = sum(x$sae) / sum(x$n), rmse = sqrt(sum(x$sse) / sum(x$n)))
  }))
  write.csv(overall, file.path(root, "overall_evaluation_metrics.csv"), row.names = FALSE)
  averaged <- aggregate(cbind(bias, mae, rmse) ~ model + set, overall, mean)
  write.csv(averaged, file.path(root, "overall_mean_of_100_evaluations.csv"),
            row.names = FALSE)
  pooled <- do.call(rbind, lapply(split(data, interaction(data$model, data$set)), function(x) {
    cbind(x[1, c("model", "set")], metric(x$observed, x$predicted))
  }))
  write.csv(pooled, file.path(root, "overall_pooled_repeated_predictions.csv"),
            row.names = FALSE)
  reductions <- do.call(rbind, lapply(c("training", "validation"), function(dataset) {
    x <- averaged[averaged$set == dataset, ]
    proposed <- x[x$model == "SPDE_corrected", ]
    data.frame(set = dataset, model = x$model,
               rmse_reduction_percent = safe_relative_reduction(x$rmse, proposed$rmse),
               mae_reduction_percent = safe_relative_reduction(x$mae, proposed$mae))
  }))
  write.csv(reductions, file.path(root, "relative_error_reductions.csv"), row.names = FALSE)
}

safe_relative_reduction <- function(comparator, proposed) {
  # A zero comparison error has no defined percentage improvement.
  ifelse(comparator > 0, 100 * (comparator - proposed) / comparator, NA_real_)
}

summary_external <- function(cfg) {
  pairs <- list()
  for (region in unlist(cfg$regions)) {
    path <- file.path(cfg$output, "bootstrap", region, "external_quadrat_predictions.csv")
    data <- read.csv(path, colClasses = c(quadrat_id = "character", marshland_id = "character"))
    reference <- read_table(cfg, region, "external", "corrected")
    if (!setequal(data$quadrat_id, reference$quadrat_id)) {
      stop("External prediction IDs differ from the original extraction table")
    }
    data <- data[match(reference$quadrat_id, data$quadrat_id), ]
    observed_fields <- c("quadrat_id", "marshland_id", "region", "x", "y", "live_snails")
    if (!isTRUE(all.equal(data[, observed_fields], reference[, observed_fields], check.attributes = FALSE))) {
      stop("External observations/coordinates changed between fitting and evaluation")
    }
    assert_finite(c(data$predicted, data$sd_boot), "external predictions/SD")
    if (any(data$sd_boot < 0)) stop("Negative bootstrap SD")
    if (anyDuplicated(data$quadrat_id)) {
      stop("Duplicate external quadrat IDs")
    }
    for (marshland in unique(data$marshland_id)) {
      x <- data[data$marshland_id == marshland, ]
      if (nrow(x) != 100) {
        stop("External validation requires all 100 quadrats per marshland")
      }
      pairs[[length(pairs) + 1]] <- data.frame(
        region = region, marshland_id = marshland, n_quadrats = 100,
        observed = sum(x$live_snails) / 100, predicted = mean(x$predicted)
      )
    }
  }
  data <- do.call(rbind, pairs)
  data$bias <- data$predicted - data$observed
  if (nrow(data) != 9 || any(table(data$region) != 3)) {
    stop("Expected nine independent marshlands, three per region")
  }
  root <- file.path(cfg$output, "external")
  dir.create(root, showWarnings = FALSE)
  write.csv(data, file.path(root, "marshland_means.csv"), row.names = FALSE)
  total_variation <- sum((data$observed - mean(data$observed))^2)
  pearson <- cor(data$observed, data$predicted, method = "pearson")
  overall <- cbind(
    metric(data$observed, data$predicted), pearson = pearson,
    spearman = cor(data$observed, data$predicted, method = "spearman"),
    pearson_r_squared = pearson^2,
    prediction_r2 = if (total_variation > 0) {
      1 - sum((data$predicted - data$observed)^2) / total_variation
    } else {
      NA_real_
    }
  )
  write.csv(overall, file.path(root, "nine_marshland_metrics.csv"), row.names = FALSE)
  regional <- do.call(rbind, lapply(split(data, data$region), function(x) {
    cbind(region = x$region[1], metric(x$observed, x$predicted))
  }))
  write.csv(regional, file.path(root, "regional_metrics.csv"), row.names = FALSE)
}
