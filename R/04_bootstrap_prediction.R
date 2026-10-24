# Environmental inputs stay fixed; bootstrap SD measures sampling/refitting dispersion.
bootstrap_predict <- function(cfg, region) {
  root <- file.path(cfg$output, "bootstrap", region)
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  train <- read_table(cfg, region, "survey", "corrected")
  grid <- read_table(cfg, region, "grid", "corrected")
  external <- read_table(cfg, region, "external", "corrected")
  variables <- names(cfg$covariates)
  columns <- c("x", "y", variables)
  targets <- rbind(grid[, columns, drop = FALSE], external[, columns, drop = FALSE])
  mesh <- get_mesh(cfg, region)
  saveRDS(mesh, file.path(root, "mesh3.rds"))
  signature <- analysis_signature(cfg, list(train = train, targets = targets), mesh)
  full_path <- file.path(root, "full_fit_checkpoint.rds")
  full_result <- load_checkpoint(full_path, signature)
  if (is.null(full_result)) {
    selected <- select_spde(train, mesh, cfg)
    full <- fit_spde(train, targets, selected$vars, mesh, cfg)
    full_result <- list(selected = selected, fit = full)
    save_checkpoint(list(signature = signature, value = full_result), full_path)
  }
  selected <- full_result$selected
  full <- full_result$fit
  write.csv(selected$history, file.path(root, "full_training_selection.csv"), row.names = FALSE)
  saveRDS(full$model, file.path(root, "full_training_model.rds"))
  write_fit_audit(full, cfg, file.path(root, "full_training"))
  single <- back_transform(full$new, cfg$negative_prediction_policy)
  write.csv(data.frame(row_id = seq_along(single), predicted = single),
            file.path(root, "single_fit_predictions.csv"), row.names = FALSE)
  blocks <- block_ids(train, cfg)
  unique_blocks <- sort(unique(blocks))
  if (length(unique_blocks) < 3) {
    stop("Insufficient occupied blocks for bootstrap")
  }
  errors <- numeric(100)
  for (bootstrap_id in 1:100) {
    prefix <- file.path(root, sprintf("bootstrap_%03d", bootstrap_id))
    checkpoint <- paste0(prefix, "_checkpoint.rds")
    result <- load_checkpoint(checkpoint, signature)
    if (is.null(result)) {
      result <- tryCatch({
        set.seed(42 + bootstrap_id - 1)
        held_out_blocks <- sample(unique_blocks, max(1, round(.1 * length(unique_blocks))))
        available <- setdiff(unique_blocks, held_out_blocks)
        sampled_blocks <- sample(available, length(available), replace = TRUE)
        train_rows <- unlist(lapply(sampled_blocks, function(b) which(blocks == b)))
        validation_rows <- which(blocks %in% held_out_blocks)
        new <- rbind(train[validation_rows, columns, drop = FALSE], targets)
        variables <- if (identical(cfg$bootstrap_variable_selection, "training_only")) {
          select_spde(train[train_rows, ], mesh, cfg)$vars
        } else {
          selected$vars
        }
        fit <- fit_spde(train[train_rows, ], new, variables, mesh, cfg)
        prediction <- back_transform(fit$new, cfg$negative_prediction_policy)
        holdout_indices <- seq_along(validation_rows)
        list(
          prediction = prediction[-holdout_indices],
          validation_rmse = metric(train$density[validation_rows], prediction[holdout_indices])$rmse,
          audit = list(
            training_rows = train_rows, validation_rows = validation_rows,
            sampled_blocks = sampled_blocks, held_out_blocks = held_out_blocks,
            variables = variables, selection_mode = cfg$bootstrap_variable_selection,
            mean = fit$design$mean, sd = fit$design$sd,
            training_support = fit$design$support_audit
          )
        )
      }, error = function(error) {
        record_failure(error, paste0(prefix, "_failure.json"),
                       list(region = region, bootstrap_id = bootstrap_id))
      })
      save_checkpoint(list(signature = signature, value = result), checkpoint)
    }
    errors[bootstrap_id] <- result$validation_rmse
    saveRDS(result$prediction, file.path(root, sprintf("prediction_%03d.rds", bootstrap_id)))
    saveRDS(result$audit, file.path(root, sprintf("audit_%03d.rds", bootstrap_id)))
    cat(region, "bootstrap", bootstrap_id, "complete\n")
  }
  weights <- inverse_rmse_weights(errors)
  write.csv(data.frame(bootstrap = 1:100, rmse = errors, weight = weights),
            file.path(root, "weights.csv"), row.names = FALSE)
  ensemble_mean <- numeric(nrow(targets))
  for (b in 1:100) {
    prediction <- readRDS(file.path(root, sprintf("prediction_%03d.rds", b)))
    ensemble_mean <- ensemble_mean + weights[b] * prediction
  }
  ensemble_variance <- numeric(nrow(targets))
  for (b in 1:100) {
    prediction <- readRDS(file.path(root, sprintf("prediction_%03d.rds", b)))
    ensemble_variance <- ensemble_variance + weights[b] * (prediction - ensemble_mean)^2
  }
  grid_rows <- seq_len(nrow(grid))
  external_rows <- nrow(grid) + seq_len(nrow(external))
  single_grid <- data.frame(grid[, c("cell_id", "region", "x", "y")],
                            predicted = single[grid_rows], sd_boot = sqrt(ensemble_variance[grid_rows]))
  ensemble_grid <- data.frame(grid[, c("cell_id", "region", "x", "y")],
                              predicted = ensemble_mean[grid_rows], sd_boot = sqrt(ensemble_variance[grid_rows]))
  metadata <- external[, c("quadrat_id", "marshland_id", "region", "x", "y", "live_snails")]
  single_external <- data.frame(metadata, predicted = single[external_rows],
                                sd_boot = sqrt(ensemble_variance[external_rows]))
  ensemble_external <- data.frame(metadata, predicted = ensemble_mean[external_rows],
                                  sd_boot = sqrt(ensemble_variance[external_rows]))
  write_prediction_csv(single_grid, file.path(root, "grid_predictions_single_fit.csv"))
  write_prediction_csv(ensemble_grid, file.path(root, "grid_predictions_bootstrap_ensemble.csv"))
  write_prediction_csv(single_external, file.path(root, "external_quadrat_predictions_single_fit.csv"))
  write_prediction_csv(ensemble_external, file.path(root, "external_quadrat_predictions_bootstrap_ensemble.csv"))
  use_ensemble <- identical(cfg$map_prediction_source, "bootstrap_ensemble")
  write_prediction_csv(if (use_ensemble) ensemble_grid else single_grid,
                       file.path(root, "grid_predictions.csv"))
  write_prediction_csv(if (use_ensemble) ensemble_external else single_external,
                       file.path(root, "external_quadrat_predictions.csv"))
  write_json(list(
    map_prediction_source = cfg$map_prediction_source,
    cv_prediction_source = "single_fit",
    same_estimator_as_primary_cv = !use_ensemble,
    sd_boot_scope = "Dispersion of bootstrap predictions about their weighted ensemble mean",
    note = "Primary single-fit CV errors are not nested validation of the weighted ensemble"
  ), file.path(root, "prediction_source.json"))
}

inverse_rmse_weights <- function(errors) {
  if (!length(errors) || any(!is.finite(errors)) || any(errors < 0)) {
    stop("Invalid bootstrap validation RMSE")
  }
  if (any(errors == 0)) {
    return(as.numeric(errors == 0) / sum(errors == 0))
  }
  # Normalise before summing to reduce overflow for extremely small positive RMSE.
  inverse <- min(errors) / errors
  inverse / sum(inverse)
}
