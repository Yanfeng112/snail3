# Optional environmental-input and mesh sensitivity analyses.
run_environment_ablation <- function(cfg) {
  variants <- cfg$sensitivity$environment_variants
  if (is.null(variants) || length(variants) < 3) {
    stop("Provide raw, downscaled and corrected table suffixes in sensitivity.environment_variants")
  }
  root <- file.path(cfg$output, "environment_ablation")
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  all <- list()
  for (region in unlist(cfg$regions)) {
    reference <- read_table(cfg, region, "survey", "corrected")
    tables <- lapply(variants, function(suffix) read_table(cfg, region, "survey", suffix))
    for (data in tables) {
      check_pair(reference, data, "survey_id")
      if (!isTRUE(all.equal(reference$density, data$density))) {
        stop("Environmental ablation variants have different outcomes")
      }
    }
    mesh <- get_mesh(cfg, region)
    folds <- get_or_create_folds(reference, cfg, region, root)
    signature <- analysis_signature(cfg, list(tables = tables, folds = folds), mesh)
    for (repeat_id in 1:10) {
      for (fold in 1:10) {
        path <- file.path(root, paste0(region, "_", repeat_id, "_", fold, ".rds"))
        result <- load_checkpoint(path, signature)
        if (is.null(result)) {
          ids <- folds$survey_id[folds$repeat_id == repeat_id & folds$fold == fold]
          validation <- reference$survey_id %in% ids
          selected <- select_spde(reference[!validation, ], mesh, cfg)
          result <- do.call(rbind, lapply(names(tables), function(name) {
            data <- tables[[name]]
            fit <- fit_spde(data[!validation, ], data[validation, ], selected$vars, mesh, cfg)
            prediction <- back_transform(fit$new, cfg$negative_prediction_policy)
            data.frame(region = region, repeat_id = repeat_id, fold = fold,
                       variant = name, survey_id = data$survey_id[validation],
                       observed = data$density[validation], predicted = prediction)
          }))
          save_checkpoint(list(signature = signature, value = result), path)
        }
        all[[length(all) + 1]] <- result
      }
    }
  }
  write.csv(do.call(rbind, all), file.path(root, "paired_predictions.csv"), row.names = FALSE)
}

run_mesh_sensitivity <- function(cfg) {
  # Full-development DIC comparison is descriptive; it is not external validation.
  root <- file.path(cfg$output, "mesh_sensitivity")
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  result <- list()
  for (region in unlist(cfg$regions)) {
    paths <- cfg$sensitivity$mesh_candidates[[region]]
    if (is.null(paths) || length(paths) < 2) {
      stop("Provide at least two archived mesh candidates for ", region)
    }
    data <- read_table(cfg, region, "survey", "corrected")
    selected <- select_spde(data, get_mesh(cfg, region), cfg)
    for (name in names(paths)) {
      mesh <- readRDS(paths[[name]])
      started <- proc.time()[["elapsed"]]
      fit <- fit_spde(data, data[1, , drop = FALSE], selected$vars, mesh, cfg)
      elapsed <- proc.time()[["elapsed"]] - started
      result[[length(result) + 1]] <- data.frame(
        region = region, mesh = name, vertices = mesh$n,
        elapsed_seconds = elapsed, fit$diagnostics
      )
      write_fit_audit(fit, cfg, file.path(root, paste(region, name, sep = "_")))
    }
  }
  write.csv(do.call(rbind, result), file.path(root, "development_fit_comparison.csv"),
            row.names = FALSE)
}
