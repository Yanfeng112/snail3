# One checkpoint is committed only after all five models finish the same fold.
run_cv <- function(cfg) {
  root <- file.path(cfg$output, "cv")
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  all_predictions <- list()
  for (region in unlist(cfg$regions)) {
    corrected <- read_table(cfg, region, "survey", "corrected")
    uncorrected <- read_table(cfg, region, "survey", "uncorrected")
    check_pair(corrected, uncorrected, "survey_id")
    if (!isTRUE(all.equal(corrected$density, uncorrected$density))) {
      stop("Corrected/uncorrected response values differ")
    }
    if (any(corrected$density < 0)) {
      stop("Negative observed density")
    }
    mesh <- get_mesh(cfg, region)
    saveRDS(mesh, file.path(root, paste0(region, "_mesh3.rds")))
    folds <- get_or_create_folds(corrected, cfg, region, root)
    signature <- analysis_signature(
      cfg, list(corrected = corrected, uncorrected = uncorrected, folds = folds), mesh
    )
    for (repeat_id in 1:10) {
      for (fold in 1:10) {
        prefix <- file.path(root, paste(region, repeat_id, fold, sep = "_"))
        path <- paste0(prefix, "_checkpoint.rds")
        result <- load_checkpoint(path, signature)
        if (is.null(result)) {
          result <- tryCatch({
            assignments <- folds[folds$repeat_id == repeat_id, ]
            held_out <- assignments$survey_id[assignments$fold == fold]
            validation <- as.character(corrected$survey_id) %in% as.character(held_out)
            train_rows <- which(!validation)
            validation_rows <- which(validation)
            selected <- select_spde(corrected[train_rows, ], mesh, cfg)
            design_c <- prepare_design(
              corrected[train_rows, ], corrected[validation_rows, ], selected$vars, cfg
            )
            design_u <- prepare_design(
              uncorrected[train_rows, ], uncorrected[validation_rows, ], selected$vars, cfg
            )
            forest_variables <- if (identical(cfg$baseline_feature_policy, "all")) {
              names(cfg$covariates)
            } else {
              selected$vars
            }
            forest_design <- prepare_design(
              corrected[train_rows, ], corrected[validation_rows, ], forest_variables,
              cfg, check_rank = FALSE
            )
            targets <- c(validation_rows, train_rows)
            prediction <- compare_models(
              corrected[train_rows, ], uncorrected[train_rows, ],
              corrected[targets, ], uncorrected[targets, ], mesh, cfg, selected$vars
            )
            rows <- lapply(names(prediction), function(model) {
              data.frame(
                region = region, repeat_id = repeat_id, fold = fold, model = model,
                set = c(rep("validation", length(validation_rows)),
                        rep("training", length(train_rows))),
                survey_id = corrected$survey_id[targets],
                observed = corrected$density[targets], predicted = prediction[[model]]
              )
            })
            list(
              predictions = do.call(rbind, rows), selection = selected,
              preprocessing = list(
                corrected_mean = design_c$mean, corrected_sd = design_c$sd,
                uncorrected_mean = design_u$mean, uncorrected_sd = design_u$sd,
                corrected_support = design_c$support_audit,
                uncorrected_support = design_u$support_audit,
                forest_variables = forest_variables,
                forest_mean = forest_design$mean, forest_sd = forest_design$sd,
                forest_support = forest_design$support_audit
              )
            )
          }, error = function(error) {
            record_failure(error, paste0(prefix, "_failure.json"),
                           list(region = region, repeat_id = repeat_id, fold = fold))
          })
          save_checkpoint(list(signature = signature, value = result), path)
        }
        write.csv(result$selection$history, paste0(prefix, "_selection.csv"), row.names = FALSE)
        saveRDS(result$preprocessing, paste0(prefix, "_preprocessing.rds"))
        all_predictions[[length(all_predictions) + 1]] <- result$predictions
        cat(region, "repeat", repeat_id, "fold", fold, "complete\n")
      }
    }
  }
  result <- do.call(rbind, all_predictions)
  validate_cv_coverage(result, cfg)
  write.csv(result, file.path(root, "all_paired_predictions.csv"), row.names = FALSE)
  invisible(result)
}

get_or_create_folds <- function(data, cfg, region, root) {
  path <- file.path(root, paste0(region, "_spatial_folds.csv"))
  if (!file.exists(path)) {
    folds <- make_folds(data, cfg)
    write.csv(folds, path, row.names = FALSE)
    return(folds)
  }
  folds <- read.csv(path, colClasses = c(survey_id = "character"))
  if (!setequal(as.character(folds$survey_id), as.character(data$survey_id)) ||
      anyDuplicated(folds[, c("survey_id", "repeat_id")])) {
    stop("Saved folds do not match the survey IDs")
  }
  expected_blocks <- setNames(block_ids(data, cfg), as.character(data$survey_id))
  if (any(as.character(folds$block_id) !=
          unname(expected_blocks[as.character(folds$survey_id)]))) {
    stop("Saved folds use different coordinates, projection or block origin")
  }
  for (repeat_id in 1:10) {
    part <- folds[folds$repeat_id == repeat_id, ]
    blocks_per_fold <- tapply(part$fold, part$block_id, function(x) length(unique(x)))
    if (nrow(part) != nrow(data) || !setequal(unique(part$fold), 1:10) ||
        any(blocks_per_fold != 1)) {
      stop("Invalid whole-block fold allocation")
    }
  }
  folds
}
