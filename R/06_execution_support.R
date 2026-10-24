# Helpers for input-bound checkpoints and auditable fit failures.
analysis_signature <- function(cfg, inputs, mesh = NULL) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    stop("Install digest before running checkpointed analyses")
  }
  scripts <- sort(list.files("R", pattern = "[.]R$", full.names = TRUE))
  source_hashes <- tools::md5sum(c(scripts, "run_analysis.R"))
  digest::digest(
    list(configuration = cfg, inputs = inputs, mesh = mesh, source = source_hashes,
         environment = list(R = R.version.string, INLA = as.character(packageVersion("INLA")),
                            ranger = as.character(packageVersion("ranger")),
                            gstat = as.character(packageVersion("gstat")))),
    algo = "sha256"
  )
}

save_checkpoint <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(pattern = "checkpoint_", tmpdir = dirname(path))
  on.exit(unlink(temporary), add = TRUE)
  payload <- list(format_version = 1L, signature = value$signature, value = value$value,
                  payload_digest = digest::digest(value$value, algo = "sha256"))
  saveRDS(payload, temporary)
  if (!file.rename(temporary, path)) {
    stop("Cannot commit checkpoint: ", path)
  }
}

load_checkpoint <- function(path, signature) {
  if (!file.exists(path)) {
    return(NULL)
  }
  object <- readRDS(path)
  if (!identical(object$format_version, 1L) ||
      !identical(object$payload_digest, digest::digest(object$value, algo = "sha256"))) {
    stop("Malformed or modified checkpoint payload: ", path)
  }
  if (!identical(object$signature, signature)) {
    stop("Checkpoint inputs/settings changed; use a new output directory: ", path)
  }
  object$value
}

record_failure <- function(error, path, context) {
  record <- list(
    status = "failed",
    context = context,
    message = conditionMessage(error),
    call = paste(deparse(conditionCall(error)), collapse = " ")
  )
  write_json(record, path)
  stop(error)
}

validate_model_result <- function(fit) {
  assert_finite(fit$new, "SPDE new-location linear predictors")
  assert_finite(fit$train, "SPDE training linear predictors")
  assert_finite(as.numeric(unlist(fit$diagnostics[1, ], use.names = FALSE)), "SPDE DIC/WAIC/MLL")
  if (any(!is.finite(fit$model$summary.fixed$mean)) ||
      any(!is.finite(fit$model$summary.fixed$sd))) {
    stop("Nonfinite SPDE fixed-effect posterior summaries")
  }
  invisible(fit)
}

validate_cv_coverage <- function(data, cfg) {
  models <- c("OK", "RF", "RF_OK", "SPDE_uncorrected", "SPDE_corrected")
  if (!setequal(unique(data$model), models) ||
      !setequal(unique(data$region), unlist(cfg$regions))) {
    stop("CV output lacks one or more required models/regions")
  }
  key <- c("region", "repeat_id", "fold", "model", "set", "survey_id")
  if (anyDuplicated(data[, key])) {
    stop("Duplicate CV prediction keys")
  }
  for (region in unlist(cfg$regions)) {
    part <- data[data$region == region, ]
    expected_ids <- as.character(read_table(cfg, region, "survey", "corrected")$survey_id)
    reference <- part[part$model == models[1],
                      c("repeat_id", "fold", "set", "survey_id", "observed")]
    reference <- reference[do.call(order, reference[, 1:4]), ]
    rownames(reference) <- NULL
    for (model in models) {
      candidate <- part[part$model == model,
                        c("repeat_id", "fold", "set", "survey_id", "observed")]
      candidate <- candidate[do.call(order, candidate[, 1:4]), ]
      rownames(candidate) <- NULL
      if (!identical(reference, candidate)) {
        stop("Models do not use the same paired records and outcomes")
      }
      x <- part[part$model == model, ]
      evaluations <- unique(x[, c("repeat_id", "fold")])
      expected <- expand.grid(repeat_id = 1:10, fold = 1:10)
      actual <- paste(evaluations$repeat_id, evaluations$fold)
      if (!setequal(actual, paste(expected$repeat_id, expected$fold))) {
        stop("Incomplete 100-evaluation CV output: ", region, "/", model)
      }
      for (repeat_id in 1:10) {
        for (fold in 1:10) {
          evaluation <- x[x$repeat_id == repeat_id & x$fold == fold, ]
          train_ids <- as.character(evaluation$survey_id[evaluation$set == "training"])
          test_ids <- as.character(evaluation$survey_id[evaluation$set == "validation"])
          if (length(intersect(train_ids, test_ids)) ||
              !setequal(c(train_ids, test_ids), expected_ids)) {
            stop("An evaluation has overlapping or incomplete training/validation IDs")
          }
        }
        validation <- x[x$repeat_id == repeat_id & x$set == "validation", ]
        if (!setequal(as.character(validation$survey_id), expected_ids)) {
          stop("A CV repeat does not validate every original survey exactly once")
        }
        if (anyDuplicated(validation$survey_id)) {
          stop("A survey is validated more than once within a CV repeat")
        }
      }
    }
  }
  invisible(TRUE)
}
