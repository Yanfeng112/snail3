# Shared contracts and training-only preprocessing.
load_analysis_packages <- function() {
  required <- c("INLA", "sf", "sp", "gstat", "ranger", "jsonlite", "Matrix", "digest")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop("Install required R packages: ", paste(missing, collapse = ", "))
  }
  for (package in required) {
    suppressPackageStartupMessages(library(package, character.only = TRUE))
  }
}

read_config <- function(path) {
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

write_json <- function(value, path) {
  jsonlite::write_json(value, path, pretty = TRUE, auto_unbox = TRUE,
                      null = "null", digits = 16)
}

write_prediction_csv <- function(data, path) {
  # Preserve binary-double values across the R-to-Python CSV boundary.
  for (name in names(data)) {
    if (is.double(data[[name]])) {
      data[[name]] <- format(data[[name]], digits = 17, scientific = TRUE, trim = TRUE)
    }
  }
  write.csv(data, path, row.names = FALSE)
}

assert_finite <- function(values, label) {
  if (any(!is.finite(values))) {
    stop("Nonfinite values: ", label)
  }
}

back_transform <- function(linear_predictor, policy) {
  assert_finite(linear_predictor, "back-transform input")
  density <- expm1(linear_predictor)
  assert_finite(density, "back-transformed density")
  if (policy == "clip_zero") {
    return(pmax(density, 0))
  }
  if (policy == "retain") {
    return(density)
  }
  stop("Set the historical negative_prediction_policy: retain or clip_zero")
}

metric <- function(observed, predicted) {
  if (length(observed) != length(predicted) || !length(observed)) {
    stop("Metrics require nonempty paired records")
  }
  assert_finite(c(observed, predicted), "metric inputs")
  error <- predicted - observed
  data.frame(n = length(observed), bias = mean(error), mae = mean(abs(error)),
             rmse = sqrt(mean(error^2)), sse = sum(error^2), sae = sum(abs(error)))
}

read_table <- function(cfg, region, table, kind) {
  classes <- switch(table,
    survey = c(survey_id = "character"),
    external = c(quadrat_id = "character", marshland_id = "character"),
    grid = c(cell_id = "numeric")
  )
  filename <- paste0(paste(region, table, kind, sep = "_"), ".csv")
  data <- read.csv(file.path(cfg$tables_directory, filename),
                   stringsAsFactors = FALSE, check.names = FALSE, colClasses = classes)
  assert_finite(as.matrix(data[, c("x", "y")]), "coordinates")
  expected <- c("region", "x", "y", names(cfg$covariates))
  id <- switch(table, survey = "survey_id", external = "quadrat_id", grid = "cell_id")
  expected <- c(expected, id, switch(table, survey = "density", external = c("marshland_id", "live_snails")))
  if (!nrow(data) || length(setdiff(expected, names(data)))) {
    stop("Empty/incomplete ", table, " input: ", filename)
  }
  if (anyNA(data[[id]]) || anyDuplicated(data[[id]]) ||
      any(!nzchar(trimws(as.character(data[[id]]))))) {
    stop("Invalid record identifiers: ", filename)
  }
  if (anyNA(data$region) || any(data$region != region)) {
    stop("Unexpected region labels: ", filename)
  }
  if (table == "survey") {
    assert_finite(data$density, "observed density")
    if (any(data$density < 0)) stop("Negative observed density")
  }
  if (table == "external") {
    assert_finite(data$live_snails, "external snail counts")
    if (any(data$live_snails < 0 | data$live_snails != floor(data$live_snails))) {
      stop("External live_snails must be nonnegative integer counts")
    }
    if (anyNA(data$marshland_id) || any(base::table(data$marshland_id) != 100)) {
      stop("External marshlands need exactly 100 identified quadrats")
    }
  }
  if (table == "grid" && any(!is.finite(data$cell_id) | data$cell_id != floor(data$cell_id) | data$cell_id < 0)) {
    stop("Invalid pixel identifiers")
  }
  data
}

check_pair <- function(first, second, id) {
  same_ids <- identical(as.character(first[[id]]), as.character(second[[id]]))
  same_coordinates <- isTRUE(all.equal(first[, c("x", "y")], second[, c("x", "y")]))
  if (!same_ids || !same_coordinates) {
    stop("Corrected/uncorrected row identities or coordinates differ")
  }
}

validate_config <- function(cfg) {
  keys <- c("xy_crs", "spde_coordinate_divisor", "alpha", "block_origin_m",
            "fixed_effect_mean", "fixed_effect_precision", "intercept_mean",
            "intercept_precision", "gaussian_precision_hyper", "negative_prediction_policy",
            "bootstrap_variable_selection", "baseline_feature_policy",
            "unseen_predictor_policy", "design_rank_policy", "map_prediction_source")
  for (key in keys) {
    if (is.null(cfg[[key]])) {
      stop("Set required analysis parameter before running: ", key)
    }
  }
  if (!cfg$negative_prediction_policy %in% c("retain", "clip_zero")) {
    stop("Invalid negative prediction policy")
  }
  if (cfg$spde_coordinate_divisor <= 0) {
    stop("SPDE coordinate divisor must be positive")
  }
  if (!cfg$bootstrap_variable_selection %in% c("fixed_structure", "training_only")) {
    stop("Set bootstrap_variable_selection to fixed_structure or training_only")
  }
  if (!cfg$baseline_feature_policy %in% c("all", "spde_selected")) stop("Choose baseline_feature_policy")
  if (!cfg$unseen_predictor_policy %in% c("error", "report")) stop("Choose unseen_predictor_policy")
  if (!cfg$design_rank_policy %in% c("error", "report")) stop("Choose design_rank_policy")
  if (!cfg$map_prediction_source %in% c("single_fit", "bootstrap_ensemble")) stop("Choose map_prediction_source")
  numeric_settings <- unlist(cfg[c("alpha", "fixed_effect_precision", "intercept_precision")])
  if (any(!is.finite(numeric_settings)) || cfg$alpha <= 1 || cfg$alpha > 2 ||
      cfg$fixed_effect_precision <= 0 || cfg$intercept_precision <= 0) {
    stop("Invalid two-dimensional Matern smoothness or Gaussian prior precision")
  }
  crs <- sf::st_crs(cfg$xy_crs)
  if (is.na(crs) || sf::st_is_longlat(crs) ||
      !tolower(crs$units_gdal) %in% c("metre", "meter", "m")) {
    stop("Use the analysis projected CRS in metres")
  }
  if (!isTRUE(cfg$historical_settings_verified)) {
    warning("Analysis settings have not been verified; confirm the configuration before interpreting results")
  }
}

prepare_design <- function(train, new, variables, cfg, check_rank = TRUE) {
  # Every fitted mean, SD and transformation is estimated from this training set.
  if (any(make.names(variables) != variables) || anyDuplicated(variables)) {
    stop("Covariate names must be unique syntactic R identifiers")
  }
  if (length(setdiff(variables, names(train))) || length(setdiff(variables, names(new)))) {
    stop("A declared predictor column is missing")
  }
  blueprint <- unlist(lapply(variables, function(variable) {
    levels <- cfg$covariates[[variable]]$levels
    if (is.null(levels)) return(variable)
    paste0(variable, "_L", seq.int(2, length(levels)))
  }))
  if (anyDuplicated(blueprint)) stop("Dummy-column names collide with another predictor")
  support <- list()
  unseen_policy <- if (is.null(cfg$unseen_predictor_policy)) "error" else cfg$unseen_predictor_policy
  means <- scales <- numeric(0)
  train_columns <- new_columns <- list()
  ownership <- character(0)
  for (variable in variables) {
    specification <- cfg$covariates[[variable]]
    if (is.null(specification)) {
      stop("Missing feature definition: ", variable)
    }
    if (is.null(specification$levels)) {
      if (!is.numeric(train[[variable]]) || !is.numeric(new[[variable]])) {
        stop("Continuous predictor is not numeric: ", variable)
      }
      a <- as.numeric(train[[variable]])
      b <- as.numeric(new[[variable]])
      transformation <- specification$transform
      if (is.null(transformation) || !transformation %in% c("identity", "log1p")) {
        stop("Specify a supported continuous transformation for ", variable)
      }
      if (transformation == "log1p") {
        if (any(c(a, b) < 0)) {
          stop("Negative log1p covariate input")
        }
        a <- log1p(a)
        b <- log1p(b)
      }
      assert_finite(c(a, b), variable)
      centre <- mean(a)
      scale <- sd(a)
      if (!is.finite(scale) || scale == 0) {
        unsupported <- any(b != centre)
        support[[variable]] <- list(constant_training = TRUE, unsupported_new_values = unsupported)
        if (unsupported && unseen_policy == "error") {
          stop("Predictor varies only outside training support: ", variable)
        }
        scale <- 1
      }
      train_columns[[variable]] <- (a - centre) / scale
      new_columns[[variable]] <- (b - centre) / scale
      means[variable] <- centre
      scales[variable] <- scale
      ownership[variable] <- variable
    } else {
      levels <- unlist(specification$levels)
      if (length(levels) < 2 || any(!as.character(train[[variable]]) %in% levels) ||
          any(!as.character(new[[variable]]) %in% levels)) {
        stop("Unknown factor level: ", variable)
      }
      unseen <- setdiff(unique(as.character(new[[variable]])), unique(as.character(train[[variable]])))
      support[[variable]] <- list(unseen_levels = unseen)
      if (length(unseen) && unseen_policy == "error") {
        stop("Prediction includes a factor category absent from training: ", variable)
      }
      # The first declared category is the reference, using the author's codebook.
      for (level_id in 2:length(levels)) {
        name <- paste0(variable, "_L", level_id)
        train_columns[[name]] <- as.numeric(as.character(train[[variable]]) == levels[level_id])
        new_columns[[name]] <- as.numeric(as.character(new[[variable]]) == levels[level_id])
        ownership[name] <- variable
      }
    }
  }
  train_matrix <- if (length(train_columns)) {
    as.data.frame(train_columns, check.names = FALSE)
  } else {
    data.frame(row.names = seq_len(nrow(train)))
  }
  new_matrix <- if (length(new_columns)) {
    as.data.frame(new_columns, check.names = FALSE)
  } else {
    data.frame(row.names = seq_len(nrow(new)))
  }
  matrix <- cbind(Intercept = 1, as.matrix(train_matrix))
  rank <- qr(matrix)$rank
  support$design_rank <- list(rank = rank, columns = ncol(matrix))
  if (check_rank && rank < ncol(matrix) && identical(cfg$design_rank_policy, "error")) {
    stop("Training fixed-effect design is rank deficient; inspect constants and collinearity")
  }
  list(train = train_matrix, new = new_matrix, mean = means, sd = scales,
       ownership = ownership, support_audit = support)
}
