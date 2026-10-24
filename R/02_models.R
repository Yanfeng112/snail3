# Five models share response transformation, records and held-out partitions.
fit_ok <- function(train, new, response, crs) {
  observed <- train[, c("x", "y")]
  observed$z <- response
  locations <- sf::st_as_sf(observed, coords = c("x", "y"), crs = crs)
  targets <- sf::st_as_sf(new[, c("x", "y"), drop = FALSE],
                         coords = c("x", "y"), crs = crs)
  if (length(unique(response)) < 2) {
    return(rep(mean(response), nrow(new)))
  }
  if (anyDuplicated(train[, c("x", "y")])) {
    stop("Kriging contains coincident training coordinates; resolve the historical aggregation rule")
  }
  empirical <- gstat::variogram(z ~ 1, locations)
  initial <- gstat::vgm(psill = var(response) * .8, model = "Exp",
                        range = max(empirical$dist) / 3, nugget = var(response) * .2)
  variogram <- gstat::fit.variogram(empirical, initial)
  if (isTRUE(attr(variogram, "singular")) || any(variogram$psill < 0) ||
      any(!is.finite(variogram$psill)) ||
      any(!is.finite(variogram$range)) || any(variogram$range[variogram$model != "Nug"] <= 0)) {
    stop("Semivariogram failed; inspect this fold instead of replacing the model silently")
  }
  prediction <- gstat::krige(z ~ 1, locations, targets, model = variogram,
                             nmax = 12, debug.level = 0)$var1.pred
  assert_finite(prediction, "kriging predictions")
  prediction
}

fit_rf <- function(train, new, response) {
  predictors <- ncol(train)
  if (!predictors) {
    stop("RF requires at least one predictor")
  }
  if (!"min.bucket" %in% names(formals(ranger::ranger))) {
    stop("This ranger build lacks min.bucket; install a compatible ranger version")
  }
  set.seed(42)
  model <- ranger::ranger(
    x = train, y = response, num.trees = 500,
    mtry = max(1, floor(predictors / 3)), min.node.size = 5, min.bucket = 5,
    seed = 42, num.threads = 1, write.forest = TRUE
  )
  list(new = predict(model, new, num.threads = 1)$predictions,
       train = predict(model, train, num.threads = 1)$predictions, model = model)
}

fit_spde <- function(train, new, variables, mesh, cfg) {
  design <- prepare_design(train, new, variables, cfg)
  # Explicit parameterisation: theta1=log(tau), theta2=log(kappa), independent N(0,1).
  spde <- INLA::inla.spde2.matern(
    mesh, alpha = cfg$alpha,
    B.tau = matrix(c(0, 1, 0), 1, 3), B.kappa = matrix(c(0, 0, 1), 1, 3),
    theta.prior.mean = c(0, 0), theta.prior.prec = diag(2)
  )
  index <- INLA::inla.spde.make.index("spatial", spde$n.spde)
  projection_train <- INLA::inla.spde.make.A(
    mesh, loc = as.matrix(train[, c("x", "y")]) / cfg$spde_coordinate_divisor
  )
  projection_new <- INLA::inla.spde.make.A(
    mesh, loc = as.matrix(new[, c("x", "y")]) / cfg$spde_coordinate_divisor
  )
  if (any(abs(Matrix::rowSums(projection_train) - 1) > 1e-6) ||
      any(abs(Matrix::rowSums(projection_new) - 1) > 1e-6)) {
    stop("Locations outside Mesh 3 or inconsistent mesh coordinate units/CRS")
  }
  fixed_train <- data.frame(Intercept = rep(1, nrow(train)), design$train,
                            check.names = FALSE)
  fixed_new <- data.frame(Intercept = rep(1, nrow(new)), design$new, check.names = FALSE)
  observed_stack <- INLA::inla.stack(
    data = list(z = log1p(train$density)), A = list(1, projection_train),
    effects = list(fixed_train, index), tag = "train"
  )
  prediction_stack <- INLA::inla.stack(
    data = list(z = rep(NA_real_, nrow(new))), A = list(1, projection_new),
    effects = list(fixed_new, index), tag = "new"
  )
  stack <- INLA::inla.stack(observed_stack, prediction_stack)
  features <- colnames(design$train)
  rhs <- paste(c("0", "Intercept", features, "f(spatial, model = spde)"), collapse = " + ")
  formula <- as.formula(paste("z ~", rhs), env = environment())
  likelihood_prior <- cfg$gaussian_precision_hyper
  if (!is.null(likelihood_prior$param)) {
    likelihood_prior$param <- unlist(likelihood_prior$param)
  }
  # Intercept is an explicit covariate in this formula, so its prior is named explicitly.
  fixed_names <- c("Intercept", features)
  fixed_prior <- list(
    mean = as.list(setNames(c(cfg$intercept_mean,
                             rep(cfg$fixed_effect_mean, length(features))), fixed_names)),
    prec = as.list(setNames(c(cfg$intercept_precision,
                             rep(cfg$fixed_effect_precision, length(features))), fixed_names))
  )
  model <- INLA::inla(
    formula, family = "gaussian", data = INLA::inla.stack.data(stack),
    control.predictor = list(A = INLA::inla.stack.A(stack), compute = TRUE),
    control.fixed = fixed_prior, control.family = list(hyper = list(prec = likelihood_prior)),
    control.compute = list(dic = TRUE, waic = TRUE, mlik = TRUE),
    num.threads = "1:1", verbose = FALSE
  )
  observed_rows <- INLA::inla.stack.index(stack, "train")$data
  prediction_rows <- INLA::inla.stack.index(stack, "new")$data
  result <- list(
    model = model, new = model$summary.linear.predictor[prediction_rows, "mean"],
    train = model$summary.linear.predictor[observed_rows, "mean"],
    design = design, vars = variables,
    diagnostics = data.frame(dic = model$dic$dic, waic = model$waic$waic,
                             mlik = model$mlik[1, 1])
  )
  validate_model_result(result)
  result
}

select_spde <- function(train, mesh, cfg) {
  variables <- names(cfg$covariates)
  # The dummy prediction row contains no held-out covariate or outcome information.
  fit <- fit_spde(train, train[1, , drop = FALSE], variables, mesh, cfg)
  history <- data.frame(step = 0, removed = "", dic = fit$diagnostics$dic)
  step <- 0
  repeat {
    coefficients <- fit$model$summary.fixed
    negligible <- variables[vapply(variables, function(variable) {
      columns <- names(fit$design$ownership)[fit$design$ownership == variable]
      length(columns) > 0 && all(abs(coefficients[columns, "mean"]) < .001)
    }, logical(1))]
    if (!length(negligible)) {
      break
    }
    best <- NULL
    removed <- NULL
    for (variable in negligible) {
      trial <- fit_spde(train, train[1, , drop = FALSE],
                        setdiff(variables, variable), mesh, cfg)
      if (trial$diagnostics$dic < fit$diagnostics$dic &&
          (is.null(best) || trial$diagnostics$dic < best$diagnostics$dic)) {
        best <- trial
        removed <- variable
      }
    }
    if (is.null(best)) {
      break
    }
    variables <- setdiff(variables, removed)
    fit <- best
    step <- step + 1
    history <- rbind(history, data.frame(step = step, removed = removed,
                                         dic = fit$diagnostics$dic))
  }
  list(vars = variables, history = history)
}

compare_models <- function(train_c, train_u, new_c, new_u, mesh, cfg, variables) {
  response <- log1p(train_c$density)
  forest_variables <- if (identical(cfg$baseline_feature_policy, "all")) {
    names(cfg$covariates)
  } else {
    variables
  }
  design <- prepare_design(train_c, new_c, forest_variables, cfg, check_rank = FALSE)
  forest <- fit_rf(design$train, design$new, response)
  corrected <- fit_spde(train_c, new_c, variables, mesh, cfg)
  uncorrected <- fit_spde(train_u, new_u, variables, mesh, cfg)
  prediction <- list(
    OK = fit_ok(train_c, new_c, response, cfg$xy_crs),
    RF = forest$new,
    RF_OK = forest$new + fit_ok(train_c, new_c, response - forest$train, cfg$xy_crs),
    SPDE_uncorrected = uncorrected$new,
    SPDE_corrected = corrected$new
  )
  lapply(prediction, back_transform, policy = cfg$negative_prediction_policy)
}
