# Install analysis dependencies and record the resulting software environment.
packages <- c("sf", "sp", "gstat", "ranger", "jsonlite", "Matrix", "renv", "digest")
needed <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(needed)) {
  install.packages(needed, repos = "https://cloud.r-project.org")
}
if (!requireNamespace("INLA", quietly = TRUE)) {
  install.packages(
    "INLA", dependencies = TRUE,
    repos = c(getOption("repos"), INLA = "https://inla.r-inla-download.org/R/stable")
  )
}
# For a version-locked environment: renv::init(bare=TRUE); renv::snapshot().
writeLines(capture.output(sessionInfo()), "R_environment_installed.txt")
