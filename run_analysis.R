# Run from the repository root; paths in JSON remain relative to this directory.
arguments <- commandArgs(trailingOnly = TRUE)
modes <- c("cv", "bootstrap", "summary", "all", "ablation", "mesh")
if (length(arguments) != 2 || !arguments[2] %in% modes) {
  stop("Usage: Rscript run_analysis.R config/study.json [cv|bootstrap|summary|all|ablation|mesh]")
}
for (file in sort(list.files("R", pattern = "[.]R$", full.names = TRUE))) {
  source(file)
}
load_analysis_packages()
cfg <- read_config(arguments[1])
validate_config(cfg)
dir.create(cfg$output, recursive = TRUE, showWarnings = FALSE)
writeLines(capture.output(sessionInfo()), file.path(cfg$output, "sessionInfo.txt"))
write_json(cfg, file.path(cfg$output, "configuration_used.json"))
mode <- arguments[2]
if (mode %in% c("cv", "all")) {
  run_cv(cfg)
}
if (mode %in% c("bootstrap", "all")) {
  for (region in unlist(cfg$regions)) {
    bootstrap_predict(cfg, region)
  }
}
if (mode %in% c("summary", "all")) {
  summary_cv(cfg)
  summary_external(cfg)
}
if (mode == "ablation") {
  run_environment_ablation(cfg)
}
if (mode == "mesh") {
  run_mesh_sensitivity(cfg)
}
