# Snail density spatial analysis
This repository contains analysis modules for environmental preprocessing, convolutional downscaling, observation-based correction, spatial validation, comparative prediction, bootstrap estimation and density-class area calculation.

The implementation follows the methods described in the study documentation. Configuration templates are provided separately from the study inputs. Raw survey records, environmental rasters, observation networks and mesh objects are not included. Complete the configuration and verify outputs against the study records before using the package to reproduce reported results. Full TensorFlow and R-INLA execution has not been validated with the study inputs in this distribution.

## Repository contents
| Location | Purpose |
|---|---|
| `python/01_raw_environment.py` | Temperature, precipitation and inundation aggregation |
| `python/02_downscaling.py` | CNN downscaling and source-residual interpolation |
| `python/03_correction.py` | GDA, GRA and Sokol observation-based correction |
| `python/04_environment_pipeline.py` | Candidate comparison, calibration and raster export |
| `python/05_extract_covariates.py` | Matched survey, grid and external-validation covariates |
| `python/06_density_area.py` | Density classification and area summaries |
| `python/07_provenance.py` | Environment and input-file checksums |
| `python/08_audit_published_table.py` | Numerical checks of tabulated external-validation pairs |
| `python/09_preflight.py` | Configuration and required-input checks |
| `python/10_audit_predictions.py` | Paired prediction checks and descriptive plots |
| `python/contracts.py` | Input validation rules |
| `python/spatial_validation.py` | CNN window partitioning and overlap diagnostics |
| `python/spatial_integrity.py` | Pixel identity, coordinate and area checks |
| `python/environment_safeguards.py` | Spatial support and physical bounds |
| `python/execution.py` | Checkpoints and file-integrity checks |
| `R/` | Kriging, random forest, INLA-SPDE, spatial CV, bootstrap and summaries |
| `config/` | Environmental, extraction and model configuration templates |
| `tests/` | Numerical, spatial and input-contract regression checks |

## Software
Use Python 3.10 for the TensorFlow/Keras 2.15 dependencies in `requirements.txt`. Install R dependencies with `setup_R.R`, and retain the actual R and package versions used for the analysis. The installer obtains available packages; exact environment restoration requires a verified dependency lockfile.

```bash
python -m pip install -r requirements.txt
Rscript setup_R.R
```

## Data and configuration
The input schema is documented in `DATA_DICTIONARY.md`. Paths in configuration files are relative to the repository root. A `null` entry is an unset parameter; the preflight command reports missing parameters and files.

Before fitting models, specify the analysis CRS, block origin, mesh objects, priors, predictor definitions, negative-prediction policy and output source. `historical_settings_verified` remains false until the configuration has been checked against the study records. This flag does not supply missing parameters.

Environmental templates additionally require a processing mask, land-cover codes, interpolation parameters, CNN partition settings and a physical-bound policy. The processing mask must use the same grid as the 100 m template. Areas without valid prediction support remain NoData.

```bash
python -m python.09_preflight --config config/environment_temperature.json --kind environment
python -m python.09_preflight --config config/extract.json --kind extract
python -m python.09_preflight --config config/study.json --kind study
```

Preflight returns a nonzero status when required inputs or settings are missing. Resolve the reported items before starting model fitting.

## Analysis sequence
Run commands from the repository root. Prepare raw time-series indicators using `python/01_raw_environment.py`; `examples_raw.py` lists the function interfaces. Apply the verified time-zone, seasonal aggregation and vertical-datum conventions before creating input rasters.

```bash
python -m python.04_environment_pipeline --config config/environment_temperature.json
python -m python.04_environment_pipeline --config config/environment_precipitation.json
python -m python.04_environment_pipeline --config config/environment_inundation.json
python -m python.05_extract_covariates --config config/extract.json
Rscript run_analysis.R config/study.json cv
Rscript run_analysis.R config/study.json bootstrap
Rscript run_analysis.R config/study.json summary
python -m python.06_density_area --config config/study.json
python -m python.10_audit_predictions --predictions outputs/cv/all_paired_predictions.csv
python -m python.07_provenance --root . --output outputs/provenance.json
```

Run temperature indicators and seasonal precipitation indicators separately with their matching input years and observation tables. Video-based calibration is retrospective: transfer to earlier years does not constitute prospective temporal validation.

## Analysis choices
- The density response is `log1p(density)` and the point prediction is `expm1(linear_predictor)`. This inverse transformation is distinct from a posterior expected density with a variance correction.
- Spatial CV uses 20 km blocks, ten folds and ten repeats. Standardisation and variable selection use the training subset. All five models use matched records and saved fold assignments.
- `baseline_feature_policy` specifies whether RF uses all candidate predictors or the SPDE-selected subset. Report the selected strategy with the comparison results.
- CNN partitions can use `legacy_cell` or `buffered_blocks`. The buffered mode removes direct overlap between 5×5 training and validation windows. It does not guarantee independence of the underlying climate process or resampled source pixels.
- `bootstrap_variable_selection` chooses fixed covariate structure or training-only reselection within each resample. The weighting errors are distinct from primary spatial CV results.
- `map_prediction_source` selects `single_fit` or `bootstrap_ensemble` for maps and external evaluation. Both outputs are retained. Primary five-model CV evaluates the single-fit estimator and is not nested validation of the weighted ensemble.
- `sd_boot` is the weighted dispersion of bootstrap point predictions. It does not include environmental-input error or full posterior predictive uncertainty.
- External evaluation averages all 100 quadrats within each marshland, including zero-snail quadrats, and uses the nine marshland means as evaluation units.
- `physical_bound_policy` chooses rejection or explicit clipping of nonphysical environmental values. Use one policy in development validation, final testing and exported rasters.
- `unseen_predictor_policy` and `design_rank_policy` control unsupported predictions and rank-deficient fixed-effect designs. Selecting `report` records the condition; it does not establish statistical identifiability.

`config/study_strict.json` contains stricter analysis choices and uses a separate output directory. It still requires study-specific files and unset parameters. Changing a modelling or preprocessing choice requires a new analysis run and corresponding method documentation.

## Density classes and area
Densities are rounded to three decimals with ROUND_HALF_UP before classification:

| Class | Density, individuals per 0.11 m² |
|---|---|
| Zero | 0 |
| Very low | 0.001–0.020 |
| Low | 0.021–0.100 |
| Moderate | 0.101–0.300 |
| High | 0.301–0.500 |
| Very high | >0.500 |

Zero-density cells remain in the prediction domain but are excluded from positive-density habitat area. Pixel IDs, coordinates, region labels, masks and grid resolution are checked before area calculation. Area calculations require metre-based coordinates. GeoTIFF outputs use float64 and NaN NoData to preserve numerical values near classification thresholds.

## Validation
```bash
python -m unittest discover -s tests -v
Rscript tests/test_R_core.R
python demo_synthetic.py
```

The Python regression suite contains 46 checks; 45 passed and one TensorFlow-dependent check was skipped in the validation environment. R files passed syntax parsing, but neither the R contract script nor INLA fitting was executed in that environment. Full CNN training and complete study-result reproduction remain unverified.

`demo_synthetic.py` fits a random forest to synthetic data and checks threshold calculations. Its 1 km toy blocks are only for a compact execution example. It does not produce CNN/INLA study results.

Optional environmental-input ablation and mesh sensitivity require additional matched inputs:

```bash
Rscript run_analysis.R config/study.json ablation
Rscript run_analysis.R config/study.json mesh
```

Retain data access information, the final configuration, software versions, mesh files, fold assignments and paired predictions with the archived analysis release.
