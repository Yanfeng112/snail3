# Input data dictionary
Coordinates must use the configured projected CRS in metres. Missing identifiers, nonfinite required values and inconsistent row identities are rejected.

## Survey table
Path example: `data/raw/survey_2023.csv`.

| Column | Definition |
|---|---|
| `survey_id` | Unique habitat identifier, stored as text |
| `region` | `dongting`, `poyang` or `anhui` |
| `x`, `y` | Projected survey coordinates |
| `density` | Mean live-snail count per 0.11 m² quadrat, including zero-snail quadrats |

## External-validation table
Path example: `data/raw/external_quadrats.csv`.

| Column | Definition |
|---|---|
| `quadrat_id` | Unique quadrat identifier |
| `marshland_id` | Marshland identifier; three marshlands per region |
| `region` | Study-region code |
| `x`, `y` | Projected quadrat coordinates |
| `live_snails` | Nonnegative integer live-snail count in a 0.11 m² quadrat |

Include 100 distinct quadrats for each marshland, including observed zeros. Model-development observations and external-validation observations must have independently documented sources. Different identifiers alone do not establish independence.

## Environmental observations
One CSV per matched indicator, with columns `site_id,period,x,y,observed`. Use consistent string period keys in the observation table and raster configuration. Each site-period combination must be unique, and a site must retain fixed coordinates. Sites shared by temperature or precipitation indicators use the same saved development/test partition.

## Raw temporal inputs
| Function | Required columns |
|---|---|
| `temperature_indicators` | `site_id,time_utc,temp_c` |
| `seasonal_precipitation` | `site_id,year,month,precip_mm` |
| `video_inundation` | `site_id,date,hour,inundated` |
| `water_level_inundation`, grid input | `cell_id,x,y,elevation_1985_m` |
| `water_level_inundation`, station input | `station_id,x,y,date,level_1985_m` |

Temperature records must be exact hourly timestamps. Video observations use 08:00, 12:00 and 16:00 local observations, coded 0/1. Water-level input represents daily 08:00 values. Normalise calendar dates, resolve duplicates and document missing-data handling. Water levels and terrain elevations must actually be converted to the same vertical datum before the inundation calculation.

## Spatial inputs
- Single-band rasters for each indicator and period.
- A 100 m projected grid template and a separate positive-valued processing mask.
- Nonoverlapping regional prediction masks on the common grid.
- Auxiliary rasters and categorical codebooks, including a defined vegetation reference category.
- Mesh objects in the coordinate units specified by `spde_coordinate_divisor`, or the boundary and explicit mesh-building parameters.

Covariate extraction exports corrected and uncorrected survey, grid and external tables with identical identifiers, coordinates and row order. Additional downscaled-only tables are required for environmental-input ablation.

## Included tabulated reference
`data/published_table3.csv` contains nine rounded marshland-level pairs for numerical checks. It is not the survey dataset or the 900-quadrat validation dataset, and is not used to fit the prediction models.
