"""Explicit calls for author-supplied raw tables; edit paths/conventions after historical verification."""

import importlib
from pathlib import Path
import pandas as pd

raw = importlib.import_module("python.01_raw_environment")
Path("data/observations").mkdir(parents=True, exist_ok=True)
# Run only the calls matching your real source schemas; example filenames are not supplied data.
# hourly=pd.read_csv('data/raw/hourly_temperature.csv')
# raw.temperature_indicators(hourly,'Asia/Shanghai').to_csv('data/observations/temperature_indicators.csv',index=False)
# monthly=pd.read_csv('data/raw/monthly_precipitation.csv')
# seasons={'spring':[3,4,5],'summer':[6,7,8],'autumn':[9,10,11],'winter':[12,1,2]}
# The example groups winter months in the calendar year; confirm the historical convention first.
# raw.seasonal_precipitation(monthly,seasons,reduction='mean').to_csv('data/observations/seasonal_precipitation.csv',index=False)
# video=pd.read_csv('data/raw/video_annotations.csv')
# raw.video_inundation(video).to_csv('data/observations/video_annual.csv',index=False)
# cells=pd.read_csv('data/raw/reconstruction_cells_1985.csv')
# water=pd.read_csv('data/raw/daily_0800_water_levels_1985.csv')
# annual=raw.water_level_inundation(cells,water,datum_verified=True)
# annual.to_csv('data/raw/inundation_cell_days.csv',index=False)
