"""Audit rounded published aggregate pairs; this is not raw-data validation."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from python.common import metrics, write_json


def main():
    d = pd.read_csv("data/published_table3.csv")
    result = metrics(d.observed, d.predicted)
    r = float(pearsonr(d.observed, d.predicted).statistic)
    result.update(
        pearson_r=r,
        pearson_r_squared=r * r,
        spearman_rho=float(spearmanr(d.observed, d.predicted).statistic),
        scope="Nine rounded Table 3 marshland means; not the unavailable raw quadrats.",
    )
    write_json("validation/published_table3_audit.json", result)
    print(result)


if __name__ == "__main__":
    main()
