"""Spatial variance decomposition table.

Produces: spatial_variance_decomposition.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from seiskit.plot_config import result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_master, FACTORS

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
full = load_master()
fac = FACTORS
full["cell"] = full.groupby(fac).ngroup()

# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------
vc = {}
for tgt in ["log_abs", "f_ratio"]:
    y = full[tgt].values
    gm = y.mean()
    sst = ((y - gm) ** 2).sum()

    def ssm(col):
        return ((full.groupby(col)[tgt].transform("mean").values - gm) ** 2).sum()

    ss_cell = ssm("cell")
    ss_ch = ssm("channel")
    ss_seed = ssm("seed")
    ss_cellch = (
        ((full.groupby(["cell", "channel"])[tgt].transform("mean").values - gm) ** 2).sum()
        - ss_cell
        - ss_ch
    )
    ss_cellseed = (
        ((full.groupby(["cell", "seed"])[tgt].transform("mean").values - gm) ** 2).sum()
        - ss_cell
        - ss_seed
    )
    resid = sst - ss_cell - ss_ch - ss_seed - ss_cellch - ss_cellseed
    vc[tgt] = dict(
        design=ss_cell / sst,
        channel=ss_ch / sst,
        seed=ss_seed / sst,
        design_x_channel=ss_cellch / sst,
        design_x_seed=ss_cellseed / sst,
        other_resid=resid / sst,
    )

vcd = pd.DataFrame(vc).T

for tgt in ["log_abs", "f_ratio"]:
    y = full[tgt].values
    gm = y.mean()
    sst = ((y - gm) ** 2).sum()

    def ssm2(col):
        return ((full.groupby(col)[tgt].transform("mean").values - gm) ** 2).sum()

    ss_seed = ssm2("seed")
    ss_ch = ssm2("channel")
    ss_seedch = (
        ((full.groupby(["seed", "channel"])[tgt].transform("mean").values - gm) ** 2).sum()
        - ss_seed
        - ss_ch
    )
    vc[tgt]["seed_x_channel"] = ss_seedch / sst

vcd = pd.DataFrame(vc).T
vcd["threeway_plus"] = vcd["other_resid"] - vcd["seed_x_channel"]

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
vcd_out = vcd[
    [
        "design",
        "seed",
        "design_x_seed",
        "channel",
        "design_x_channel",
        "seed_x_channel",
        "threeway_plus",
    ]
].copy()
vcd_out.to_csv(result_path("data", "spatial_variance_decomposition.csv"))
print("saved spatial_variance_decomposition.csv")
print((vcd_out * 100).round(2).to_string())
