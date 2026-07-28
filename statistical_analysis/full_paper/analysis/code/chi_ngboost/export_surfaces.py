"""Export NGBoost predictive surfaces on the unique factorial grid.

Builds the 243 × 101 (cell × node) design without seed replication, predicts
μ, σ, and Normal quantiles τ ∈ {0.05, 0.50, 0.95}, and writes CSVs under
figure_dir("chi_ngboost", "surfaces") for downstream symbolic regression.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FACTORS,
    FEATURES,
    METRICS,
    N_CELLS,
    N_NODES,
    add_design_columns,
    factorial_grid,
    load_ratios,
    models_dir,
    surfaces_dir,
)
from train_ngboost import predict_params  # noqa: E402

warnings.filterwarnings("ignore")

SR_TAUS = (0.05, 0.50, 0.95)


def main() -> None:
    out = surfaces_dir()
    print("Loading ratios and building factorial grid …")
    df_full = add_design_columns(load_ratios())
    grid = factorial_grid(df_full)
    X = grid[FEATURES].to_numpy(dtype=float)
    n_cells = int(grid["cell"].nunique())
    n_nodes = int(grid["node"].nunique())
    print(f"Grid shape={grid.shape}  cells={n_cells}  nodes={n_nodes}")

    rows_meta = []
    for metric in METRICS:
        mpath = models_dir() / f"ngboost_{metric}.pkl"
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing {mpath}")
        model: NGBRegressor = joblib.load(mpath)
        mu, sigma = predict_params(model, X)
        sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
        mu = np.asarray(mu, dtype=float)

        surf = grid.copy()
        surf["mu"] = mu
        surf["median"] = mu  # Normal: median ≡ mean
        surf["sigma"] = sigma
        surf["log_sigma"] = np.log(sigma)
        for tau in SR_TAUS:
            z = float(stats.norm.ppf(tau))
            surf[f"q{int(tau * 100):02d}"] = mu + sigma * z

        path = out / f"ngboost_surface_{metric}.csv"
        surf.to_csv(path, index=False)
        rows_meta.append(
            {
                "metric": metric,
                "n_rows": len(surf),
                "n_cells": n_cells,
                "n_nodes": n_nodes,
                "path": path.name,
            }
        )
        print(f"Wrote {path}")

    meta = pd.DataFrame(rows_meta)
    lines = [
        "# NGBoost factorial-grid surfaces",
        "",
        "## Definitions",
        "",
        r"- Grid: unique combinations of the five factorial factors × node "
        rf"(\(N_{{\mathrm{{cells}}}}\times N_x = {N_CELLS}\times {N_NODES}\)); seed axis dropped.",
        r"- Surfaces are NGBoost predictive quantities on \(Y=\ln\chi\), not observed responses.",
        r"- Under Normal NGBoost, \(\mathrm{median}(\mathbf{x})\equiv\mu(\mathbf{x})\) and "
        r"\(q_\tau(\mathbf{x})=\mu(\mathbf{x})+\sigma(\mathbf{x})\,z_\tau\).",
        r"- Exported columns: `mu`, `median`, `sigma`, `log_sigma`, `q05`, `q50`, `q95` "
        r"plus design (`cell`, factors, z-scored FEATURES).",
        r"- Intended consumer: `chi_sr` symbolic regression (engineering approximations).",
        "",
        "## Files",
        "",
        meta.to_markdown(index=False),
        "",
        "## Design columns",
        "",
        f"- Factors: {', '.join(FACTORS)}",
        f"- Features: {', '.join(FEATURES)}",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(meta.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
