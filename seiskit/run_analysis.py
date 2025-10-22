"""Simple runner functions for a single-case analysis.

These functions provide a small API that example scripts can call. Heavy
OpenSees operations are kept in functions so they can be mocked in tests.
"""

from pathlib import Path

from seiskit.utils import compute_ricker

try:
    import openseespy.opensees as ops  # type: ignore
except Exception:  # pragma: no cover
    ops = None

import numpy as np


def write_time_series(path: str, samples: np.ndarray, dt: float) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # OpenSees Path requires headerless whitespace-separated values
    with p.open("w", encoding="utf-8") as f:
        for v in samples:
            f.write(f"{v}\n")


def run_case_basic(
    run_id: str,
    output_dir: str = "results",
    Vs: float = 200.0,
    poiss: float = 0.3,
    rho: float = 2100.0,
    Ly: float = 140.0,
    Lx: float = 260.0,
    hx: float = 5.0,
    duration: float = 15.0,
    dt: float = 0.001,
):
    """High-level entry point for the simple homogeneous case.

    This function prepares a Ricker time series and delegates to OpenSees
    setup. If OpenSees is not available it still writes the time series and
    returns the path where results would be stored.
    """
    run_output_path = Path(output_dir) / run_id
    run_output_path.mkdir(parents=True, exist_ok=True)

    # create ricker
    samples = compute_ricker(0.75, 1.4 + dt, duration, dt)
    ts_path = run_output_path / "ricker.txt"
    write_time_series(str(ts_path), samples, dt)

    if ops is None:
        return {"status": "no-opensees", "ts": str(ts_path)}

    from seiskit.analysis import perform_analysis_spatial as perform_analysis

    # Create homogeneous material property arrays
    ndivx = int(Lx / hx)
    ndivy = int(Ly / hx)
    vs_data = np.full((ndivy, ndivx), Vs)
    rho_data = np.full((ndivy, ndivx), rho)
    nu_data = np.full((ndivy, ndivx), poiss)

    res = perform_analysis(
        run_id=run_id,
        vs_data=vs_data,
        rho_data=rho_data,
        nu_data=nu_data,
        output_dir=output_dir,
        Ly=Ly,
        Lx=Lx,
        hx=hx,
        duration=duration,
        dt=dt,
    )
    
    # Return consistent status format
    if res.startswith("Finished"):
        return {"status": res, "ts": str(ts_path)}
    else:
        return {"status": res, "ts": str(ts_path)}
