"""UbiQTree for LightGBM Boosters (Dubey et al., arXiv:2508.09639).

Implements Algorithms 1–5 from the paper:

1. Dirichlet-weighted tree sampling (softmax tree weights × Dir(α w))
2. Constrained TreeSHAP (per-tree SHAP mean + 0.5·diag(Σ) adjustment)
3. SHAP variance decomposition → aleatoric / epistemic / entanglement
4. Uncertainty aggregation → μ, σ, CI, entropy, sign stability
5. End-to-end E_SHAP pipeline

Plus a Dempster–Shafer Bel/Pl gap (ignorance / conflict) on the empirical
SHAP distribution across Dirichlet samples.

LightGBM models are stored as ``Booster`` objects; trees are extracted via
``model_to_string`` surgery (byte-accurate ``tree_sizes``). Per-tree SHAP is
precomputed once and reused for all Dirichlet draws.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy.stats import entropy as scipy_entropy

from shap import TreeExplainer

# ---------------------------------------------------------------------------
# LightGBM tree extraction
# ---------------------------------------------------------------------------


def extract_trees(booster: lgb.Booster, indices: list[int] | np.ndarray) -> lgb.Booster:
    """Return a new Booster containing only the selected trees (0-based)."""
    text = booster.model_to_string()
    m0 = re.search(r"^Tree=0", text, flags=re.M)
    if m0 is None:
        raise ValueError("Booster model string has no Tree=0 block")
    pm = re.search(r"^parameters:", text, flags=re.M)
    if pm is None:
        raise ValueError("Booster model string has no parameters: footer")
    header = text[: m0.start()]
    trees_section = text[m0.start() : pm.start()]
    footer = text[pm.start() :]

    tree_texts = re.split(r"(?=^Tree=\d+)", trees_section, flags=re.M)
    tree_texts = [t for t in tree_texts if t.startswith("Tree=")]
    n_trees = len(tree_texts)
    indices = [int(i) for i in indices]
    if any(i < 0 or i >= n_trees for i in indices):
        raise IndexError(f"tree index out of range (n_trees={n_trees})")

    selected: list[str] = []
    for new_i, ti in enumerate(indices):
        t = re.sub(r"^Tree=\d+", f"Tree={new_i}", tree_texts[ti], count=1)
        selected.append(t)
    sizes = [len(t.encode("utf-8")) for t in selected]
    new_header_lines = []
    for line in header.split("\n"):
        if line.startswith("tree_sizes="):
            new_header_lines.append("tree_sizes=" + " ".join(map(str, sizes)))
        else:
            new_header_lines.append(line)
    new_text = "\n".join(new_header_lines) + "".join(selected) + footer
    return lgb.Booster(model_str=new_text)


def tree_gain_weights(booster: lgb.Booster, beta: float = 5.0) -> np.ndarray:
    """Softmax weights from per-tree total split gain (boosting analogue of OOB)."""
    dump = booster.dump_model()
    gains = []
    for t in dump["tree_info"]:
        g = t.get("split_gain", None)
        if g is None or (isinstance(g, list) and len(g) == 0):
            gains.append(1.0)
        else:
            gains.append(float(np.sum(np.asarray(g, dtype=float))))
    g = np.asarray(gains, dtype=float)
    g = g / (g.max() + 1e-12)
    w = np.exp(beta * (g - g.max()))
    return w / w.sum()


# ---------------------------------------------------------------------------
# Per-tree SHAP cache
# ---------------------------------------------------------------------------


def per_tree_shap(
    booster: lgb.Booster,
    X: np.ndarray,
    *,
    progress: bool = False,
) -> np.ndarray:
    """Compute TreeSHAP for each tree alone.

    Returns
    -------
    phi_trees : ndarray, shape (n_trees, n_samples, n_features)
    """
    n_trees = booster.num_trees()
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    out = np.empty((n_trees, n, p), dtype=float)
    iterator = range(n_trees)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="per-tree SHAP", leave=False)
        except ImportError:
            pass
    for t in iterator:
        bt = extract_trees(booster, [t])
        sv = TreeExplainer(bt).shap_values(X)
        out[t] = np.asarray(sv, dtype=float)
    return out


# ---------------------------------------------------------------------------
# Algorithms 1–5
# ---------------------------------------------------------------------------


@dataclass
class UbiQResult:
    """Per-feature UbiQTree summaries (aggregated over explain instances)."""

    mean: np.ndarray
    std: np.ndarray
    ci_95: np.ndarray  # (2, n_features)
    entropy: np.ndarray
    sign_stability: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    entanglement: np.ndarray
    belief: np.ndarray
    plausibility: np.ndarray
    ignorance: np.ndarray  # Pl − Bel on empirical support
    # Instance-level arrays for optional plots: (n_samples_dirichlet, n_instances, n_feat)
    samples_adj: np.ndarray | None = None
    # Within-draw tree means / vars: (S, n_instances, n_feat)
    mu_s: np.ndarray | None = None
    var_s: np.ndarray | None = None


def _hist_entropy(samples: np.ndarray, bins: int = 10) -> float:
    hist, edges = np.histogram(samples, bins=bins, density=True)
    probs = hist * np.diff(edges)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(scipy_entropy(probs))


def _sign_stability(samples: np.ndarray) -> float:
    """Fraction of samples matching the sign of the mean."""
    mu = float(np.mean(samples))
    if abs(mu) < 1e-15:
        # near-zero mean: treat zero as positive for stability
        return float(np.mean(np.sign(samples) >= 0))
    return float(np.mean(np.sign(samples) == np.sign(mu)))


def _dst_bel_pl(samples: np.ndarray, n_bins: int = 15) -> tuple[float, float, float]:
    """Dempster–Shafer Bel/Pl on the empirical SHAP histogram.

    Singletons get BPA = bin mass. For the full support interval A = [min, max]:
    Bel(A)=1, Pl(A)=1. More usefully we report Bel/Pl for the central mass
    interval covering the interquartile range of the samples (focal set of
    interest). Ignorance = Pl − Bel on that IQR focal set.

    Construction (singleton BPA on histogram bins):
      Bel(IQR) = mass of bins whose centres lie in IQR
      Pl(IQR)  = 1 − mass of bins wholly outside IQR
    """
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size < 4 or np.allclose(samples, samples[0]):
        return 1.0, 1.0, 0.0
    q25, q75 = np.percentile(samples, [25, 75])
    hist, edges = np.histogram(samples, bins=n_bins, density=False)
    mass = hist / hist.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    in_iqr = (centres >= q25) & (centres <= q75)
    # Bel: only evidence wholly supporting IQR (bins inside)
    bel = float(mass[in_iqr].sum())
    # Pl: everything not wholly against IQR
    wholly_out = (edges[1:] < q25) | (edges[:-1] > q75)
    pl = float(1.0 - mass[wholly_out].sum())
    pl = max(pl, bel)
    ign = float(np.clip(pl - bel, 0.0, 1.0))
    return bel, pl, ign


def dirichlet_sample_indices(
    weights: np.ndarray,
    *,
    n_draws: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Algorithm 1: Dirichlet-weighted categorical resampling of trees.

    Returns ``(n_draws, n_trees)`` integer indices (with replacement).
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    n_trees = len(w)
    out = np.empty((n_draws, n_trees), dtype=int)
    for s in range(n_draws):
        pi = rng.dirichlet(alpha * w)
        # numerical guard
        pi = np.clip(pi, 1e-16, None)
        pi /= pi.sum()
        out[s] = rng.choice(n_trees, size=n_trees, replace=True, p=pi)
    return out


def constrained_treeshap(
    phi_trees_subset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algorithm 2 on a (n_trees_sub, n_instances, n_feat) SHAP stack.

    Returns
    -------
    phi_adj : (n_instances, n_feat)
    mu : (n_instances, n_feat)
    var : (n_instances, n_feat)  — per-feature variance across trees
    """
    # mu across trees
    mu = phi_trees_subset.mean(axis=0)
    var = phi_trees_subset.var(axis=0, ddof=0)
    # Covariance adjustment: 0.5 * diag(Σ) per instance
    # Σ is feature×feature across trees for each instance
    n_t, n_inst, n_feat = phi_trees_subset.shape
    adj = np.zeros((n_inst, n_feat), dtype=float)
    for i in range(n_inst):
        # (n_t, n_feat)
        cov = np.cov(phi_trees_subset[:, i, :], rowvar=False, ddof=0)
        if np.ndim(cov) == 0:
            adj[i] = 0.5 * float(cov)
        else:
            adj[i] = 0.5 * np.diag(cov)
    phi_adj = mu + adj
    return phi_adj, mu, var


def decompose_variance(
    mu_s: np.ndarray,
    var_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algorithm 3.

    Parameters
    ----------
    mu_s, var_s : (S, n_instances, n_feat)

    Returns per-feature (averaged over instances):
      A = E_s[var_s]           aleatoric
      E = Var_s[mu_s]          epistemic
      C = Cov_s(mu_s, var_s)   entanglement
    """
    # Average metrics over instances for stable global summary
    # Instance-wise then mean:
    A_inst = var_s.mean(axis=0)  # (n_inst, feat) — E_s[σ²]
    E_inst = mu_s.var(axis=0, ddof=0)  # (n_inst, feat) — Var_s[μ]
    # Cov across Dirichlet draws for each (instance, feature)
    S, n_inst, n_feat = mu_s.shape
    C_inst = np.empty((n_inst, n_feat), dtype=float)
    for i in range(n_inst):
        for j in range(n_feat):
            C_inst[i, j] = float(np.cov(mu_s[:, i, j], var_s[:, i, j], ddof=0)[0, 1])
    return A_inst.mean(0), E_inst.mean(0), C_inst.mean(0)


def aggregate_uncertainty(phi_samples: np.ndarray) -> dict[str, np.ndarray]:
    """Algorithm 4 on (S, n_instances, n_feat) adjusted SHAP samples.

    Aggregates over Dirichlet draws; reports feature-level means of
    instance-level metrics.
    """
    # Flatten instances into the sample axis for global feature metrics
    S, n_inst, n_feat = phi_samples.shape
    flat = phi_samples.reshape(S * n_inst, n_feat)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0, ddof=0)
    ci = np.percentile(flat, [2.5, 97.5], axis=0)
    H = np.array([_hist_entropy(flat[:, j]) for j in range(n_feat)])
    SS = np.array([_sign_stability(flat[:, j]) for j in range(n_feat)])
    return dict(mean=mean, std=std, ci_95=ci, entropy=H, sign_stability=SS)


def e_shap(
    booster: lgb.Booster,
    X: np.ndarray,
    *,
    phi_trees: np.ndarray | None = None,
    n_draws: int = 200,
    alpha: float = 0.5,
    beta: float = 5.0,
    random_state: int = 0,
    store_samples: bool = True,
    progress: bool = False,
) -> UbiQResult:
    """Algorithm 5 — full UbiQTree end-to-end for one Booster + explain set."""
    X = np.asarray(X, dtype=float)
    if phi_trees is None:
        phi_trees = per_tree_shap(booster, X, progress=progress)
    n_trees = phi_trees.shape[0]
    weights = tree_gain_weights(booster, beta=beta)
    if len(weights) != n_trees:
        weights = np.ones(n_trees) / n_trees

    rng = np.random.default_rng(random_state)
    idx = dirichlet_sample_indices(weights, n_draws=n_draws, alpha=alpha, rng=rng)

    n_inst, n_feat = X.shape[0], phi_trees.shape[2]
    mu_s = np.empty((n_draws, n_inst, n_feat), dtype=float)
    var_s = np.empty((n_draws, n_inst, n_feat), dtype=float)
    adj_s = np.empty((n_draws, n_inst, n_feat), dtype=float)

    iterator = range(n_draws)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Dirichlet draws", leave=False)
        except ImportError:
            pass

    for s in iterator:
        subset = phi_trees[idx[s]]  # (n_trees, n_inst, n_feat)
        phi_adj, mu, var = constrained_treeshap(subset)
        adj_s[s] = phi_adj
        mu_s[s] = mu
        var_s[s] = var

    A, E, C = decompose_variance(mu_s, var_s)
    agg = aggregate_uncertainty(adj_s)

    # DST Bel/Pl/ignorance from Dirichlet-sample SHAP (flattened)
    flat = adj_s.reshape(n_draws * n_inst, n_feat)
    bel = np.empty(n_feat)
    pl = np.empty(n_feat)
    ign = np.empty(n_feat)
    for j in range(n_feat):
        bel[j], pl[j], ign[j] = _dst_bel_pl(flat[:, j])

    return UbiQResult(
        mean=agg["mean"],
        std=agg["std"],
        ci_95=agg["ci_95"],
        entropy=agg["entropy"],
        sign_stability=agg["sign_stability"],
        aleatoric=A,
        epistemic=E,
        entanglement=C,
        belief=bel,
        plausibility=pl,
        ignorance=ign,
        samples_adj=adj_s if store_samples else None,
        mu_s=mu_s if store_samples else None,
        var_s=var_s if store_samples else None,
    )


def result_to_rows(
    result: UbiQResult,
    feature_names: list[str],
    *,
    extra: dict[str, Any] | None = None,
) -> list[dict]:
    """Flatten a UbiQResult into CSV-ready row dicts."""
    rows = []
    extra = extra or {}
    for j, f in enumerate(feature_names):
        rows.append(
            {
                **extra,
                "factor": f,
                "mean_shap": float(result.mean[j]),
                "std_shap": float(result.std[j]),
                "ci_low": float(result.ci_95[0, j]),
                "ci_high": float(result.ci_95[1, j]),
                "entropy": float(result.entropy[j]),
                "sign_stability": float(result.sign_stability[j]),
                "aleatoric": float(result.aleatoric[j]),
                "epistemic": float(result.epistemic[j]),
                "entanglement": float(result.entanglement[j]),
                "belief": float(result.belief[j]),
                "plausibility": float(result.plausibility[j]),
                "ignorance": float(result.ignorance[j]),
            }
        )
    return rows
