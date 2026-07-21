"""Detect whether an intensity distribution actually contains a subpopulation.

The default expectation is **no subpopulation**. A forced threshold (Otsu, a
percentile) always splits a distribution in two and so manufactures a
"subpopulation" even from a single blob. This module instead *tests* for one and
only quantifies it when the data genuinely shows two separated modes.

Method (on log10 intensity, where these distributions are roughly Gaussian):

1. Fit a 1-component and a 2-component Gaussian model (1-D EM, dependency-free)
   and compare by BIC - does the data prefer two components at all?
2. Require the fitted 2-component mixture to be **genuinely bimodal**: a real
   density valley must exist between the two means (two Gaussians that overlap
   too much sum to a single hump - no valley, no subpopulation). This is the
   guard that stops large n from "detecting" a slightly non-Gaussian blob.
3. Require the smaller mode to be **non-trivial** (default >= 5% of cells), so a
   handful of outliers is not called a subpopulation.

Only if all three hold is a subpopulation reported, together with its size
(fraction of cells in the high mode), where the modes sit, their separation, and
the **antimode** (the valley) as the natural threshold between them - replacing
the arbitrary forced cut.

``diptest`` (Hartigan's dip test), if installed, is reported as an independent
non-parametric second opinion but is not required.

numpy + scipy only.
"""

from __future__ import annotations

import numpy as np

_MIN_N = 50            # below this, don't claim anything
_MIN_WEIGHT = 0.05     # smaller mode must hold at least this fraction
_MIN_DELTA_BIC = 6.0   # 2-component must beat 1-component by this (positive evidence)


def _normpdf(x, m, s):
    s = max(float(s), 1e-12)
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def _em2(x, iters=300, tol=1e-7):
    """1-D two-component Gaussian mixture via EM. Deterministic (quartile init)."""
    x = np.asarray(x, dtype=float)
    sd = x.std() or 1.0
    m = np.percentile(x, [25.0, 75.0]).astype(float)
    if m[0] == m[1]:
        m = np.array([x.min(), x.max()], dtype=float)
    s = np.array([sd, sd], dtype=float)
    w = np.array([0.5, 0.5])
    var_floor = (0.02 * sd) ** 2 + 1e-12
    prev = -np.inf
    for _ in range(iters):
        p0 = w[0] * _normpdf(x, m[0], s[0])
        p1 = w[1] * _normpdf(x, m[1], s[1])
        tot = p0 + p1 + 1e-300
        r0, r1 = p0 / tot, p1 / tot
        n0, n1 = r0.sum(), r1.sum()
        if n0 < 1e-6 or n1 < 1e-6:
            break
        w = np.array([n0, n1]) / x.size
        m = np.array([(r0 * x).sum() / n0, (r1 * x).sum() / n1])
        v0 = max((r0 * (x - m[0]) ** 2).sum() / n0, var_floor)
        v1 = max((r1 * (x - m[1]) ** 2).sum() / n1, var_floor)
        s = np.sqrt(np.array([v0, v1]))
        ll = float(np.log(tot).sum())
        if abs(ll - prev) < tol:
            break
        prev = ll
    order = np.argsort(m)
    return w[order], m[order], s[order], ll


def _bic(loglik, k, n):
    return -2.0 * loglik + k * np.log(n)


def _valley(w, m, s):
    """Density minimum strictly between the two means, if one exists.

    Returns (antimode_log, is_bimodal). The mixture is bimodal only when its
    density actually dips between the modes - two overlapping Gaussians with no
    interior minimum are one hump, i.e. no subpopulation."""
    grid = np.linspace(m[0], m[1], 256)
    dens = w[0] * _normpdf(grid, m[0], s[0]) + w[1] * _normpdf(grid, m[1], s[1])
    j = int(np.argmin(dens))
    is_bimodal = 0 < j < len(grid) - 1 and dens[j] < dens[0] and dens[j] < dens[-1]
    return float(grid[j]), bool(is_bimodal)


def detect_subpopulation(values, min_weight=_MIN_WEIGHT, min_delta_bic=_MIN_DELTA_BIC):
    """Test one intensity distribution for a high-intensity subpopulation.

    ``values`` are linear intensities. Returns a JSON-serializable dict; the key
    field is ``detected`` (bool). When ``detected``, ``antimode`` is the linear
    threshold between the modes and ``fraction_high`` the share of cells above it.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    n = int(v.size)
    rep: dict = {"n": n, "detected": False, "verdict": "unimodal (no subpopulation)"}
    if n < _MIN_N:
        rep["verdict"] = f"too few cells (n={n}) to test"
        return rep
    x = np.log10(v)

    # 1-component
    mu, sd = float(x.mean()), float(x.std() or 1e-9)
    ll1 = float(np.log(_normpdf(x, mu, sd) + 1e-300).sum())
    bic1 = _bic(ll1, 2, n)
    # 2-component
    w, m, s, ll2 = _em2(x)
    bic2 = _bic(ll2, 5, n)
    antimode_log, is_bimodal = _valley(w, m, s)
    sep = float(abs(m[1] - m[0]) / np.sqrt((s[0] ** 2 + s[1] ** 2) / 2.0))
    minw = float(min(w))

    rep.update({
        "delta_bic": round(bic1 - bic2, 2),
        "mode_separation_sd": round(sep, 2),
        "smaller_mode_weight": round(minw, 3),
        "bimodal_density": is_bimodal,
    })
    # Optional Hartigan dip test (non-parametric second opinion).
    try:
        import diptest
        rep["dip_p_value"] = round(float(diptest.diptest(x)[1]), 4)
    except Exception:  # noqa: BLE001 - optional
        pass

    detected = (is_bimodal and minw >= min_weight and (bic1 - bic2) > min_delta_bic)
    if detected:
        antimode = float(10.0 ** antimode_log)
        frac_high = float(np.mean(v > antimode))
        rep.update({
            "detected": True,
            "verdict": "subpopulation detected (two separated modes)",
            "mode_low": round(float(10.0 ** m[0]), 1),
            "mode_high": round(float(10.0 ** m[1]), 1),
            "antimode": antimode,
            "fraction_high": round(frac_high, 4),
            "high_mode_weight": round(float(w[1]), 4),
        })
    return rep


def format_detection(by_group: dict) -> str:
    """One line per group summarizing detection."""
    lines = ["Subpopulation detection (default: none):"]
    for g, r in by_group.items():
        if r.get("detected"):
            lines.append(f"  {g}: DETECTED - {100 * r['fraction_high']:.1f}% in a high mode "
                         f"(> {r['antimode']:.0f}); modes {r['mode_low']:.0f} vs "
                         f"{r['mode_high']:.0f}, separation {r['mode_separation_sd']:.1f} SD")
        else:
            lines.append(f"  {g}: {r['verdict']}"
                         + (f" (separation {r.get('mode_separation_sd')} SD)"
                            if "mode_separation_sd" in r else ""))
    return "\n".join(lines)
