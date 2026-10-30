#!/usr/bin/env python3
"""Tune the methods' settings, on twin experiments of their own, and write results/tuning.json.

    python tune.py                      # every method with a grid
    python tune.py --method "LM"        # the methods whose label contains this

The protocol, fixed in advance so that the tuning cannot be tuned:

- **The truths are not the scored ones**: seeds offset by ``OFFSET``, so a tuned row is not
  fitted to the truths it is scored on.
- **One objective**: the posterior's CRPS of the forecast water cut, a proper score of what the
  forecast is for. Not the misfit, which rewards overfitting.
- **One case and size**: ``coarse``, at ``NE`` members. That the tuning carries over to the other
  cases and sizes is then something the benchmark tests.
- **The same budget**: a setting may be chosen only if it spends no more simulations, on
  average, than the default. (The grids leave the iterations or steps at most the default's,
  but an iterating scheme also spends simulations on the steps it rejects.)
- **The default unless beaten**: a setting replaces the default only if its mean beats the
  default's by more than twice the standard error of their difference (truth by truth).

``methods.py`` adds a tuned row for each method whose default was beaten.
"""

import os

# As in run.py, before numpy is imported
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cases import NE
from methods import METHODS, REFERENCE
from run import commit, experiment, stderr

HERE = Path(__file__).parent
CASE, TRUTHS, OFFSET = "coarse", 20, 10_000  # tuning truth i is drawn with run.TRUTH_SEED + OFFSET + i
OBJECTIVE = "forecast_wwct_crps"  # lower is better


def mda(steps: int, ratio: float) -> dict:
    """An ES-MDA schedule of ``steps`` inflations, each ``ratio`` times the next, whose inverses sum to 1."""
    alphas = np.array([ratio ** (steps - 1 - k) for k in range(steps)], dtype=float)
    alphas *= np.sum(1 / alphas)
    return {"mda": {"tot_assim_steps": steps, "inflation_param": [float(f"{a:.4g}") for a in alphas]}}


def iteration(**kw) -> dict:
    return {"iteration": kw}


def default_first(default, values) -> list:
    return [default] + [v for v in values if v != default]


# The grids: {label of a method: {option: [(description, settings to merge into its dataassim)]}}.
# The first of each option's values is the default: PET's, or the method's in methods.py.
SCHEDULES = [
    (f"{k} steps, α ratio {r}", mda(k, r))
    for k, r in default_first((4, 1), itertools.product((2, 3, 4), (1, 2)))
]
STEP_LENGTHS = [
    (f"γ = {g}, γ max = {m}", iteration(gamma=g, gamma_max=m))
    for g, m in default_first((0.2, 0.5), itertools.product((0.1, 0.2, 0.4), (0.5, 1.0)))
]
GRIDS = {
    "LM-EnRML (4 iterations)": {
        "damping": [
            (f"λ = {lam}, λ factor = {f}", iteration(**{"lambda": lam, "lambda_factor": f}))
            for lam, f in default_first((100, 5), itertools.product((1, 10, 100, 1000), (2, 5, 10)))
        ],
    },
    "GN-EnRML (4 iterations)": {"step": STEP_LENGTHS},
    "GN-EnRML (4 iterations), subspace2 analysis": {"step": STEP_LENGTHS},
    "GN-EnRML (4 iterations), margis analysis": {"step": STEP_LENGTHS},
    "ES-MDA (4 steps, α = 4)": {"schedule": SCHEDULES},
    "ES-MDA (4 steps, α = 4), enif analysis": {
        "schedule": SCHEDULES,
        "neighbourhood": [(f"neighbourhood {k}", {"enif": {"neighbourhood_expansion": k}}) for k in (2, 1, 3)],
    },
    "ES-MDA (4 steps, α = 4), adaptive localization (3σ threshold)": {
        "cutoff": [(f"{c}σ threshold", {"localization": {"cutoff": float(c)}}) for c in (3, 2, 4, 5)],
    },
    "ES-MDA (4 steps, α = 4), distance localization (Gaspari-Cohn, 300 m)": {
        "radius": [(f"{r} m", {"localization": {"radius": float(r)}}) for r in (300, 150, 450, 600)],
    },
}


def merge(base: dict, extra: dict) -> dict:
    """``base``, with ``extra`` merged into it, dicts recursively."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def points(method: dict, grid: dict) -> list:
    """Each point of the grid: its description, and the method's settings at it. The first is the default."""
    out = []
    for combo in itertools.product(*grid.values()):
        settings = method["dataassim"]
        for _, extra in combo:
            settings = merge(settings, extra)
        out.append(("; ".join(d for d, _ in combo), settings))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", action="append", help="label substring; default: all with a grid")
    parser.add_argument("--truths", type=int, default=TRUTHS, help="other than the default: a check, not written")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    by_label = {m["label"]: m for m in METHODS}
    labels = [lab for lab in GRIDS if not args.method or any(s in lab for s in args.method)]
    grid_points = {lab: points(by_label[lab], GRIDS[lab]) for lab in labels}
    jobs = [(lab, k, i) for lab in labels for k in range(len(grid_points[lab])) for i in range(args.truths)]
    print(f"{len(jobs)} experiments: {sum(map(len, grid_points.values()))} settings x {args.truths} truths", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=logging.disable, initargs=(logging.WARNING,)) as pool:
        method_at = lambda lab, k: {"label": lab, "dataassim": grid_points[lab][k][1]}  # noqa: E731
        done = list(pool.map(experiment, *zip(*[(CASE, method_at(lab, k), OFFSET + i, NE) for lab, k, i in jobs])))

    out_file = HERE / "results" / "tuning.json"
    tuning = json.loads(out_file.read_text()) if out_file.exists() else {}
    for lab in labels:
        default = 0  # the grid's first point
        scores = [[d["posterior"][OBJECTIVE] for (lb, kk, _), d in zip(jobs, done) if (lb, kk) == (lab, k)]
                  for k in range(len(grid_points[lab]))]
        runs = [float(np.mean([d["runs"] for (lb, kk, _), d in zip(jobs, done) if (lb, kk) == (lab, k)]))
                for k in range(len(grid_points[lab]))]
        affordable = [k for k in range(len(scores)) if runs[k] <= runs[default]]
        best = min(affordable, key=lambda k: np.mean(scores[k]))
        gain = np.array(scores[default]) - np.array(scores[best])
        chosen = best if gain.mean() > 2 * stderr(gain) else default
        tuning[lab] = {
            "commit": commit(), "case": CASE, "ne": NE, "truths": TRUTHS, "objective": OBJECTIVE,
            "default": default, "chosen": chosen,
            "points": [{"description": desc, "dataassim": s, "runs": r, OBJECTIVE: sc}
                       for (desc, s), r, sc in zip(grid_points[lab], runs, scores)],
        }
        print(f"\n{lab}" + ("" if chosen != default else "  (the default stands)"))
        for k, (desc, _) in enumerate(grid_points[lab]):
            d = np.array(scores[k]) - np.array(scores[default])
            mark = " <- default" * (k == default) + " <- chosen" * (k == chosen and k != default)
            mark += " (over budget)" * (k not in affordable)
            print(f"  {np.mean(scores[k]):.4f} ({d.mean():+.4f} ± {stderr(d):.4f}), {runs[k]:.0f} runs: {desc}{mark}")

    if args.truths != TRUTHS:
        return
    out_file.write_text(json.dumps(dict(sorted(tuning.items(), key=lambda kv: list(GRIDS).index(kv[0]))), indent=1) + "\n")
    print(f"\nWrote {out_file.relative_to(HERE)}")


assert REFERENCE not in GRIDS, "the reference is not tuned: the other rows are compared with it"

if __name__ == "__main__":
    main()
