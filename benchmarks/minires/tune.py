#!/usr/bin/env python3
"""Tune the methods' settings, on twin experiments of their own, and write results/tuning.json.

    python tune.py                      # every method with a grid
    python tune.py --method "LM"        # the methods whose label contains this

The protocol, fixed in advance so that the tuning cannot be tuned:

- **The truths are not the scored ones**: seeds offset by ``OFFSET``, so a tuned row is not
  fitted to the truths it is scored on.
- **One objective**: the posterior's CRPS of the forecast water cut, a proper score of what the
  forecast is for. Not the misfit, which rewards overfitting.
- **One case and size**: ``base``, at ``NE`` members. That the tuning carries over to the other
  cases and sizes is then something the benchmark tests.
- **The same budget**: a setting may be chosen only if it spends no more simulations, on
  average, than the default. (The grids leave the iterations or steps at most the default's,
  but an iterating scheme also spends simulations on the steps it rejects.)
- **Selected on some truths, confirmed on others**: the best point of the grid is picked on
  ``SELECT`` truths; being the best of many, its gain there is biased upwards (the winner's
  curse), so it and the default are then run on ``CONFIRM`` fresh truths, and it replaces the
  default only if, there, its mean beats the default's by more than twice the standard error
  of their difference (truth by truth).

The printout lists every point of the grid: a best point at the grid's edge says the grid should
be widened, which is why the grids below extend well past the choices of earlier tunings.
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
CASE, OFFSET = "base", 10_000  # tuning truth i is drawn with run.TRUTH_SEED + OFFSET + i
SELECT, CONFIRM = 20, 20  # the truths the best point is selected on, and the fresh ones it is confirmed on
OBJECTIVE = "forecast_wwct_crps"  # lower is better


def mda(steps: int, ratio: float) -> dict:
    """An ES-MDA schedule of ``steps`` inflations, each ``ratio`` times the next, whose inverses sum to 1."""
    alphas = np.array([ratio ** (steps - 1 - k) for k in range(steps)], dtype=float)
    alphas *= np.sum(1 / alphas)  # not rounded: PET checks that the inverses sum to 1
    return {"mda": {"tot_assim_steps": steps, "inflation_param": [float(a) for a in alphas]}}


def iteration(**kw) -> dict:
    return {"iteration": kw}


def default_first(default, values) -> list:
    return [default] + [v for v in values if v != default]


def schedule(k: int, r: float) -> tuple:
    alphas = mda(k, r)["mda"]["inflation_param"]
    shown = f"{alphas[0]:g}" if r == 1 else ", ".join(f"{a:.4g}" for a in alphas)
    return (f"{k} steps, α ratio {r}", mda(k, r), ("4 steps, α = 4", f"{k} steps, α = {shown}"))


# The grids: {label of a method: {option: [(description, settings to merge into its dataassim,
# (the part of the method's label they change, and what to)]}}. The first of each option's values
# is the default: PET's, or the method's in methods.py.
SCHEDULES = [schedule(k, r) for k, r in default_first((4, 1), itertools.product((2, 3, 4), (1, 2, 4)))]
STEP_LENGTHS = [
    (
        f"γ = {g}, γ max = {m}",
        iteration(gamma=g, gamma_max=m),
        ("(4 iterations)", f"(4 iterations, γ = {g}, γ max = {m})"),
    )
    for g, m in default_first((0.2, 0.5), itertools.product((0.1, 0.2, 0.4), (0.5, 1.0)))
]
GRIDS = {
    "LM-EnRML (4 iterations)": {
        "damping": [
            (
                f"λ = {lam}, λ factor = {f}",
                iteration(**{"lambda": lam, "lambda_factor": f}),
                ("(4 iterations)", f"(4 iterations, λ = {lam}, λ factor = {f})"),
            )
            for lam, f in default_first((100, 5), itertools.product((1, 10, 100, 1000), (2, 5, 10)))
        ],
    },
    "GN-EnRML (4 iterations)": {"step": STEP_LENGTHS},
    "GN-EnRML (4 iterations), subspace2 analysis": {"step": STEP_LENGTHS},
    "GN-EnRML (4 iterations), margis analysis": {"step": STEP_LENGTHS},
    "ES-MDA (4 steps, α = 4)": {"schedule": SCHEDULES},
    "ES-MDA (4 steps, α = 4), enif analysis": {
        "schedule": [sch for sch in SCHEDULES if sch[0].startswith("4 steps")],
        "neighbourhood": [
            (
                f"neighbourhood {k}",
                {"enif": {"neighbourhood_expansion": k}},
                ("enif analysis", f"enif analysis (neighbourhood {k})"),
            )
            for k in (2, 1, 3)
        ],
    },
    "ES-MDA (4 steps, α = 4), adaptive localization (3σ threshold)": {
        "cutoff": [
            (f"{c}σ threshold", {"localization": {"cutoff": float(c)}}, ("(3σ threshold)", f"({c}σ threshold)"))
            for c in (3, 1, 1.5, 2, 2.5, 4)
        ],
    },
    "ES-MDA (4 steps, α = 4), distance localization (Gaspari-Cohn, radius 1 range)": {
        "radius": [
            (f"radius {r:g} range", {"localization": {"radius": float(r)}}, ("radius 1 range", f"radius {r:g} range"))
            for r in (1, 0.5, 0.75, 1.5, 2)
        ],
    },
}


def merge(base: dict, extra: dict) -> dict:
    """``base``, with ``extra`` merged into it, dicts recursively."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def points(method: dict, grid: dict) -> list:
    """Each point of the grid: its description, the method's settings at it, and the label of its
    tuned row, should it be chosen. The first is the default."""
    out = []
    for combo in itertools.product(*grid.values()):
        settings, label = method["dataassim"], method["label"]
        for _, extra, (old, new) in combo:
            settings = merge(settings, extra)
            assert old in label, (old, label)
            label = label.replace(old, new, 1)
        out.append(("; ".join(d for d, _, _ in combo), settings, f"{label}, tuned"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", action="append", help="label substring; default: all with a grid")
    parser.add_argument("--truths", type=int, default=SELECT, help="to select on; other than the default: a check, not written")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    by_label = {m["label"]: m for m in METHODS}
    labels = [lab for lab in GRIDS if not args.method or any(s in lab for s in args.method)]
    grid_points = {lab: points(by_label[lab], GRIDS[lab]) for lab in labels}
    method_at = lambda lab, k: {"label": grid_points[lab][k][2], "dataassim": grid_points[lab][k][1]}  # noqa: E731
    pool = ProcessPoolExecutor(args.workers, initializer=logging.disable, initargs=(logging.WARNING,))

    # Selection: every point of every grid, on the SELECT truths
    jobs = [(lab, k, i) for lab in labels for k in range(len(grid_points[lab])) for i in range(args.truths)]
    print(f"{len(jobs)} experiments: {sum(map(len, grid_points.values()))} settings x {args.truths} truths", flush=True)
    with pool:
        done = list(pool.map(experiment, *zip(*[(CASE, method_at(lab, k), OFFSET + i, NE) for lab, k, i in jobs])))
        default = 0  # the grid's first point
        scores, runs, best = {}, {}, {}
        for lab in labels:
            scores[lab] = [[d["posterior"][OBJECTIVE] for (lb, kk, _), d in zip(jobs, done) if (lb, kk) == (lab, k)]
                           for k in range(len(grid_points[lab]))]
            runs[lab] = [float(np.mean([d["runs"] for (lb, kk, _), d in zip(jobs, done) if (lb, kk) == (lab, k)]))
                         for k in range(len(grid_points[lab]))]
            affordable = [k for k in range(len(scores[lab])) if runs[lab][k] <= runs[lab][default]]
            best[lab] = min(affordable, key=lambda k: np.mean(scores[lab][k]))

        # Confirmation: the best point and the default, on CONFIRM fresh truths
        confirm = [(lab, k, i) for lab in labels if best[lab] != default for k in (default, best[lab]) for i in range(CONFIRM)]
        print(f"{len(confirm)} experiments to confirm {sum(best[lab] != default for lab in labels)} choices", flush=True)
        done2 = list(pool.map(experiment, *zip(*[(CASE, method_at(lab, k), OFFSET + SELECT + i, NE) for lab, k, i in confirm]))) if confirm else []

    out_file = HERE / "results" / "tuning.json"
    out_file.parent.mkdir(exist_ok=True)
    tuning = json.loads(out_file.read_text()) if out_file.exists() else {}
    for lab in labels:
        pts = grid_points[lab]
        confirmed = {k: [d["posterior"][OBJECTIVE] for (lb, kk, _), d in zip(confirm, done2) if (lb, kk) == (lab, k)]
                     for k in (default, best[lab])}
        gain = np.array(confirmed[default]) - np.array(confirmed[best[lab]]) if best[lab] != default else np.zeros(1)
        chosen = best[lab] if best[lab] != default and gain.mean() > 2 * stderr(gain) else default
        tuning[lab] = {
            "commit": commit(), "case": CASE, "ne": NE, "truths": {"select": SELECT, "confirm": CONFIRM}, "objective": OBJECTIVE,
            "default": default, "best": best[lab], "chosen": chosen,
            "confirmed": {str(k): v for k, v in confirmed.items()},
            "points": [{"description": desc, "label": tuned_label, "dataassim": s, "runs": r, OBJECTIVE: sc}
                       for (desc, s, tuned_label), r, sc in zip(pts, runs[lab], scores[lab])],
        }
        print(f"\n{lab}" + ("" if chosen != default else "  (the default stands)"))
        affordable = [k for k in range(len(pts)) if runs[lab][k] <= runs[lab][default]]
        for k, (desc, _, _) in enumerate(pts):
            gain_k = np.array(scores[lab][default]) - np.array(scores[lab][k])  # positive: better than the default
            mark = " <- default" * (k == default) + " <- best" * (k == best[lab] and k != default)
            mark += " (over budget)" * (k not in affordable)
            print(f"  {np.mean(scores[lab][k]):.4f} (gain {gain_k.mean():+.4f} ± {stderr(gain_k):.4f}), {runs[lab][k]:.0f} runs: {desc}{mark}")
        if best[lab] != default:
            print(f"  confirmed on {CONFIRM} fresh truths: gain {gain.mean():+.4f} ± {stderr(gain):.4f} -> "
                  + ("chosen" if chosen != default else "not confirmed; the default stands"))

    if args.truths != SELECT:
        return
    out_file.write_text(json.dumps(dict(sorted(tuning.items(), key=lambda kv: list(GRIDS).index(kv[0]))), indent=1) + "\n")
    print(f"\nWrote {out_file.relative_to(HERE)}")


assert REFERENCE not in GRIDS, "the reference is not tuned: the other rows are compared with it"

if __name__ == "__main__":
    main()
