#!/usr/bin/env python3
"""Run the methods on the cases' twin experiments, and write the tables in RESULTS.md.

    python run.py                       # every method, every case, the default number of truths
    python run.py --case coarse         # one case
    python run.py --ne 20               # one ensemble size (default: the case's own sizes)
    python run.py --method "ES-MDA"     # the methods whose label contains this, and the reference
    python run.py --quick               # a check while developing: the coarse case, a few truths

A run of other than the case's own number of truths is a check, printed and not written to
results/: its truths are the first of the official run's, so it is printed next to the official
scores on those same truths, and any difference is due to the code (the experiments are seeded).

Each (method, truth) is one twin experiment, run in its own process and its own
temporary directory, so that what PET writes there (results, logs) goes nowhere.
The scores of each are kept in ``results/<case>-ne<ne>.json``, with the PET commit each
method was run at; RESULTS.md is generated from those. A run of a subset of the methods
replaces their entries and leaves the others'.
"""

import os

# The parallelism is over experiments, not within them: no BLAS threads, and no joblib pools
# (EnIF's regressions would otherwise each start one the size of the machine)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_var, "1")

import argparse
import contextlib
import io
import json
import logging
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cases import CASES, DATATYPES, HISTORY, NE, NSTEPS, OBS_STEPS, SCALARS, twin
from input_output.config import normalize
from methods import COMMON, METHODS, REFERENCE
from pipt.update_schemes.registry import get_scheme
from scores import scores
from simulator.minires import MiniRes

HERE = Path(__file__).parent
TRUTH_SEED, ENSEMBLE_SEED = 1000, 2000  # truth i is drawn with TRUTH_SEED + i
QUICK = {"case": ["coarse"], "truths": 5, "ne": [NE]}


class CountingMiniRes(MiniRes):
    """MiniRes, counting the simulations the method asks for."""

    runs = 0

    def run_fwd_sim(self, state, member_index=0, **kwargs):
        self.runs += 1
        return super().run_fwd_sim(state, member_index, **kwargs)


def experiment(case_name: str, method: dict, i: int, ne: int) -> dict:
    """One twin experiment: the prior's and the posterior's scores, and the simulations spent."""
    case = CASES[case_name]
    with (
        tempfile.TemporaryDirectory() as tmp,
        contextlib.chdir(tmp),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        case.write_prior_mean("priormean.npz")
        tw = twin(case, TRUTH_SEED + i, "priormean.npz")

        index = pd.Index(OBS_STEPS, name="steps")
        pd.DataFrame(tw.obs, index=index, columns=DATATYPES).to_csv("data.csv")
        variance = [[str(["abs", float(s**2)]) for s in row] for row in tw.std]
        pd.DataFrame(variance, index=index, columns=DATATYPES).to_csv("var.csv")

        dataassim = {**COMMON, **method["dataassim"], "data": "data.csv", "datavar": "var.csv", "savefolder": "Results"}
        if "localization" in dataassim:  # the grid and the wells are the case's, not the method's
            loc = {"field": [case.n, case.n], **dataassim["localization"]}
            if loc["name"] == "distance_loc":
                loc["field"] = [1, case.n, case.n]
                loc["entries"] = [{"taper": loc["taper_func"], **e} for e in case.distance_entries(loc.pop("radius"))]
            dataassim["localization"] = loc
        keys_da, keys_sim, keys_en = normalize(
            dataassim,
            case.simulator(OBS_STEPS),
            case.ensemble(ne, ENSEMBLE_SEED + i, "priormean.npz"),
        )
        sim = CountingMiniRes(keys_sim)
        scheme = get_scheme(keys_da["scheme"], keys_da["analysis"])(keys_da, keys_en, sim)
        idX = scheme.ensemble.idX
        prior = np.array(scheme.ensemble.enX, copy=True)
        scheme.run_assimilation()
        posterior = np.array(scheme.ensemble.enX)

        out = {"runs": sim.runs}
        for label, enX in (("prior", prior), ("posterior", posterior)):
            states = {k: enX[a:b] for k, (a, b) in idX.items()}
            members = [{k: v[:, j] for k, v in states.items()} for j in range(enX.shape[1])]
            out[label] = scores(states, np.array([case.run(m) for m in members]), tw)
        return out


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
# The tables of each case, each {score: column}
TABLES = {
    "The data and its forecast": {
        "misfit": "misfit",
        "forecast_wwct_rmse": "RMSE forecast WWCT",
        "forecast_wwct_crps": "CRPS forecast WWCT",
        "failed": "failed members",
        "runs": "simulations",
    },
    "The log-permeability": {"logk_rmse": "RMSE", "logk_corr": "corr", "logk_spread": "spread", "logk_crps": "CRPS"},
    **{f"The scalar `{n}`": {f"{n}_error": "error", f"{n}_spread": "spread", f"{n}_crps": "CRPS"} for n in SCALARS},
}
COUNTS = ["failed", "runs"]  # shown as plain means

# Which way is better (+1 higher, -1 lower), for the scores that have one: a posterior
# worse than the prior on these is flagged. The misfit has a target (1), not a direction.
BETTER = {
    "forecast_wwct_rmse": -1,
    "forecast_wwct_crps": -1,
    "logk_rmse": -1,
    "logk_corr": +1,
    "logk_crps": -1,
    **{f"{n}_{s}": -1 for n in SCALARS for s in ("error", "crps")},
}
WORSE = "⚠"


def stderr(values: np.ndarray) -> float:
    return values.std(ddof=1) / np.sqrt(len(values))


def cell(values, ref=None, prior=None, better=0) -> str:
    """The mean over the truths; ± the standard error of its difference from ``ref``, truth by
    truth; and flagged if worse than ``prior``, by more than twice that error of the difference."""
    values = np.asarray(values, dtype=float)
    text = f"{values.mean():.3g}"
    if ref is not None and len(values) > 1:
        error = stderr(values - np.asarray(ref))  # to one significant figure, never as 1e+01
        text += f" ± {np.format_float_positional(error, precision=1, unique=False, fractional=False, trim='-')}"
    if better and prior is not None:
        gain = better * (values - np.asarray(prior))
        if gain.mean() < 0 and (len(values) == 1 or -gain.mean() > 2 * stderr(gain)):
            text += f" {WORSE}"
    return text


def provenance(runs: dict) -> tuple:
    """The commit and date most rows were run at, and the rows run at another commit."""
    commits = [r["commit"] for r in runs.values()]
    usual = max(set(commits), key=commits.count)
    date_ = max(r["date"] for r in runs.values() if r["commit"] == usual)
    return usual, date_, {label: r["commit"] for label, r in runs.items() if r["commit"] != usual}


def label_of(label: str, other: dict) -> str:
    """The row's label, marked if it is the reference, or was run at another commit."""
    return label + (" (reference)" if label == REFERENCE else "") + (f" [`{other[label]}`]" if label in other else "")


def render_case(title: str, res: dict) -> list:
    meta, runs = res["meta"], res["methods"]
    prior = next(iter(runs.values()))["prior"]  # the same for every method: the same seeds
    ref = runs[REFERENCE]["posterior"]
    usual, date_, other = provenance(runs)
    lines = [
        "",
        title,
        "",
        f"{meta['truths']} truths, {meta['ne']} members; history of {HISTORY} of {NSTEPS} steps, "
        f"observed at steps {OBS_STEPS}. PET at `{usual}`, {date_}"
        + (", except the rows marked with another commit." if other else "."),
    ]
    rows = [("prior", prior, None)] + [(label, r["posterior"], r["runs"]) for label, r in runs.items()]
    for heading, columns in TABLES.items():
        lines += ["", f"**{heading}**", ""]
        lines += ["| | " + " | ".join(columns.values()) + " |", "|---" * (len(columns) + 1) + "|"]
        for label, sc, sims in rows:
            cells = []
            for k in columns:
                if k == "runs":
                    cells.append(f"{np.mean(sims):.0f}" if sims else "")
                elif k not in sc:  # a score added since the official results were run
                    cells.append("")
                elif k in COUNTS:
                    cells.append(f"{np.mean(sc[k]):.2g}")
                else:
                    against = None if label == REFERENCE else ref.get(k)
                    flag_vs = None if label == "prior" else prior.get(k)
                    cells.append(cell(sc[k], against, flag_vs, BETTER.get(k, 0)))
            lines += [f"| {label_of(label, other)} | " + " | ".join(cells) + " |"]
    return lines


# The scores shown against the ensemble size, each {score: heading}
BY_SIZE = {"forecast_wwct_crps": "CRPS forecast WWCT", "logk_crps": "logK CRPS", "logk_spread": "logK spread"}


def render_sizes(title: str, by_ne: dict) -> list:
    """A few scores of each method, at each ensemble size: ``by_ne`` is ``{ne: results}``."""
    sizes = sorted(by_ne)
    truths = {by_ne[ne]["meta"]["truths"] for ne in sizes}
    lines = ["", title, "", f"{', '.join(map(str, sorted(truths)))} truths. The ± are against ES at the same size."]
    labels = [m["label"] for m in METHODS if any(m["label"] in by_ne[ne]["methods"] for ne in sizes)]
    for k, heading in BY_SIZE.items():
        lines += ["", f"**{heading}**", ""]
        lines += ["| | " + " | ".join(f"{ne} members" for ne in sizes) + " |", "|---" * (len(sizes) + 1) + "|"]
        for label in ["prior", *labels]:
            cells = []
            for ne in sizes:
                runs = by_ne[ne]["methods"]
                prior = next(iter(runs.values()))["prior"]
                ref = runs[REFERENCE]["posterior"]
                sc = prior if label == "prior" else runs.get(label, {}).get("posterior", {})
                if k not in sc:
                    cells.append("")
                    continue
                against = None if label == REFERENCE else ref.get(k)
                flag_vs = None if label == "prior" else prior.get(k)
                cells.append(cell(sc[k], against, flag_vs, BETTER.get(k, 0)))
            lines += [f"| {label_of(label, {})} | " + " | ".join(cells) + " |"]
    return lines


def render(results: dict) -> str:
    """``results`` is ``{case: {ne: results}}``."""
    lines = [
        "# MiniRes benchmark: results",
        "",
        "Generated by `run.py`; do not edit by hand. See `cases.py` for the cases, `methods.py` for the",
        "methods and their settings, and `scores.py` for what is scored.",
        "",
        "- Each cell is the mean over the twin experiments (truths).",
        "- The ± is the standard error of the cell's *difference* from the reference row's (ES).",
        "  Every row is run on the same truths, from the same prior ensembles,",
        "  so that variability due to these somewhat cancels.",
        f"- {WORSE}: worse than the prior, by more than twice the standard error of the difference.",
        "- A lower (data) misfit is not necessarily better; could be overfitting.",
        "- Failed simulations are left out of the data and forecast scores, and counted in `failed members`.",
    ]
    for case_name, by_ne in results.items():
        case = CASES[case_name]
        if NE in by_ne:
            lines += render_case(f"## Case `{case_name}`: {case.n} x {case.n} cells", by_ne[NE])
        if len(by_ne) > 1:
            lines += render_sizes(f"### Case `{case_name}`: by ensemble size", by_ne)
    return "\n".join(lines) + "\n"


def first_truths(res: dict, truths: int, labels) -> dict:
    """The official results ``res``, of the methods ``labels``, cut down to their first ``truths``."""

    def cut(v):
        return {k: cut(w) for k, w in v.items()} if isinstance(v, dict) else v[:truths] if isinstance(v, list) else v

    methods = {label: cut(r) for label, r in res["methods"].items() if label in labels}
    return {"meta": {**res["meta"], "truths": truths}, "methods": methods}


def commit() -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()

    # The last commit of PET's code, which is what the numbers depend on (not of the benchmark itself)
    src = str(HERE.parents[1] / "src")
    return git("log", "-1", "--format=%h", "--", src) + ("-dirty" if git("status", "--porcelain", "--", src) else "")


def results_file(case_name: str, ne: int) -> Path:
    return HERE / "results" / f"{case_name}-ne{ne}.json"


def load_all() -> dict:
    """Every official result there is: ``{case: {ne: results}}``, in the order of CASES and size."""
    found = {}
    for case_name in CASES:
        files = sorted(HERE.glob(f"results/{case_name}-ne*.json"), key=lambda f: int(f.stem.split("-ne")[1]))
        if files:
            found[case_name] = {int(f.stem.split("-ne")[1]): json.loads(f.read_text()) for f in files}
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=list(CASES), action="append", help="default: all")
    parser.add_argument("--method", action="append", help="label substring (and the reference); default: all")
    parser.add_argument("--truths", type=int, help="default: the case's own")
    parser.add_argument("--ne", type=int, action="append", help="default: the case's own sizes")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--quick", action="store_true", help=f"default to {QUICK}")
    args = parser.parse_args()
    if args.quick:
        args.case = args.case or QUICK["case"]
        args.truths = args.truths or QUICK["truths"]
        args.ne = args.ne or QUICK["ne"]

    # The reference is always run: the other rows' ± are against it
    picked = [m for m in METHODS if not args.method or any(s in m["label"] for s in args.method)]
    methods = [m for m in METHODS if m in picked or m["label"] == REFERENCE]

    # Every experiment of every (case, size) in one pool, so that many cores are kept busy
    runs = [(c, ne, args.truths or CASES[c].truths) for c in args.case or CASES for ne in args.ne or CASES[c].sizes]
    jobs = [(c, ne, m, i) for c, ne, truths in runs for m in methods for i in range(truths)]
    print(f"{len(jobs)} experiments: " + ", ".join(f"{c} ({ne} members, {t} truths)" for c, ne, t in runs), flush=True)
    with ProcessPoolExecutor(args.workers, initializer=logging.disable, initargs=(logging.WARNING,)) as pool:
        done = list(pool.map(experiment, *zip(*[(c, m, i, ne) for c, ne, m, i in jobs])))

    stamp = {"commit": commit(), "date": str(date.today())}
    wrote = False
    for case_name, ne, truths in runs:
        res_file = results_file(case_name, ne)
        old = json.loads(res_file.read_text()) if res_file.exists() else None
        compatible = old and old["meta"] == {"truths": truths, "ne": ne}
        results = {"meta": {"truths": truths, "ne": ne}, "methods": dict(old["methods"]) if compatible else {}}
        for m in methods:
            mine = [d for (c, n, mm, _), d in zip(jobs, done) if (c, n, mm) == (case_name, ne, m)]
            results["methods"][m["label"]] = {
                **stamp,
                "runs": [d["runs"] for d in mine],
                **{s: {k: [d[s][k] for d in mine] for k in mine[0][s]} for s in ("prior", "posterior")},
            }
        results["methods"] = {m["label"]: results["methods"][m["label"]] for m in METHODS if m["label"] in results["methods"]}

        # Only the case's own number of truths is official; any other is a check
        if truths != CASES[case_name].truths:
            print("\n".join(render_case(f"## Case `{case_name}`, {ne} members: this run", results)))
            if old and old["meta"]["truths"] >= truths:
                official = first_truths(old, truths, results["methods"])
                if official["methods"]:
                    print("\n".join(render_case(f"## Case `{case_name}`, {ne} members: official, on the same truths", official)))
            print()
            continue
        res_file.parent.mkdir(exist_ok=True)
        res_file.write_text(json.dumps(results, indent=1) + "\n")
        wrote = True

    if wrote:
        (HERE / "RESULTS.md").write_text(render(load_all()))
        print("Wrote RESULTS.md")


if __name__ == "__main__":
    main()
