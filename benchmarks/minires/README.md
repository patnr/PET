# MiniRes benchmark

Twin experiments on a small waterflood, simulated with [MiniRes](https://github.com/patnr/MiniRes),
to compare PET's update schemes, analyses and localizations on equal terms: the same cases, the
same truths, the same scores.

> [!WARNING]
> This is not (yet) a representative or authoritative benchmark. It has one reservoir, at two
> resolutions; few truths; and methods run at default or textbook settings, mostly untuned.
> Its results say how the methods fare on *this* case and *these* settings, and can change with
> either. Use it to catch regressions and to guide development, not to rank methods.

## Purpose

- **A score kept up to date.** `RESULTS.md` records how each method does, so that a change to PET's
  code can be checked against it.
- **Scores a misfit cannot give.** The truth is known, so an update is scored on the unknowns
  (the log-permeability field, and the residual oil saturation `sor`) and on a forecast that was
  not assimilated, as well as on how well it fits the data.

## Layout

| | |
|---|---|
| `cases.py` | The cases: the reservoir, its wells and schedule, the prior, the observations, and the twin experiments drawn from them. |
| `methods.py` | The methods compared, each a label and its PET settings (`[dataassim]`). |
| `scores.py` | What an ensemble is scored on. |
| `run.py` | Runs the methods on the cases, and generates `RESULTS.md`. |
| `tune.py` | Tunes the methods' settings, on twin experiments of its own, into `results/tuning.json`; `methods.py` adds a tuned row for each method whose default it beat. |
| `results/` | The scores of each case and ensemble size (`<case>-ne<ne>.json`), per method and truth, with the PET commit each method was run at; `RESULTS.md` is generated from them. |
| `RESULTS.md` | The tables. Generated; do not edit by hand. |

`tests/test_benchmark_minires.py` checks that each method still runs, on one tiny experiment.

## Running

Needs the `minires` extra (`pip install -e .[minires]`). From this directory:

```sh
python run.py --quick               # while developing: the coarse case, 5 truths, about a minute
python run.py                       # the official run: every method, case and ensemble size
python run.py --ne 20               # only this ensemble size (default: the case's own sizes)
python run.py --method "ES-MDA"     # only the methods whose label contains this, and the reference (ES)
```

A run of other than the case's own number of truths, such as a quick run, prints its tables
and writes nothing. It is
printed next to the official scores on the same truths; the experiments are seeded, so any
difference is due to the code. Its numbers are tentative: with so few truths, the differences
between methods are often within the noise.

The official run writes `results/` and `RESULTS.md`. It is redone from time to time, not with
every change, so `RESULTS.md` may lag behind the methods and scores defined here. A run of some
of the methods replaces only their rows; a row run at another PET commit than the rest of its
table is marked with that commit. The full tables are at 50 members (`NE` of `cases.py`); a case
with other sizes (`sizes`) also gets a few scores tabled against the ensemble size.

## Tuning

The methods run at their defaults, which makes a comparison of rows partly a comparison of
settings. `tune.py` tunes them by a protocol fixed in advance (see its docstring): on truths of
its own, not the scored ones; for the forecast's CRPS; on `coarse`, at 50 members; for no more
simulations than the default; and the default stands unless beaten by more than twice the
standard error of the difference. A tuned row is then shown next to its default, and how far
it carries over to the other cases and ensemble sizes is for the tables to tell.

```sh
python tune.py                      # every method with a grid (GRIDS), about 1200 experiments
python tune.py --truths 2           # a check: printed, not written
```

## Adding a method or a score

- A method: an entry in `METHODS` of `methods.py`.
- A score: a key in the output of `scores.scores`, and a column in `tables` of `run.py`. If lower
  is better, its name ends in `_rmse`, `_error` or `_crps` (or see `better`), so that a posterior
  worse than the prior is flagged.
- A case: an entry in `CASES` of `cases.py`.
