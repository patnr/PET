# MiniRes benchmark

Twin experiments on a small waterflood, simulated with [MiniRes](https://github.com/patnr/MiniRes),
to compare PET's update schemes, analyses and localizations on equal terms: the same cases, the
same truths, the same scores.

> [!WARNING]
> This is a small benchmark, not an authoritative one. It has one reservoir, varied a few ways;
> 20 truths per case; and methods at their defaults, or at settings tuned on the base case alone.
> Its results say how the methods fare on *these* cases and *these* settings, and can change with
> either. Use it to catch regressions, to guide development, and to see what each method is good
> and bad at; not to rank methods in general.

## Purpose

- **A score kept up to date.** `RESULTS.md` records how each method does, so that a change to PET's
  code can be checked against it.
- **Scores a misfit cannot give.** The truth is known, so an update is scored on the unknowns
  (the log-permeability field, and a scalar of the fluid in one case), on how well calibrated it
  is, and on a forecast that was not assimilated, as well as on how well it fits the data.

## The cases

A 1000 m square reservoir with nine producers on a 3 x 3 grid and four injectors between them,
observed over the first 16 of 40 report steps (every second one), by which time about half of the
producers have watered out; the forecast is the rest. The field was chosen by screening (see
`cases.py`) so that the data improve it substantially (its RMSE falls by about 40 %) and so that
the forecast tells the methods apart, and apart from how well they fit the history: at 25 x 25
cells, with fewer wells, or with a shorter correlation range, it does neither nearly as well.

| case | what it varies | what it is to show |
|---|---|---|
| `base` | — (50 x 50 cells, range 450 m, oil 5 times as viscous as water) | the main tables, and the tables by ensemble size (20, 50, 100) |
| `coarse` | 25 x 25 cells | the same at a fifth of the cost, for quick checks; smeared fronts make it less nonlinear |
| `rough` | range 300 m | more degrees of freedom than the data fix: where spurious correlations cost, and localization pays |
| `viscous` | oil 20 times as viscous | abrupt breakthroughs, a strongly nonlinear history: where iterating pays |
| `scalars` | `sor` unknown too | a global scalar next to the field, which a localization must leave alone |

## Layout

| | |
|---|---|
| `cases.py` | The cases: the reservoir, its wells and schedule, the prior, the observations, and the twin experiments drawn from them. |
| `methods.py` | The methods compared, each a label and its PET settings (`[dataassim]`). |
| `scores.py` | What an ensemble is scored on. |
| `run.py` | Runs the methods on the cases, and generates `RESULTS.md`. |
| `tune.py` | Tunes the methods' settings, on twin experiments of its own, into `results/tuning.json`; `methods.py` adds a tuned row for each method whose default it beat. |
| `results/` | The scores of each case and ensemble size (`<case>-ne<ne>.json`), per method and truth, with the PET commit each method was run at; `RESULTS.md` is generated from them. |
| `findings.py` | What the tables show, a few sentences per heading of `RESULTS.md`, and which of each case's tables it shows. Written by hand; check them when the results are redone. |
| `RESULTS.md` | The tables. Generated; do not edit by hand. |

`tests/test_benchmark_minires.py` checks that each method still runs, on one tiny experiment.

## Running

Needs the `minires` extra (`pip install -e .[minires]`). From this directory:

```sh
python run.py --quick --workers 4     # while developing: the coarse case, 3 truths, a few minutes
python run.py                         # the official run: every method, case and ensemble size
python run.py --case base --ne 50     # one case, one ensemble size (default: the case's own sizes)
python run.py --method "ES-MDA"       # only the methods whose label contains this, and the reference (ES)
python run.py --show                  # print every table of the official results, and run nothing
```

A run of other than the case's own number of truths, such as a quick run, prints its tables
and writes nothing. It is printed next to the official scores on the same truths; the
experiments are seeded, so any difference is due to the code. Its numbers are tentative: with
so few truths, the differences between methods are often within the noise.

The official run writes `results/` and `RESULTS.md`. It takes a few thousand experiments of a
few minutes each, so it is a job for a workstation with a hundred cores or more, and is redone
from time to time, not with every change: `RESULTS.md` may lag behind the methods and scores
defined here. A run of some of the methods replaces only their rows; a row run at another PET
commit than the rest of its table is marked with that commit. The full tables are at 50 members
(`NE` of `cases.py`); the base case also gets a few scores tabled against the ensemble size.

## Reading the tables

- The reference row is ES, a single update: the ± of every other row is the standard error of
  its difference from ES, truth by truth, and `wins vs ES` counts the truths on which it
  forecasts better. Two rows other than ES are told apart only roughly, by their ±.
- The forecast is scored by the CRPS of the water cut over the unseen steps, and by the CRPS of
  the day each producer waters out. The field by its RMSE and CRPS, and by its spread/RMSE: about 1
  for a calibrated ensemble, well below it for an overconfident one.
- The misfit is shown as a median, with a count of the truths on which the posterior fits the
  history worse than the prior did: a blow-up.
- The iterating rows spend 250 simulations where ES spends 100; `ES (one update, 125 members)`
  spends the same 250 on members instead, for an equal-budget comparison.
- `base` gets every table; a case that varies it gets only the table that carries its story
  (`TABLES` of `findings.py`), the others being its findings' "the picture is that of `base`" in
  numbers. `run.py --show` prints them all.

## Tuning

The methods run at their defaults, which makes a comparison of rows partly a comparison of
settings. `tune.py` tunes them by a protocol fixed in advance (see its docstring): on truths of
its own, not the scored ones; for the forecast's CRPS; on `base`, at 50 members; for no more
simulations than the default; and the best point of a grid replaces the default only if it is
confirmed, on fresh truths, to beat it by more than twice the standard error of the difference.
A tuned row is then shown next to its default, and how far it carries over to the other cases
and ensemble sizes is for the tables to tell.

```sh
python tune.py                      # every method with a grid (GRIDS), about 1500 experiments
python tune.py --truths 2           # a check: printed, not written
```

## Adding a method, a score or a case

- A method: an entry in `METHODS` of `methods.py`.
- A score: a key in the output of `scores.scores`, and a column in `tables` of `run.py`. If lower
  is better, its name ends in `_rmse`, `_error` or `_crps` (or see `better`), so that a posterior
  worse than the prior is flagged.
- A case: an entry in `CASES` of `cases.py`.
