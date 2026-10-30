"""The cases: waterfloods, observed over their history and scored on their forecast.

The reservoir is a 1000 x 1000 x 20 m box in metric units (m, day, bar, mD, cP), with
injectors on rate and producers on bottom-hole pressure, 40 report steps of 60 days. The
producers' rates, and when they water out, are therefore outcomes of the permeability
around them. The base case has one injector at the centre and four producers towards the
corners; the others vary it, each to put a different demand on the methods.

The unknowns are the log-permeability field and some scalars of the fluid (the residual oil
saturation ``sor``, by default); the rest of the fluid is known. Each twin experiment draws
its truth from the same prior as the ensemble, simulates it to the end, and perturbs the
history part into observations; the forecast part is what the posterior is scored against,
unseen.
"""

from dataclasses import dataclass, field

import numpy as np

from misc.sampling import random_stream
from misc.structures.layout import StateLayout
from pipt.misc_tools.extract_tools import extract_prior_info
from simulator.minires import MiniRes

LX = 1000.0  # extent [m]
DZ = 20.0  # thickness [m]: MiniRes is areal, so the rates are per unit thickness
DT = 60.0  # report step [days]
NSTEPS = 40
NE = 50  # the ensemble size of the full tables

# Observation error, as (relative, absolute): the larger applies.
OBS_ERROR = {"WWCT": (0.05, 0.02), "WOPR": (0.10, 5.0), "WBHP": (0.01, 1.0)}

# The prior of log K: about log(200 mD), sd 0.8, with a smooth variogram of the case's range
# (in metres): PET's "exp" is exp(-3 (h/range)^1.9), i.e. all but Gaussian.
LOGK_MEAN, LOGK_SD = np.log(200.0), 0.8

CORNERS = ((0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85))
CENTRE = ((0.5, 0.5),)
NINE = tuple((x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9))  # producers on a 3 x 3 grid
FOUR = ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7))  # injectors between them


@dataclass(frozen=True)
class Case:
    name: str
    n: int  # cells per side
    truths: int  # the number of twin experiments run by default
    sizes: tuple = (NE,)  # the ensemble sizes run by default
    producers: tuple = CORNERS  # positions, as fractions of the extent
    injectors: tuple = CENTRE
    injection: float = 1000.0  # the total injection rate [m3/day], shared by the injectors
    history: int = 16  # the observed steps, of which every ``obs_every``-th is assimilated
    obs_every: int = 4
    vo: float = 5.0  # the oil viscosity [cP]; the water's is 1
    range: float = 300.0  # the prior's correlation range [m]
    # The unknown scalars of the fluid: {name: (mean, sd, limits)}
    scalars: dict = field(default_factory=lambda: {"sor": (0.22, 0.04, [0.05, 0.40])})

    @property
    def dx(self):
        return LX / self.n

    @property
    def producer_names(self) -> list:
        return [f"PROD{k + 1}" for k in range(len(self.producers))]

    @property
    def injector_names(self) -> list:
        return ["INJ"] if len(self.injectors) == 1 else [f"INJ{k + 1}" for k in range(len(self.injectors))]

    @property
    def datatypes(self) -> list:
        prods, injs = self.producer_names, self.injector_names
        return [f"WWCT:{w}" for w in prods] + [f"WOPR:{w}" for w in prods] + [f"WBHP:{w}" for w in injs]

    @property
    def obs_steps(self) -> list:
        return list(range(self.obs_every, self.history + 1, self.obs_every))

    @property
    def forecast_steps(self) -> list:
        return list(range(self.history + 1, NSTEPS + 1))

    def model(self) -> dict:
        """``[simulator.model]``, less what the state supplies (K, and the scalars' values)."""
        rate = self.injection / len(self.injectors) / DZ
        wells = [
            {"name": name, "xy": [fx * LX, fy * LX], "rate": rate, "rw": 0.15}
            for name, (fx, fy) in zip(self.injector_names, self.injectors)
        ]
        for name, (fx, fy) in zip(self.producer_names, self.producers):
            wells.append({"name": name, "xy": [fx * LX, fy * LX], "bhp": 250.0, "rw": 0.15})
        fluid = {"vw": 1.0, "vo": self.vo, "swc": 0.2, "sor": 0.22}
        return {
            "Lx": LX, "Ly": LX, "Nx": self.n, "Ny": self.n,
            "por": 0.2, "ct": 1e-5, "cdarcy": 0.008527,
            "fluid": {**fluid, **{k: mean for k, (mean, _, _) in self.scalars.items()}},
            "wells": wells,
        }

    def simulator(self, reportpoint) -> dict:
        """The ``[simulator]`` section, reporting at ``reportpoint``."""
        return {
            "dt": DT, "reporttype": "steps", "reportpoint": list(reportpoint),
            "datatype": self.datatypes, "parallel": 1, "p0": 300.0, "s0": 0.2,
            "model": self.model(),
        }

    def ensemble(self, ne: int, seed: int, mean_file: str) -> dict:
        """The ``[ensemble]`` section; ``mean_file`` is where :meth:`write_prior_mean` put the field's mean."""
        ens = {
            "ne": ne, "seed": seed, "disable_tqdm": True, "save_prior": False,
            "state": ["permx", *self.scalars],
            "prior_permx": {
                "vario": "exp", "mean": mean_file, "var": LOGK_SD**2,
                "range": self.range / self.dx, "aniso": 1.0, "angle": 0.0, "grid": [self.n, self.n, 1],
            },
        }
        for name, (mean, sd, limits) in self.scalars.items():
            ens[f"prior_{name}"] = {"mean": mean, "var": sd**2, "limits": limits}
        return ens

    def distance_entries(self, radius: float) -> list:
        """Distance-localization entries: a taper of ``radius`` metres on the field, at each datum's well."""
        cells = {w["name"]: [int(w["xy"][0] // self.dx), int(w["xy"][1] // self.dx)] for w in self.model()["wells"]}
        return [
            {"x": x, "y": y, "z": 0, "radius": int(round(radius / self.dx)), "z_range": ":", "aniso": 1.0,
             "rotation": 0.0, "data_type": d, "time": "*", "param": "permx"}
            for d in self.datatypes
            for x, y in [cells[d.split(":")[1]]]
        ]

    def write_prior_mean(self, path: str) -> None:
        np.savez(path, permx=np.full(self.n**2, LOGK_MEAN))

    def run(self, state: dict) -> np.ndarray:
        """Simulate one member to the end: ``(NSTEPS, len(datatypes))``, or ``nan`` if it failed."""
        sim = MiniRes(self.simulator(range(1, NSTEPS + 1)))
        records = sim.run_fwd_sim(state, 0)
        if records is False:
            return np.full((NSTEPS, len(self.datatypes)), np.nan)
        return np.array([[row[d][0] for d in self.datatypes] for row in records])


CASES = {
    c.name: c
    for c in [
        Case("coarse", 25, truths=20, sizes=(20, 50, 100)),
        Case("fine", 50, truths=8),
        # Oil 50 times as viscous as water: the water breaks through early and abruptly, so the
        # history is strongly nonlinear in the permeability. Where iterating should pay.
        Case("unfavourable", 25, truths=20, vo=50.0),
        # Nine producers and four injectors, on a field of half the correlation range: many data,
        # each informative about its own neighbourhood only. Where localization should pay.
        Case("pattern", 25, truths=20, producers=NINE, injectors=FOUR, range=150.0),
        # Every step of the history assimilated: 144 data for 50 members, closely spaced in time.
        # Where the analyses for more data than members (subspace, margIS) should pay.
        Case("dense", 25, truths=20, obs_every=1),
        # Only the first 6 steps observed, before any water breaks through: rates and pressures,
        # nearly linear in the permeability. Where a single update should do.
        Case("early", 25, truths=20, history=6, obs_every=2),
    ]
}


@dataclass
class Twin:
    """One twin experiment: a truth, its full response, and the observations of its history."""

    case: Case
    truth: dict  # the true state, {variable: values}
    response: np.ndarray  # (NSTEPS, len(datatypes)), noiseless
    obs: np.ndarray  # (len(obs_steps), len(datatypes)), perturbed
    std: np.ndarray  # the observation error, same shape


def twin(case: Case, seed: int, mean_file: str) -> Twin:
    """Draw a truth from the prior, simulate it, and perturb the history into observations."""
    keys = case.ensemble(1, seed, mean_file)
    enX, layout = StateLayout.from_prior_info(extract_prior_info(keys), 1, rng=random_stream(seed), save=False)
    truth = {k: v[:, 0] for k, v in layout.to_dict(enX).items()}
    response = case.run(truth)

    exact = response[np.array(case.obs_steps) - 1]
    rel, ab = np.array([OBS_ERROR[d.split(":")[0]] for d in case.datatypes]).T
    std = np.maximum(rel * np.abs(exact), ab)
    obs = exact + std * np.random.default_rng(seed).standard_normal(exact.shape)
    # A water cut outside [0, 1], or a negative rate, would be unphysical
    quantity = [d.split(":")[0] for d in case.datatypes]
    lower = np.where(np.isin(quantity, ["WWCT", "WOPR"]), 0.0, -np.inf)
    upper = np.where(np.isin(quantity, ["WWCT"]), 1.0, np.inf)
    return Twin(case, truth, response, obs.clip(lower, upper), std)
