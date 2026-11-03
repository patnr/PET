"""What the tables of RESULTS.md show, in a few sentences per heading: written by hand, rendered by run.py.

Keyed by case (its full tables, at NE members), or by ``<case>/sizes`` (its tables by ensemble
size). Written from the official runs of 2026-11-03: when those are redone, check that these
still hold. Each says what the case was to show, whether it does, and what follows for PET.
"""

FINDINGS = {
    "base": (
        "The data are worth having: the field's RMSE falls by 43 % (0.80 to 0.45, margIS tuned; 0.47 for LM-EnRML "
        "and distance localization) and the forecast's CRPS by 75 % (0.142 to 0.035, distance localization), and "
        "the field and the forecast rank the rows alike. Iterating beats members: LM-EnRML at 50 members (0.046) "
        "beats ES at 125 (0.057) for the same 250 simulations, on 15 of 20 truths. Localization beats both: distance "
        "localization at the prior's range (0.035) beats LM-EnRML by 0.011 ± 0.004 (14/20) and the best unlocalized "
        "row, margIS tuned, by 0.018 ± 0.005 (18/20), at the same field RMSE and better calibrated (spread/RMSE 0.73 "
        "against 0.46). ES-MDA trails LM-EnRML (0.056 against 0.046, 13/20 to LM) and is the most overconfident row "
        "(spread/RMSE 0.30). GN-EnRML at PET's default step lengths is no better than ES-MDA; at γ max = 1 (tuned) "
        "it forecasts like LM-EnRML but collapses more (0.30). EnIF stays calibrated (1.0) but learns the field "
        "less (RMSE 0.54) and forecasts no better than ES with 125 members. Adaptive localization at the universal "
        "or 3σ threshold is likewise no better than ES at 125 members; at 1.5σ (tuned) it matches LM-EnRML, but is "
        "far behind distance localization (0.050 against 0.035, 15/20 to distance) and learns less of the field "
        "(0.56 against 0.47). For PET: GN-EnRML's default γ max = 0.5 and the adaptive localization's default "
        "thresholds are poor on this case; the distance localization's and LM-EnRML's defaults stand."
    ),
    "base/sizes": (
        "At 20 members, ES learns nothing of the forecast (0.132, prior 0.145), and the unlocalized iterating rows "
        "collapse (spread/RMSE 0.08 to 0.36) yet still forecast at 0.076 to 0.093; distance localization at 20 "
        "members (0.044) matches LM-EnRML at 100 (0.040) and beats every unlocalized row at 50. At 100 members, the "
        "differences shrink (0.036 to 0.045 for the better rows), but ES at 100 (0.065) is still worse than any "
        "unlocalized iterating row at 50. EnIF gains least from members (0.095 to 0.068); the adaptive localization at 1.5σ "
        "catches the distance localization at 100 (0.036 both). In short: at 50 members, this case needs iterations "
        "and localization; at 100, the choice of iterating scheme matters little, and localization still helps."
    ),
    "coarse": (
        "The same picture at a fifth of the cost: the field improves by 44 %, distance localization forecasts best "
        "(0.028), ES-MDA is the most overconfident row (0.29). The differences between the iterating rows are "
        "smaller: ES-MDA is within 0.006 ± 0.003 of LM-EnRML, and GN-EnRML tuned within 0.002. EnIF does much "
        "better here (0.048, against 0.071 on `base`): its neighbourhood is counted in cells, so it reaches "
        "further on a coarse grid. Good enough for a quick check; not for ranking the iterating schemes."
    ),
    "rough": (
        "Where localization should pay, it does, on the field: with a range of 300 m the data improve the field by "
        "only 22 % at best (distance localization, 0.62; EnIF, 0.63), and ES, ES-MDA and GN-EnRML tuned end up "
        "worse than the prior on it (⚠), every unlocalized iterating row overconfident (spread/RMSE 0.26 to 0.60). Distance "
        "localization's field RMSE beats LM-EnRML's by 0.067 ± 0.017 (18/20). On the forecast the ordering is "
        "as on `base` (distance localization 0.039, LM-EnRML 0.046, ES-MDA 0.054), the margins a little "
        "narrower. The adaptive localization at 1.5σ matches LM-EnRML on the forecast but not on the field "
        "(0.74, worse than EnIF's 0.63 and the distance localization's 0.62)."
    ),
    "viscous": (
        "Where iterating should pay, it pays most: ES at 0.068 against LM-EnRML's 0.030, ES-MDA's 0.036, and "
        "distance localization's 0.023 (20/20). ES at 125 members (0.040) is beaten by LM-EnRML at 50 on 17 of 20 "
        "truths. The step control matters more here than elsewhere: GN-EnRML at its default step lengths (0.048) "
        "is the worst of the iterating rows, and its tuned γ max = 1 (0.036) is worse than LM-EnRML by "
        "0.007 ± 0.002 (18/20 to LM). ES-MDA's four steps at α = 4 lose less to LM-EnRML here (0.006 ± 0.004) "
        "than its collapse (spread/RMSE 0.25) would suggest."
    ),
    "scalars": (
        "The unknown `sor` (prior error 0.032, sd 0.040) is learnt by the unlocalized iterating rows to an error "
        "of about 0.020, with a spread of 0.008 to 0.015: overconfident by two to three times. ES learns nothing "
        "of it (0.029) but shrinks its spread to 0.014 all the same. EnIF and the adaptive localization at the "
        "universal threshold leave it at the prior: they localize a global scalar away. The distance localization "
        "leaves `sor` unlocalized, as `run.py` tells it to, and does as well as LM-EnRML on it (0.022). The extra "
        "unknown costs ES-MDA on the forecast (0.060, against 0.056 on `base`, and now 8/20 against LM-EnRML) "
        "and LM-EnRML nothing (0.047)."
    ),
}
