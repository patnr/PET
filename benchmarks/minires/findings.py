"""What the tables of RESULTS.md show, in a few sentences per heading: written by hand, rendered by run.py.

Keyed by case (its full tables, at NE members), or by ``<case>/sizes`` (its tables by ensemble
size). Written from the official runs of 2026-10-30: when those are redone, check that these
still hold.
"""

FINDINGS = {
    "coarse": (
        "ES-MDA and LM-EnRML fit the data to about the noise level (misfit about 2) and forecast best of the "
        "defaults (CRPS 0.044 and 0.049, against ES's 0.063); GN-EnRML at its default step lengths barely "
        "improves on ES. The localizations and EnIF fit the data less closely, but keep more spread and score "
        "best on the field (logK CRPS about 0.405). Tuning helps the localizations most: distance localization "
        "at 600 m forecasts as well as ES-MDA, and scores better on the field."
    ),
    "coarse/sizes": (
        "At 20 members, ES, LM-EnRML, GN-EnRML (but with margIS) and ES-MDA collapse (logK spread 0.27 to 0.44, "
        "against the prior's 0.79) and end up worse than the prior on the field; EnIF and the default "
        "localizations do not. Distance localization forecasts almost as well at 20 members as at 100 (CRPS "
        "0.053 and 0.050), while ES needs 100 members to match it. At 100 members, the methods differ little."
    ),
    "fine": (
        "Much as `coarse`: the resolution changes little, except that EnIF fits the data worse (misfit 30, "
        "against 13). ES-MDA (tuned: 0.038) and distance localization at 600 m (0.040) forecast best. With "
        "only 8 truths, few of the differences between the iterating methods are resolved."
    ),
    "unfavourable": (
        "Iterating pays most here: LM-EnRML, ES-MDA and GN-EnRML at γ max = 1 forecast about a third better "
        "than ES (CRPS 0.029, 0.028 and 0.028, against 0.041), while GN-EnRML at its default step lengths does "
        "no better than ES. Distance localization at 300 m now and then makes the fit far worse than the "
        "prior's (misfit 200 ± 100, from a few truths); at the tuned 600 m it does not."
    ),
    "pattern": (
        "Without localization or EnIF, every scheme ends up worse than the prior on the field, though it fits "
        "the data. Of those with one, the adaptive localizations forecast worse than ES, while distance "
        "localization at 300 m forecasts best of all (CRPS 0.033, against ES's 0.056). The tuning's 600 m, "
        "chosen on `coarse`, is too wide here, where the prior's range is 150 m: it forecasts no better, and "
        "is worse than the prior on the field."
    ),
    "early": (
        "Before the water breaks through, a single update does about as well as any: LM-EnRML and GN-EnRML "
        "forecast no better than ES (CRPS 0.076, against 0.075), and ES-MDA only slightly better (0.072), "
        "each at 2.5 times the simulations. Little is learnt about the forecast by any method (the prior's "
        "CRPS is 0.131)."
    ),
}
