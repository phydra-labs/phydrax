# Stoichiometric flux balance analysis

This minimal network has bounded substrate uptake and a biomass drain. Only the
internal metabolite is constrained to steady-state balance.

```python
from phydrax.bioinformatics.systems import (
    Compartment,
    Reaction,
    Species,
    StoichiometricNetwork,
    flux_balance_analysis,
    flux_variability_analysis,
)

network = StoichiometricNetwork(
    [Compartment("cell")],
    [
        Species("substrate", "cell", boundary_condition=True),
        Species("intermediate", "cell"),
    ],
    [
        Reaction(
            "uptake",
            ["substrate", "intermediate"],
            [-1.0, 1.0],
            lower_bound=0.0,
            upper_bound=10.0,
            exchange=True,
        ),
        Reaction(
            "biomass",
            ["intermediate"],
            [-1.0],
            lower_bound=0.0,
            upper_bound=1000.0,
            objective_coefficient=1.0,
            exchange=True,
        ),
    ],
    objective_sense="maximize",
)

fba = flux_balance_analysis(
    network,
    detect_alternate_optima=True,
    max_auxiliary_solves=4,  # two endpoint solves per reaction
)
assert bool(fba.successful)
print(fba.fluxes, fba.objective_value)
print(fba.evidence.alternate.alternate_optimum)

fva = flux_variability_analysis(
    network,
    objective_fraction=1.0,
    max_auxiliary_solves=4,
)
assert bool(fva.valid)
print(fva.minimum_fluxes, fva.maximum_fluxes)
```

The exact-model claim covers the declared linear steady-state constraints, bounds, gene
rules, and objective; execution uses numerical optimization tolerances. Inspect native
primal/dual/KKT and mass-balance evidence. FBA does not establish kinetics, regulation,
causality, or growth in an organism. FVA preflights its complete two-solves-per-reaction
family and raises `FluxCapacityError` rather than returning a partial reaction prefix.
Add exact chemical compositions and call `audit_stoichiometry` when elemental/charge
balance is part of the intended claim.
