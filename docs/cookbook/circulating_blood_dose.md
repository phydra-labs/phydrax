# Circulating-blood dose

This workflow computes a research-only absorbed-dose score for blood moving through an explicit finite compartment network. It does not model immune response, lymphocyte depletion, toxicity, DNA repair, survival, treatment outcome, or clinical risk.

```python
import numpy as np

from phydrax.applications.radiation_biophysics import circulating_blood as cb

model = cb.CirculatingBloodModel(
    compartments=(
        cb.BloodCompartment("central", 1.0e-3),
        cb.BloodCompartment("peripheral", 2.0e-3),
    ),
    flows=(
        cb.BloodFlow("central", "peripheral", 2.0e-4),
        cb.BloodFlow("peripheral", "central", 2.0e-4),
    ),
)
prepared = cb.prepare_circulating_blood_model(model)
quantity = cb.circulating_blood_dose_rate_quantity(
    "compartment-dose-rate",
    cb.ABSORBED_DOSE_RATE_REFERENCE,
)
schedule = cb.PiecewiseConstantDoseRateSchedule(
    (
        cb.DoseRateInterval(
            0.0,
            2.0,
            quantity,
            np.asarray((0.5, 0.1)),
        ),
    )
)
result = cb.integrate_circulating_blood_dose(
    prepared,
    schedule,
    np.asarray((1.0, 0.0)),
    t0_s=0.0,
    t1_s=2.0,
)
```

`BloodFlow` is a physical volume flow. Preparation derives each transition intensity as flow divided by source-compartment volume and compiles it to the existing `FiniteStateGenerator`. A nonabsorbing compartment must have an outgoing route; return paths are never inferred.

`DoseRateInterval` admits only the exact circulating-blood Gy/s profiles for absorbed dose, dose to water, or dose to medium. Relative dose, kerma, LET, activity, and generic dose-rate strings are refused. Intervals are half-open, ordered, and nonoverlapping. Uncovered schedule gaps carry zero reward.

The deterministic route propagates probability and integrates compartment occupation exactly for every constant interval using a block matrix exponential. It is invariant to splitting an unchanged interval and does not approximate occupation from output save points.

The stochastic route uses the existing exact SSA solver and scores actual dwell intervals. A `PoissonClockRealization` fixes replay identity. Event-capacity exhaustion is explicit and invalidates affected histories; it is never interpreted as zero dose. Conditional measurement uncertainty remains separate from path-distribution uncertainty.

`prepare_spatial_compartment_mixture` can derive compartment dose-rate vectors from exact-matching dose and weight images. It requires identical shape and affine, nonnegative weights, and per-voxel normalization. The current route is explicitly well mixed. It does not infer compartments from anatomical labels, resample grids, or sample a DVH.

`tools/circulating_blood_dose_qualification.py` exercises analytic transition, exact occupation, interval splitting, replay, and capacity behavior. The associated capability profiles remain unreleased pending independently governed physiology and reference evidence.
