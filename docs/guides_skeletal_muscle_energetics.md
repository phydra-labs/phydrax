# Skeletal-muscle energetics

`UchidaUmberger2010Plan` implements the pinned OpenSim Uchida–Umberger 2010
muscle metabolic-power formulation at reference revision
`86b30588374650fbaf012a345a836a64f6855522`, validated in
[Uchida et al. 2016](https://doi.org/10.1371/journal.pone.0150378).
It is an algebraic phenomenological observation over a physical muscle-fiber
trajectory; it is not ATP chemistry or temperature.

Inputs per muscle are excitation, activation, active fiber force in N, active
force–length multiplier, fiber length in m, and fiber velocity in m/s. Negative
velocity means shortening. Parameters provide muscle mass, slow-twitch fraction,
optimal fiber length, and maximum normalized contraction velocity.

Outputs retain combined activation/maintenance heat, shortening/lengthening heat,
mechanical work, heat after the 1 W/kg floor, muscle metabolic power in W, and the
slow-recruitment fraction. The pinned policy includes orderly recruitment, negative
mechanical work, immediate correction of negative total muscle power, and the heat
floor. Its branch, cap, correction, and floor surfaces are only piecewise
differentiable.

Basal whole-body power is excluded rather than allocated to muscles. The runtime
integrates a supplied power trace to J only through the explicit
`integrate_metabolic_energy_joule` operator.

Shorten 2007 states are not converted into ATP turnover or heat: they do not supply
ATP hydrolysis/resynthesis, ADP/PCr, oxidative/glycolytic flux, chemical free energy,
or energy balance. No thermal field is implemented because no source-backed mapping
from muscle power to local retained heat density, geometry, perfusion, material
properties, and thermal boundary conditions was available. Mechanical control effort
is never called metabolism.

Run:

```text
python examples/skeletal_muscle_energetics.py
python tools/skeletal_muscle_energetics_qualification.py
python benchmarks/skeletal_muscle_energetics.py
```
