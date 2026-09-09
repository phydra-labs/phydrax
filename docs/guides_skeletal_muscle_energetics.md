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

Outputs retain combined activation/maintenance heat, corrected shortening/lengthening
heat, mechanical work, heat after the 1 W/kg floor, muscle metabolic power in W, and
the slow-recruitment fraction. They also expose intrinsic shortening/lengthening
heat, immediate negative-work dissipation, and the heat-floor adjustment separately.
The pinned policy includes orderly recruitment, negative mechanical work, immediate
correction of negative total muscle power, and the heat floor. Its branch, cap,
correction, and floor surfaces are only piecewise differentiable.

Basal whole-body power is excluded rather than allocated to muscles. The runtime
integrates a supplied power trace to J only through the explicit
`integrate_metabolic_energy_joule` operator.

Shorten 2007 states are not converted into ATP turnover or heat: they do not supply
ATP hydrolysis/resynthesis, ADP/PCr, oxidative/glycolytic flux, chemical free energy,
or energy balance. Mechanical control effort is never called metabolism.

## Explicit retained heat and scalar Pennes

`uchida_umberger_retained_heat` creates a separate `RetainedHeatLedger` from an
accepted, explicitly identified Uchida result. Its source is muscle mass multiplied
by `heat_rate_W_per_kg`, **not** `muscle_metabolic_power_W`. The decomposition is:

- activation/maintenance heat;
- intrinsic shortening/lengthening heat;
- immediate negative-work dissipation = corrected minus intrinsic
  shortening/lengthening heat;
- heat-floor adjustment = corrected total heat minus activation/maintenance and
  corrected shortening/lengthening heat;
- basal heat only with separately supplied basal evidence (the Uchida adapter
  supplies none).

These categories sum to corrected heat exactly up to arithmetic roundoff. Neither
the negative-work correction nor the floor is added to an already corrected heat
rate. Chemical storage/export remain separate transfers; perfusion and exterior
exchange are never retained sources. Positive external mechanical work remains
with its mechanical owner.

`skeletal_muscle.thermal.Pennes1948Plan` supplies a separate **numerical scalar
field**, not a built-in human muscle case. The caller must supply a complete
`Pennes1948EvidenceBundle`: immutable raw-asset SHA256, URI, reuse rights, units
and purpose for equations, geometry, regions, properties, perfusion, initial
conditions, boundaries, source projection, retention, and validation. The bundle
records the fixed reference coordinate frame; lengths are m and volumetric blood
perfusion is explicitly 1/s. A URI or a property catalogue alone is not a qualified
physiological case.

The equation is tissue storage minus scalar conduction = retained heat plus
blood volumetric heat capacity × volumetric perfusion × (arterial − tissue
temperature), following [Pennes 1948](https://doi.org/10.1152/jappl.1948.1.2.93).
This numerical identity uses native continuous affine-tetrahedral P1 FEM,
consistent capacity, conservative backward Euler, and native PCG over prepared
sparse actions. Regionwise conductivity, volumetric capacity, blood capacity,
perfusion, arterial temperature and all boundary values are explicit, constant
coefficients. Temperature-dependent coefficients and other time schemes are not
silently substituted.

Every exterior facet must have exactly one insulated, prescribed-outward-flux,
convection, or Dirichlet owner. Positive flux leaves tissue, matching native
`conditions.thermal.HeatFlux`; convection is h × (tissue − ambient), matching
`Convection`. Native strong FE constraints own Dirichlet elimination. Shared
Dirichlet vertices must agree, and initial data must satisfy their constraints.
`RetainedHeatProjection` uses sparse nonnegative source-to-cell fractions whose
sum is one per source. Integrated cell source in W/m³ equals accepted retained W.
Source intervals are constant over each step and must start at accepted thermal
time. The source, state and physical coefficients must use one prepared precision.

The lifecycle is Plan → `prepare()` → `initial_state()` → `propose(state, ledger)`
→ `candidate.commit(state, prepared=prepared, source_state_id=...)`. A candidate
contains prior and proposed states, retained and perfusion cell fields, all
energy terms, linear status, and immutable prepared/source identities.
Commit checks the prior complete state and current physical-parameter/boundary
snapshot, so stale, rejected, unbalanced, or failed proposals cannot partially
advance temperature, time, counters, or cumulative ledgers.

The signed balance in J is storage change + perfusion outflow + prescribed flux
outflow + convection outflow + Dirichlet reaction outflow − retained source.
Dirichlet reaction is obtained from the **unconstrained** weak residual; interior
conduction cancels globally. Geometry, sparse maps, ownership and numerical policy
are fixed; continuous physical parameters remain trainable. Smooth-branch
derivatives are those of the converged linear field, not of source-switch events.

## Evidence limits

The included unit-cube source asset and drivers are **manufactured numerical
verification only**, with deliberately nonphysiological coefficients.
The unmodified [Gagnon 2025 workbook](https://data.mendeley.com/datasets/bdkhkmy4xm/1)
is retained under CC-BY-4.0 as an unconsumed held-out candidate. It has no complete
unit/participant/spatial registration dictionary and contains raw anomalies;
no units are inferred, values repaired, or human validation claimed.
[González-Alonso 2000](https://pmc.ncbi.nlm.nih.gov/articles/PMC2269891/) distinguishes
exercise heat storage, blood removal and mechanical work, but its raw traces and
co-registered geometry have not been acquired. IT’IS V4.2 acquisition identified
reference-linked scalar property candidates; its
[reuse terms](https://itis.swiss/meta-navigation/legal-notice) do not establish
redistribution permission, so no IT’IS rows or universal tissue defaults ship.

Automatic anatomical muscle coupling remains gated by local retention and
registration evidence. Anisotropic working-human conductivity requires directional
properties and 3-D thermometry. Muscle-specific vascular corrections require
source-compatible vessel anatomy; LTNE/discrete vascular networks and temperature
feedback into cells/mechanics remain separately gated. The field owns no force
and makes no distributed-execution claim.

Run:

```text
python examples/skeletal_muscle_energetics.py
python tools/skeletal_muscle_energetics_qualification.py
python benchmarks/skeletal_muscle_energetics.py
python examples/skeletal_muscle_thermal.py
python tools/skeletal_muscle_thermal_qualification.py --smoke
python benchmarks/skeletal_muscle_thermal.py --smoke
```
