# Moist thermodynamics and the evolving column

`phydrax.applications.atmosphere.MoistThermodynamicPlan` represents dry air,
water vapor, liquid, and ice with one caloric reference convention. Vapor and
dry air are ideal gases; condensed phases have negligible volume but retain
all their mass, heat capacity, and energy. Every water fraction is kg water per
kg **total moist mass**, not per kg dry air or per kg gas. Density includes
condensate loading.

## Caloric and saturation model

At reference temperature T₀ (default 273.16 K), liquid internal energy is zero,
ice energy is −Lf, dry internal energy is zero, and vapor internal energy is
Lv − Rv T₀. The temperature dependence is constant cv for gases and constant c
for condensed phases. Gas enthalpies add R T to their internal energies;
condensate enthalpy equals internal energy in the zero-volume approximation.
Therefore vapor-minus-liquid enthalpy at T₀ is exactly Lv, and
liquid-minus-ice enthalpy is exactly Lf. These reference choices matter for
surface and precipitation budgets: all reservoirs must use the same references.

The partial vapor pressure is ρ qv Rv T, and total pressure is
ρ [(1 − qt) Rd + qv Rv] T. Liquid/ice contribute mass, not gas pressure. This is
not a gas-only mixture with condensate tracer corrections.

For either condensed phase c, the saturation relation is

    ln(es(T)/es(T₀)) = [(L₀ − Δcp T₀)(1/T₀ − 1/T)
                        + Δcp ln(T/T₀)] / Rv

where Δcp = cp,v − cc and L₀ is Lv over liquid or Lv + Lf over ice. Thus
its temperature derivative is exactly L(T)/(Rv T²), using the enthalpy
difference of the declared caloric model. Changing a latent heat or caloric
constant changes both saturation and adjustment consistently.

Stable ice is selected below T₀ and stable liquid above T₀. At precisely T₀,
`equilibrium(ρ, T, qt)` selects liquid when temperature alone cannot determine
the freezing fraction. Energy/enthalpy adjustment instead resolves coexistence
algebraically: vapor stays saturated and added latent energy melts ice at fixed
T₀. There is **no empirical mixed-phase temperature band**, metastable
supercooling, nucleation model, salinity, pressure-dependent melting, finite
condensate volume, or critical-point equation of state. The default admitted
temperature range is 150–400 K. Constructor checks require positive latent
internal energies and positive fusion energy over the declared range.

## Thermodynamic API and certificates

- `energy(density, temperature, vapor, liquid, ice)` returns J/kg internal energy.
- `enthalpy(temperature, vapor, liquid, ice)` returns J/kg enthalpy.
- `pressure(density, temperature, vapor, liquid, ice)` returns Pa.
- `phase_energies(T)` and `phase_enthalpies(T)` return dry/vapor/liquid/ice tuples.
- `gas_constant(qv, ql, qi)` includes total-mass condensate loading.
- `heat_capacity(qv, ql, qi, at_constant_pressure=False)` returns the frozen-
  composition capacity, not the derivative of a phase-equilibrating parcel.
- `saturation_pressure(T, phase="liquid" or "ice")` returns Pa.
- `equilibrium(density, temperature, total_water)` partitions at given ρ/T.
- `adjust(density, total_water, specific_internal_energy)` conserves e at fixed ρ.
- `adjust_isobaric(pressure, total_water, specific_enthalpy)` conserves h at fixed p.

The returned `MoistAdjustmentResult` contains `temperature`, `vapor`, `liquid`,
`ice`, `density`, `pressure`, `successful`, `residual`, and `derivative_valid`.
Inputs broadcast; success and derivative-validity masks are **elementwise**.
The adjustment residual is signed recovered-minus-requested energy (J/kg).
The fixed-temperature equilibrium partition is algebraic, with zero root
residual; its success mask still checks the admitted physical domain.

Adjustment uses the native `SmallRootKernel` with a fixed iteration envelope
(default 24 masked Newton steps), native linear solves, and a caloric branch
selected by freezing-limit energies. A root-domain bracket and independently
recomputed physical residual certify the answer. The residual certificate
includes a floating-point scale floor; requesting a sub-roundoff tolerance
does not make float32 into float64. Out-of-domain and failed-convergence values
are returned for diagnosis with `successful=False`, not silently accepted or
clipped into a successful thermodynamic state.

JAX implicit differentiation uses the diagonal caloric Jacobian through
`lax.custom_root`, not differentiation of iteration counts or branch decisions.
`derivative_valid` admits successful interiors of ice-saturated,
liquid-saturated, and unsaturated branches. It excludes saturation onset,
freezing/coexistence, and temperature-domain boundaries. Callers **must check
this mask** before interpreting a derivative; derivatives of a failed solve or
of a switching surface are not qualified. The focused tests and qualification
script compare regular-branch derivatives with centered perturbations. They
do not claim derivatives of nucleation, precipitation onset, or branch changes.

## Conservative process helpers

`conservative_vertical_mixing(specific_quantity, layer_mass, exchange_rate)`
returns the extensive layer-inventory rate. Vertical is the last axis, top to
bottom. The nonnegative exchange rate (s⁻¹) is defined on internal interfaces,
with neighboring harmonic mass as the exchange scale. Internal exchanges are
equal and opposite and boundary fluxes are zero. Explicit-step positivity is a
caller responsibility, not a clipped unreported source.

`conservative_radiation(temperature, layer_mass, heat_capacity,
equilibrium_temperature, relaxation_time)` returns layer energy rates and their
opposite summed environmental energy rate. This is Newtonian grey cooling, not
spectrally resolved radiative transfer. Retaining the opposite environmental
inventory makes its energy exchange explicitly conservative.

`precipitate(thermodynamics, density, total_water, specific_internal_energy,
layer_volume, fraction=...)` removes that fraction of each condensed phase and
its **paired phase enthalpy**. It reduces total density and renormalizes all
mass fractions and specific energy, then readjusts the remaining parcel.
`MoistPrecipitationResult` returns `density`, `total_water`,
`specific_internal_energy`, `thermodynamics`, `precipitated_water`,
`precipitated_energy`, and `successful`. Water/energy transfers are extensive
on the volume's area convention. Failed cells return unchanged input
inventories and zero transfers. The caller must add accepted transfers to its
actual reservoir; diagnostics alone are not a receiving state.

Ice precipitation may carry negative reference enthalpy. Its signed transfer
must not be clipped to zero. Removal conserves both total water and energy
when the receiving reservoir is counted. The helper removes condensate into
the reservoir directly: it does not resolve droplet fall speed, rain transit,
evaporation along a fall path, or gravitational/kinetic energy conversion.
For a hydrostatic application, use the isobaric thermodynamic closure and its
own layer-mass/pressure work balance instead of interpreting isochoric removal
as an isobaric enthalpy update.

## Time-evolving column and restart

`MoistColumnPlan` is a fixed-geometric-volume thermodynamic column, **not** a
hydrostatic, convective, or compressible dynamics solver. Its conserved state
contains dry/vapor/liquid/ice layer masses, layer internal energies, layer
volumes, water and energy in a surface/precipitation reservoir, an environmental
radiative-energy inventory, and accumulated prescribed external heat.

Initialize with density, temperature, total water fraction, and layer volume
(m³ per m² horizontal area). Provide reservoir water and reference-consistent
energy when evaporation is enabled. The bottom layer is the last index.
`step(state, dt, heating_rate=..., surface_vapor_flux=...,
surface_energy_flux=...)` performs actual time evolution:

1. Refresh prescribed forcings only on the configured accepted-step cadence.
2. Exchange water/energy vertically, apply paired radiation, and add prescribed
   heating (W/m² per layer).
3. Transfer surface vapor (kg/(m² s), positive into air), carrying vapor
   enthalpy evaluated at the declared surface temperature, plus separately
   prescribed surface heat (W/m²). Subtract both from the real reservoir.
4. Equilibrate and remove an exponential condensate fallout fraction into the
   receiving water/enthalpy reservoir.
5. Re-equilibrate and certify water/energy budgets before atomically committing.

Vertical water exchange transports the donor parcel's water-specific internal
energy (the vapor/liquid/ice energy weighted within its water inventory). A
separate conservative temperature-gradient conduction flux transfers sensible
heat. Dry masses do not move. In particular, equal-temperature unsaturated
layers exchange vapor energy ev times the transferred water mass, not
(ev − ed) times that mass; humidity equilibration must not create a spurious
temperature split.

The surface vapor flux can be signed, but no negative final water inventories
are accepted. Invalid forcing/timestep, exhausted reservoir, or thermodynamic
failure rejects the **whole column step**. Rejected attempts leave all physical
inventories, time, accepted-step count, held forcing, and cadence phase unchanged.
`MoistColumnStepResult` retains attempted budget residuals and returns zero
committed precipitation on rejection. `advance` performs a native JAX scan and
returns final state plus per-step acceptance; callers must inspect acceptance.

`MoistColumnState` includes all numeric inventories, held forcing, and cadence
state. Persist it with `plan.save_checkpoint(path, state)` and restore it with
`plan.load_checkpoint(path)`. These methods use the native atomic,
checksum-validated, pickle-free array archive and store the **actual static
plan identity** in its manifest. Loading checks that identity against the plan
before constructing a state, with no caller-supplied state template. Different
thermodynamic constants or process/cadence parameters are rejected. Array
shape, finite inventories, cadence, and exact JAX dtype availability are
admitted on load. Equinox leaf-only serialization is not the checkpoint
contract: it omits static fields and cannot establish the originating plan.
Its `total_water` includes air and surface/precipitation reservoir;
`total_energy` includes air, surface, and radiative environment. Subtract the
change in `external_energy` when checking a heated rollout's closed balance.

## Runnable example and qualification

From the worktree, with its Python environment and local package selected:

```sh
PYTHONPATH=. python examples/moist_column.py --steps 120
PYTHONPATH=. python tools/moist_column_qualification.py --layers 8 --steps 120
```

The example prints evolved temperatures, phase inventories, reservoir changes,
and water/energy residuals; `--restart PATH` writes a plan-bound native checkpoint.
The qualification script measures compilation-plus-first-run and warmed
rollout times, verifies actual water/energy budgets, compares regular-branch
thermal derivatives, and checks cadence-preserving serialized restart equality.
It raises on rejection or failed assertions; printed metrics are measured on
the executing backend, not prescribed performance claims.

Focused regressions are in
`tests/unit/applications/test_moist_atmosphere.py`: dry/load limits, saturation
caloric consistency, phase conservation, isobaric closure, latent freezing
coexistence, branch derivatives, failed certificates, paired precipitation
enthalpy, conservative process rates, restart cadence, and rejection rollback.
