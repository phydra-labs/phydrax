# Interactive liquid surface exchange

`WetSlabPlan` and `BulkSurfaceExchangePlan` provide an energy-bearing liquid
surface and boundary-disequilibrium fluxes for a moist atmosphere. They are
separate from the existing equilibrium-column reference and geophysics coupling
APIs. The surface is not a prescribed temperature reservoir: its one temperature
is recovered from its current water and energy inventories.

This is a **forced thermodynamic surface-layer model**, not an ocean dynamics,
sea-ice, momentum-transfer, or fully resolved turbulent boundary-layer model.
Wind speed and measurement height are inputs. No momentum stress, mechanical
work, gustiness, or natural-convection heat source is silently introduced.

## State, caloric reference, and liquid domain

A `WetSlabState(water_mass, energy)` contains only water in kg/m² and total slab
energy in J/m². For dry areal heat capacity `C_dry` in J/m²/K, liquid mass `M`,
liquid specific heat `c_l`, and the native moist reference temperature `T_ref`,

```text
E_slab = C_dry (T_s - T_ref) + M h_liquid(T_s)
       = (C_dry + M c_l) (T_s - T_ref)
T_s    = T_ref + E_slab / (C_dry + M c_l).
```

`MoistThermodynamicPlan.phase_energies` supplies the liquid calorics. Its liquid
has negligible volume, so liquid internal energy equals liquid enthalpy. The
slab's dry energy is zero at the same temperature. As water leaves or arrives,
heat capacity changes; retaining a constant water heat capacity or separately
updating an SST array would violate this contract.

```python
import jax.numpy as jnp
from phydrax.applications.atmosphere import (
    BulkSurfaceExchangePlan,
    MoistThermodynamicPlan,
    WetSlabPlan,
)

thermo = MoistThermodynamicPlan()
slab_plan = WetSlabPlan(thermo, dry_heat_capacity=5.0e5)
slab = slab_plan.initialize(temperature=300.0, water_mass=20.0)
surface_temperature = slab_plan.temperature(slab, thermo)
assert bool(slab_plan.admissible(slab, thermo))
```

`dry_heat_capacity` must be a positive finite scalar. Water must be finite and
nonnegative. The default temperature domain is the native freezing reference
through the thermodynamic maximum. Custom bounds must remain inside that liquid
domain. A freezing proposal is **rejected**, not clamped and not converted to
unmodeled ice. When snow energy is delivered by a coupled atmosphere, its actual
phase enthalpy must enter the same slab energy budget; accepting its conversion
to liquid requires the resulting slab to remain admissible.

A slab has no pressure state. The exchange law separately requires liquid
saturation pressure below local air pressure, excluding boiling at the supplied
pressure. This is not a salt-water or pressure-dependent melting model.

`temperature(state, thermodynamics)` checks the bound thermodynamic plan
identity; mismatched caloric references raise a structural error. It does not
certify a proposed state. Use `admissible` to obtain elementwise status and reject
a coupled proposal atomically. `initialize` refuses invalid initial states.

## Signed bulk fluxes and donor energy

All kernel inputs are SI arrays and broadcast over independent boundary cells:

| Input | Meaning | Unit |
|---|---|---|
| `air_temperature` | Boundary gas temperature | K |
| `air_density` | Dry-air-plus-vapor density, excluding condensate | kg/m³ |
| `air_vapor` | Vapor fraction of dry-air-plus-vapor mass | kg/kg |
| `air_pressure` | Total gas pressure used for the shallow boundary layer | Pa |
| `surface_temperature` | Temperature derived from the slab | K |
| `wind_speed` | Nonnegative externally supplied ventilation speed | m/s |
| `measurement_height` | Positive shallow surface-layer height | m |

For an atmospheric cell containing condensate, pass
`gas_fraction = 1 - q_liquid - q_ice`,
`air_density = total_density * gas_fraction`, and
`air_vapor = q_vapor / gas_fraction`. The operator does not advect cloud
condensate or treat its heat capacity as a gas sensible-heat flux.

With `epsilon = R_d / R_v`, native liquid saturation pressure `e_s(T_s)`,
and total gas pressure `p`, the saturated boundary specific humidity is

```text
q_s = epsilon e_s / [p - (1 - epsilon) e_s].
```

For heat/moisture transfer coefficients `C_H`, `C_E`, wind `U`, and stability
multiplier `f`,

```text
H = rho cp_air C_H U f (T_s - T_air)       [W/m²]
F = rho C_E U f (q_s - q_air)             [kg/m²/s]
Q_water = F h_vapor(T_donor)              [W/m²]
T_donor = T_s for q_s >= q_air; T_air for q_s < q_air.
```

Positive rates are **surface to atmosphere**. Evaporation has positive `F`; dew
has negative `F`. Both sensible-heat signs follow the actual temperature
difference. Saturation and vapor donor enthalpies are obtained from the same
`MoistThermodynamicPlan`, including nondefault latent heats and heat capacities.
`water_enthalpy` is an **energy flux**, not a specific enthalpy.

The total thermal energy transfer is exactly `H + Q_water`. Do not add a further
`latent_heat * F`: phase-change energy is already in the vapor enthalpy. At zero
sensible transfer the exact finite slab response is

```text
T_new - T_old = -DeltaM [h_vapor(T_donor) - h_liquid(T_old)]
                    / [C_dry + (M_old - DeltaM) c_l].
```

Thus evaporation cools the slab and dew warms it on the liquid domain. Dew uses
the **air donor temperature**, not the receiving slab temperature.

```python
exchange = BulkSurfaceExchangePlan(
    heat_transfer_coefficient=1.2e-3,
    moisture_transfer_coefficient=1.2e-3,
    stability="bulk-richardson",
)
rates = exchange.evaluate(
    thermo,
    air_temperature=295.0,
    air_density=1.1,
    air_vapor=0.005,
    air_pressure=1.0e5,
    surface_temperature=surface_temperature,
    wind_speed=5.0,
    measurement_height=10.0,
)
assert bool(rates.successful)
# rates.sensible_heat and rates.water_enthalpy: W/m²
# rates.water_mass: kg/m²/s
```

Inputs must be finite; gas density and height are positive, vapor is in `[0,1)`,
wind is nonnegative, air temperature lies inside the native caloric domain, and
surface temperature is liquid and below the local boiling-pressure limit.
`successful` certifies that rate evaluation, not a subsequent finite transfer.

## Declared stability law

`stability="neutral"` uses `f = 1`. The default `"bulk-richardson"` applies an
explicit empirical bulk-Richardson closure. Define `a = R_v/R_d - 1` and use a
first-order dry-adiabatic height correction to the air temperature:

```text
T_virtual_air     = (T_air + g z / cp_air) (1 + a q_air)
T_virtual_surface = T_s (1 + a q_s)
Ri = g z (T_virtual_air - T_virtual_surface) / (T_virtual_air U²)
f  = (1 + 5 Ri)^(-2)     for Ri >= 0
f  = (1 - 16 Ri)^(1/4)   for Ri < 0.
```

The shared-pressure and first-order height approximations are appropriate only
for a shallow surface layer, not deep-column buoyancy calculations. The
multiplier suppresses stable exchange and enhances unstable exchange. It is a
declared empirical shape, **not** an iterative Monin–Obukhov similarity solution
or a claim of field-calibrated accuracy for arbitrary conditions. The fixed
constants 5 and 16 and the branch exponents are part of the numerical model;
`gravity` is a positive static configuration parameter.

At exactly zero wind, all exchange is exactly zero without a speed floor.
For unstable disequilibrium the exchange approaches zero proportionally to
`sqrt(U)`, so its wind derivative at zero is not regular. The model does not
reinterpret this as natural convection. Zero temperature and humidity
differences produce zero flux at any admissible ventilation.

## Exact paired finite transfers and rejection

`paired_surface_transfer` applies the same already-integrated water and total
energy to a fixed-volume atmospheric cell and the opposite slab budget. Air
inventories are dry mass and total water in kg/m², internal energy in J/m², and
volume per area in m. The native moist equilibrium adjustment supplies phase
partition and temperature before and after the transfer.

```python
from phydrax.applications.atmosphere import paired_surface_transfer

volume = 100.0
vapor = 0.005
air_temperature = 295.0
density = 1.0e5 / (thermo.gas_constant(vapor, 0.0, 0.0) * air_temperature)
mass = density * volume
air_energy = mass * thermo.energy(density, air_temperature, vapor, 0.0, 0.0)
# Evaluate at this actual gas density, rather than reusing an unrelated state.
rates = exchange.evaluate(
    thermo, air_temperature, density, vapor, 1.0e5,
    slab_plan.temperature(slab, thermo), 5.0, 10.0,
)
assert bool(rates.successful)
dt = 10.0
result = paired_surface_transfer(
    thermo, slab_plan, slab,
    air_dry_mass=mass * (1 - vapor),
    air_water_mass=mass * vapor,
    air_internal_energy=air_energy,
    air_volume=volume,
    water_mass=dt * rates.water_mass,
    energy=dt * (rates.sensible_heat + rates.water_enthalpy),
)
assert bool(result.successful)
```

For accepted elements, exactly the same `DeltaM` and `DeltaE` enter both sides:

```text
M_air_new  = M_air_old  + DeltaM
M_slab_new = M_slab_old - DeltaM
E_air_new  = E_air_old  + DeltaE
E_slab_new = E_slab_old - DeltaE.
```

Floating-point sums have ordinary rounding error; there is no hidden residual
redistribution. The helper rejects negative remaining surface water, excessive
dew removal beyond available donor vapor, invalid air equilibrium, or an
out-of-domain slab. **Both sides remain unchanged and committed transfers are
zero** for every rejected batch element. There is no post-computation cap on
water or energy. Exhausting exactly all surface water is admissible if the
remaining dry slab and air are physical; requesting more than is available is
not. Separate cells are separate transactions.

For an RK or other coupled integrator, supply its integrated flux quadratures
rather than independently recomputing a second surface budget. Certify all
stage rate statuses. Precipitation, external radiative heat, and mechanical work
must retain their own explicitly owned paired or external budgets.

## Differentiation, parameter identity, and qualification

Dry slab heat capacity and the two transfer coefficients are scalar numeric
JAX leaves. They remain fixed during a forward trajectory but can be replaced
with `eqx.tree_at`, differentiated, or partitioned using PHYDRAX's native
trainable-state machinery. Constructors and rate/state certification enforce
positive heat capacity and nonnegative coefficients, including after numeric
parameter replacement. Moist caloric constants remain the existing native
nontrainable thermodynamic reference.

The humidity disequilibrium, not the numerically zero flux, selects the donor
when the moisture coefficient is zero. Its coefficient derivative therefore
follows the physically admissible nonnegative one-sided limit, including dew
carrying air-temperature vapor enthalpy.

Static `plan_id` describes the structural law and caloric configuration, not the
values of numeric calibration leaves. Restart or experiment identity must retain
those numeric leaves through the owning application's native checkpoint path;
a static fingerprint alone does not identify calibrated parameter values.

State evaluation, finite transfers, and JAX transforms support batched states.
Derivatives are branch-local: `rates.derivative_valid` excludes active zero-wind
boundaries, `Ri = 0`, and a donor sign switch when air and surface enthalpies
differ. Coincident donor temperatures remove that donor kink. It also excludes
physical temperature/humidity domain boundaries. Finite AD numbers alone do
not certify derivatives across rejection, freezing, or other phase transitions;
coupled callers must combine this metadata with native moist phase-adjustment
and transaction derivative status.

Run the dedicated evidence tool from the repository root:

```sh
PYTHONPATH=. python tools/surface_exchange_qualification.py --steps 120 --time-step 10
```

It prints JSON with actual sensible/water-energy fluxes in W/m², water fluxes in
kg/m²/s, closed air/slab residuals, evaporation/dew SST changes, analytic
no-gradient/no-wind limits, atomic exhaustion rejection, and AD versus centered
finite differences for boundary inputs and numeric coefficient/capacity scales.
These are measured scenarios, not a universal scientific qualification claim.
The dedicated unit tests additionally exercise donor enthalpy references,
water-dependent heat capacity, incompatible references, freezing rejection,
batched atomicity, and derivative-boundary status.

For quantity semantics at application boundaries, use the existing native
`GeophysicalQuantity`/units machinery described in
[geophysical contracts](guides_geophysical_contracts.md); the numeric moist
surface kernels themselves take SI arrays, as does native moist thermodynamics.
