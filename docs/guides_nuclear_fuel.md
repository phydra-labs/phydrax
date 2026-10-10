# Nuclear fuel performance

`phydrax.applications.nuclear_fuel` simulates one light-water-reactor fuel rod
through a prescribed irradiation history: heat conduction, finite-strain
mechanics with pellet-cladding contact, fission-gas and helium behavior, rod
gas pressure, and cladding corrosion, solved together at every time step. It
builds on Phydrax finite elements, nonlinear solvers, nuclear data, the
production runtime, and the observation and inference packages.

## Scope and limits

The modeled system is sintered UO₂ pellets (95–97% of theoretical density) in
stress-relief-annealed Zircaloy-4 cladding under PWR coolant conditions. Two
geometric representations share the material laws:

- `layered-radial`: radial finite elements in axial zones, with generalized
  plane-strain axial force balance and one communicating gas volume;
- `axisymmetric-rz`: one connected meridional mesh with radial and axial
  displacement, axial conduction, and interface exchange.

The application does not model MOX, absorber-bearing or metallic fuel, TRISO,
plates, accident-tolerant cladding, BWR coolant, 3-D defects, LOCA/RIA,
post-CHF heat transfer, cladding breach, chemical PCI/SCC failure probability,
full-core neutronics, or isotope depletion. Mechanical contact stress is not a
failure prediction. Fission gas entering the rod free volume is not release
outside the cladding.

The capability profile is **candidate-unreleased**. The repository
verification campaign is synthetic numerical verification (equation
reproduction, conservation, convergence, restart), not experimental
validation. Comparison against measured rods requires IFPE histories whose
data rights are pending, plus an independent locked evaluation. See
[nuclear fuel sources](nuclear_fuel_sources.md) for the cited equations, data
access, and the source data a physical (non-synthetic) run requires.

## Quick start

The workflow is: specification → irradiation history → coolant boundary →
model bundle → admission → preparation → simulation → results and evidence.
The inputs below are the synthetic numerical inputs of the verification tool;
they are not data for any real rod.

```python
from pathlib import Path

import jax
import numpy as np

from phydrax.applications.nuclear_fuel import (
    admit_fuel_rod,
    AthermalRelease,
    BoreEmissivity,
    CladdingSpecification,
    EnergyDeposition,
    FissionGasModel,
    FuelModelBundle,
    FuelRodSimulation,
    FuelRodSpecification,
    FuelStepStatus,
    GapConductanceLaw,
    GasAccommodationLaw,
    HeliumGasModel,
    HeliumTransportTable,
    IrradiationHistory,
    PelletSpecification,
    PlenumTemperatureClosure,
    prepare_fuel_rod,
    PrescribedWallTemperature,
    RadialPowerProfileModel,
    RelocationHistory,
    RodSpecification,
)
from phydrax.materials import PropertyValidityRange
from phydrax.nuclear import FissionGasYieldTable
from tools.nuclear_fuel_qualification import numerical_helium_production

jax.config.update("jax_enable_x64", True)

radius, height, length = 4.095e-3, 10.0e-3, 0.3
specification = FuelRodSpecification(
    PelletSpecification(
        radius=radius,
        height=height,
        density_fraction=0.95,
        uranium_235_weight_fraction=0.035,
        grain_radius=5.0e-6,
        resintering_density_change=100.0,
        roughness=1.0e-6,
        open_porosity_fraction=0.005,
    ),
    CladdingSpecification(
        inner_radius=4.18e-3, outer_radius=4.75e-3, roughness=0.5e-6, cold_work=0.5
    ),
    RodSpecification(
        stack_length=length, plenum_length=0.03, fill_pressure=0.103e6, fill_temperature=300.0
    ),
)

times = np.array([0.0, 60.0, 120.0, 3600.0])
power = np.array([[0.0], [0.0], [3000.0], [3000.0]])  # W/m, one axial zone
wall = np.array([300.0, 600.0, 600.0, 600.0])  # K
history = IrradiationHistory(
    times=times,
    zone_edges=np.array([0.0, length]),
    linear_power=power,
    fast_flux=power * 5.0e13,
    coolant_inlet_temperature=wall,
    coolant_pressure=np.array([0.103e6, 15.5e6, 15.5e6, 15.5e6]),
    coolant_mass_flux=np.full_like(times, 3000.0),
)
coolant = PrescribedWallTemperature(times, wall[:, None])

surface_to_volume = 2.0 / radius + 2.0 / height
bundle = FuelModelBundle(
    applicability="equation-reproduction",
    radial_power=RadialPowerProfileModel(
        energy_per_fission=FissionGasYieldTable.endf_b_viii_0().energy_per_fission,
        scattering_cross_section_barn=5.0,
        scattering_source_id="synthetic-numerical-verification:not-spectrum-data",
        form_depth_unit_m=1.0e-3,
    ),
    fission_gas=FissionGasModel(
        mode_count=8,
        athermal=AthermalRelease(
            geometric_surface_to_volume=surface_to_volume,
            total_surface_to_volume=surface_to_volume,
        ),
    ),
    helium=HeliumGasModel(
        radial_cell_count=40,
        transport=HeliumTransportTable(
            temperature=(293.0, 2500.0),
            diffusivity=(1e-20, 1e-20),
            henry_constant=(1e12, 1e12),
            validity=PropertyValidityRange(
                {"temperature": (293.0, 2500.0, "K")},
                source_id="synthetic-numerical-verification:helium-transport",
            ),
            source_id="synthetic-numerical-verification:helium-transport",
        ),
    ),
    helium_source=numerical_helium_production(),
    relocation_history=RelocationHistory(
        policy="fission-activated-permanent-power-recoverable",
        source_id="declared-numerical-translation:not-FAST-caller",
    ),
    gap=GapConductanceLaw(accommodation=GasAccommodationLaw.gap_comparison()),
    bore_emissivity=BoreEmissivity(
        value=0.3,
        temperature_range=(293.0, 2500.0),
        source_id="synthetic-numerical-verification:gray-bore",
    ),
    plenum_temperature=PlenumTemperatureClosure(
        offset=0.0,
        source_id="declared-numerical-plenum:coolant-outlet-zero-offset",
        prescribed_source_id="declared-numerical-plenum:terminal-wall-proxy",
    ),
    deposition=EnergyDeposition(
        fuel_fraction=0.95,
        cladding_fraction=0.025,
        source_id="synthetic-numerical-verification:declared-energy-partition",
    ),
)

admission = admit_fuel_rod(
    specification, history, coolant, bundle=bundle, reduction="layered-radial"
)
prepared = prepare_fuel_rod(admission, fuel_cells=4, cladding_cells=4, quadrature_order=3)

simulation = FuelRodSimulation(
    prepared, time_step=60.0, checkpoint_directory=Path("/tmp/fuel-quick-start")
)
result = simulation.run()
simulation.close()

final = result.final_state
print(bool(result.successful), float(final.time))
print(float(np.max(final.fields.temperature_fuel)), float(final.fields.pressure))
for attempt in result.attempts:
    status = FuelStepStatus(int(attempt.status))
    print(float(attempt.time), status.name, bool(attempt.source_support.all()))
```

`numerical_helium_production()` returns a `HeliumSource` built from synthetic
decay and yield data; a physical run supplies evaluated data through
`phydrax.nuclear.HeliumProductionPlan` (see [Model bundle](#model-bundle)).

The repository examples run longer histories with the same inputs:

```sh
python examples/nuclear_fuel_base_irradiation.py --smoke \
  --checkpoint-directory /tmp/phydrax-fuel-checkpoints
python examples/nuclear_fuel_calibration.py --maximum-optimizer-steps 24
```

`nuclear_fuel_base_irradiation.py` runs startup heating and pressurization,
power ascension, base hold, ramp, load following, shutdown, cooldown, and
restart. It takes two steps, writes a checkpoint, resumes from it, and prints
physical observables, the energy and gas ledgers, attempt status codes, and the
attempts outside source support. `--reduction axisymmetric-rz`
selects the meridional mesh; `--base-duration`, `--base-power`,
`--transient-duration`, `--time-step`, `--fuel-cells`, and `--axial-layers`
change the history and mesh; `--restart` resumes an existing checkpoint
directory. Without `--smoke`, the base hold is 3.0×10⁷ s at 20,000 W/m.

## Inputs

### Fabrication

`PelletSpecification`, `CladdingSpecification`, and `RodSpecification` describe
the as-fabricated design; `FuelRodSpecification` binds them. Values are SI.
An optional `units` mapping of `phydrax.units.UnitDefinition` converts named
dimensioned inputs once at construction; wrong dimensions and unknown names
are refused. Pressures are absolute.

`fill_pressure` and `fill_temperature` fix the fill-gas inventory in the
as-fabricated free volume; the hot rod pressure is solved from the current
inventory, free volume, and temperatures. `fill_gas_mole_fractions` follow
`FUEL_GAS_SPECIES` order (He, Ar, Kr, Xe, N2). `FuelForm`, `CladdingAlloy`,
and `CladdingHeatTreatment` name the admitted material system.

### Irradiation history

`IrradiationHistory` holds strictly increasing times from zero, axial
`zone_edges` that span the fuel stack, per-zone recoverable linear fission
power [W/m] and fast flux, and rod-wide coolant inlet temperature, pressure,
and mass flux. Power and flux are piecewise linear in time;
`energy_per_length` and `fast_fluence` are their exact integrals, so interval
averages do not depend on the step partition. Queries outside the history,
backward intervals, and zero-length intervals are refused.
`IrradiationHistory.conditions(time)` returns the interpolated
`IrradiationConditions`.

Burnup is referenced to the initial heavy-metal inventory
(`phydrax.nuclear.HeavyMetalReference`); `ExposureState` reports specific
energy, FIMA, fission density, and fast fluence. Fission-gas yields and
recoverable fission energies come from `FissionGasYieldTable.endf_b_viii_0()`
in the fissioning order U-235, U-238, Pu-239, Pu-240, Pu-241, Pu-242.

### Coolant boundaries

| Boundary | `CoolantBoundaryKind` | Behavior |
|---|---|---|
| `PrescribedWallTemperature(times, wall_temperatures)` | `prescribed-wall-temperature` | Cladding outer surface temperature per axial zone, piecewise linear in time; must cover the whole history. |
| `SingleChannelCoolant(hydraulic_diameter=..., flow_area_per_rod=...)` | `single-channel` | FRAPCON forced convection and Jens–Lottes subcooled boiling with IAPWS-IF97 liquid properties and a steady enthalpy rise; requires positive `coolant_mass_flux` at every history time. |
| `CoupledChannelCoolant(history, wall_transfer, release_index, trust_policy, profile_id=..., at_time=...)` | `coupled-channel` | Channel enthalpy unknowns per axial zone solved with the rod through `phydrax.applications.thermofluids.LiquidChannelHeatBalance`. |

The wall law reports `nucleate_boiling_supported` over the Jens–Lottes
measured pressure, bulk-temperature, and heat-flux range; that support appears
in step evidence under the `coolant` source slot. It does not certify fully
developed turbulent forced convection.

`CoupledChannelCoolant` requires a trusted release index containing a channel
profile with the exact support tuple returned by
`CoupledChannelCoolant.support_tuple(history, wall_transfer)` under
`thermofluids.channel-heat`: the history ID, wall-law ID, cell count,
`iapws-if97-region-1` water, and the `reference-mass-upwind` method. The channel
model has prescribed through-flow and fixed reference masses; it has no
momentum equation and no two-phase flow. Channel admission does not release
the fuel capability.

## Model bundle

`FuelModelBundle` holds every closure of the rod model. The UO₂ and Zircaloy
property, eigenstrain, creep, corrosion, and hydrogen laws are the fixed
reference laws of the material system; `cladding_creep` and
`cladding_plasticity` accept configured `ZircaloyCreep` and
`ZircaloyPlasticity` instances. The caller supplies:

| Argument | Type | Content |
|---|---|---|
| `applicability` | `FuelApplicabilityPolicy` | Required; see [Source support](#source-support-and-applicability). |
| `radial_power` | `RadialPowerProfileModel` | TUBRNP radial power with a spectrum-averaged scattering cross section, its source ID, and the form-depth unit. |
| `fission_gas` | `FissionGasModel` | Intragranular diffusion (`TurnbullDiffusivity`, `SpeightTrapping`), grain-face bubbles, grain growth, microcracking, HBS, and `AthermalRelease`. |
| `helium` | `HeliumGasModel` | Helium transport on spherical radial cells with a `HeliumTransportTable`. |
| `helium_source` | `HeliumSource` | `HeliumProductionPlan`, its fixed `NuclideInventory`, and reference volume [m³]. |
| `relocation_history` | `RelocationHistory` | History translation of the relocation correlation and its source. |
| `gap` | `GapConductanceLaw` | Gas, solid-contact, and radiation gap conductance with a `GasAccommodationLaw`. |
| `bore_emissivity` | `BoreEmissivity` | Plenum bore gray emissivity over its sourced temperature range. |
| `plenum_temperature` | `PlenumTemperatureClosure` | Coolant outlet temperature plus an offset, with separate sources for channel and prescribed-wall boundaries. |
| `deposition` | `EnergyDeposition` | Fractions of fission energy deposited directly in fuel and cladding; the remainder heats the coolant. |
| `friction` | `CoulombFriction` or `None` | Coulomb coefficient for axial contact; `None` is frictionless. |

Each closure record validates its values at construction and carries a
`source_id`; records with numeric values also carry a `closure_id`. The
`bundle_id` covers the applicability policy, every law and validity range, and
each closure ID, so any change of closure or source changes the bundle
identity.

`HeliumSource` requires ground-state He-4 and an inventory whose network and
nuclide identities match the production plan. The inventory amounts are held
fixed (an actinide alpha-source approximation); the application does not
deplete actinides.

### Source support and applicability

`FuelApplicabilityPolicy` has no default:

- `"strict-source-support"`: every closure must be inside its source validity
  range at every step. A converged physical step outside a range is a terminal
  domain exit; the run stops without retrying.
- `"equation-reproduction"`: the cited equations are evaluated outside their
  ranges, for synthetic numerical work and source comparison. Each step still
  reports which closures were out of range.

The relocation-history and plenum-temperature closures have no source support
at any state (`STATICALLY_UNSOURCED_SLOTS`). Under `"strict-source-support"`,
`FuelModelBundle` therefore refuses construction with a `ValueError` naming
`relocation_history` and `plenum_temperature`; synthetic and
equation-verification work uses `"equation-reproduction"`.

Neither policy accepts a nonpositive temperature, invalid geometry, failed local
root, negative inventory, failed nonlinear solve, or broken balance. Validity
coordinates keep their source meaning: the NFI conductivity density range is
checked against fabricated porosity, and rod-average burnup limits are not
applied to local rim burnup.

## Admission and preparation

`admit_fuel_rod(specification, history, coolant, bundle=..., reduction=...,
axial_interaction=...)` checks that the history zones cover the fuel stack,
that the pellet density is within the admitted 95–97% TD, that a prescribed
wall table matches the zones and covers the history, that a single channel has
positive mass flux, that a coupled channel uses the same history, and that
`axial_interaction="coulomb"` has a `CoulombFriction` closure. It returns a
`FuelRodAdmission`.

`prepare_fuel_rod(admission, fuel_cells=..., cladding_cells=...,
quadrature_order=..., axial_cells_per_zone=...)` builds the geometry, finite
element spaces, solver coordinates, and nonlinear solver, and returns a
`PreparedFuelRod`. `maximum_coupling_evaluations` (default 4) and
`coupling_tolerance` (default `1e-9`) control the gas/thermomechanical coupling
described in [Time step](#time-step). `nonlinear_termination` and
`linear_resources` override the Newton termination and linear-solve resources.

`PreparedFuelRod.initial_state(fill_pressure_scale=...)` returns the
fabricated `FuelRodState`. `fill_pressure_scale` scales the fill inventory, is
differentiable for calibration, and must be one finite positive scalar.

## Simulation and results

`FuelRodSimulation(prepared, time_step=..., checkpoint_directory=...)` runs the
history on the Phydrax production runtime. It stops exactly at every history
breakpoint (and every prescribed wall-temperature time), adapts the step
between `minimum_time_step` and `maximum_time_step`, retries a refused step
with a smaller step up to `maximum_retries` times, and writes a durable
checkpoint every `checkpoint_interval` accepted steps. `acceptance_policy`
takes a `FuelAcceptancePolicy` with the per-step increment limits and balance
tolerances. Call `close()` to release the checkpoint store.

`run()` returns a `FuelRodResult`:

- `states`: accepted `FuelRodState` values in time order; `final_state` and
  `times` are shortcuts;
- `attempts`: one `FuelStepEvidence` per executed attempt, accepted or
  refused, with matching `attempt_ids`;
- `successful` and `runtime_result`: the production-runtime outcome.

A `FuelRodState` holds the physical fields (`FuelRodFields`, SI units), the
fuel and cladding integration-site histories, `ExposureState`, the radial power
state, and the conserved `FuelGasLedger` (moles per species in
`FUEL_GAS_SPECIES` order) and `FuelEnergyLedger` (joules relative to the cold
fabricated state). Each ledger exposes its `balance_error`.

### Step evidence

`FuelStepEvidence` describes one attempted step:

| Field | Content |
|---|---|
| `status` | `FuelStepStatus` code; `accepted` is `status == ACCEPTED`. |
| `terminal_domain_exit` | Converged physical step outside its admitted range. |
| `solver.nonlinear` | `FuelNonlinearEvidence`: Newton status, residual, iterations. |
| `solver.linear` | `FuelLinearEvidence`: final linear solve status, rank, condition estimate, residual. Unavailable rank is −1 and unavailable condition is NaN. |
| `solver.coupling` | `FuelCouplingEvidence`: coupling status, iterations, evaluations, `joint_residual` (norm of the live coupled residual at the endpoint in scaled solver coordinates: thermomechanics with the endpoint gas plus the hydrostatic-stress closure), `gas_admissible` (physical admissibility of the endpoint gas), and the work totals `thermomechanical_iterations`, `thermomechanical_residual_evaluations`, `gas_block_evaluations` (both evaluation counts include the endpoint verification). |
| `solver.local_roots_converged` | All constitutive and gas local roots converged. |
| `physics` | `FuelPhysicalEvidence`: physical admissibility, `in_range`, minimum gap, contact complementarity and friction errors, increments, accommodation extrapolation. |
| `balances` | `FuelBalanceEvidence`: `energy_error` [J] and `gas_error` [mol]. |
| `source_support` | `FuelSourceSupport`: per-slot source support. |

`source_support.of(slot)` returns the support of one `FuelSourceSlot` (for
example `"gap"`, `"cladding_creep"`, or `"coolant"`); `source_support.all()`
reports whether every closure is inside its range; `source_support.slots` and
`source_support.source_ids` list the slots and their sources. `all()` and
`of()` also work on stacked evidence.

When several checks fail, `status` reports the first in this order:
`NONLINEAR_FAILURE`, `COUPLING_FAILURE`, `JOINT_RESIDUAL_FAILURE`,
`LOCAL_ROOT_FAILURE`, `PHYSICAL_FAILURE`, `TERMINAL_DOMAIN_EXIT`,
`ENERGY_IMBALANCE`, `GAS_IMBALANCE`, `CONTACT_FAILURE`, `INCREMENT_LIMIT`. A
solver failure outranks the physical checks because its candidate has no
physical meaning.

### Restart and checkpoints

Checkpoints store the complete accepted state (fields, constitutive and gas
histories, ledgers) and the step controller. `run(restart=True)` resumes from
the latest checkpoint in `checkpoint_directory`; a new `FuelRodSimulation` on
the same prepared rod and directory can resume a closed run. The V14 and
`contact-coolant` verification cases check that the resumed states equal an
uninterrupted run bitwise.

`run(initial_state)` accepts only an unirradiated state at time zero. An
irradiated rod cannot be initialized from burnup alone; it resumes from a
checkpoint.

### Sharing one compiled step

`FuelNumericalExecution(prepared)` compiles one fuel step for every rod with the
same structure; `fuel_numerical_execution_id(prepared)` returns that structural
identity. Passing `numerical_execution=` to `FuelRodSimulation` reuses the
compiled step, while each rod keeps its own prepared, bundle, and history
identities, checkpoints, and results. A rod whose mesh, reduction, boundary,
axial interaction, or closure roles differ is refused at binding with a request
to regroup.

## Physical model

### Time step

Each step is partitioned implicit backward Euler. A bounded fixed-point
iteration alternates two blocks at the same proposed step:

1. the gas block advances grain growth, fission gas, HBS, passive Kr/Xe
   tracers, and helium at every fuel site from the accepted state, using the
   current temperature, hydrostatic stress, and rod helium pressure;
2. the thermomechanical block solves temperature, displacement, axial strain,
   contact, friction, rod pressure, free gas, coolant heat flux, and any channel
   enthalpy with that gas candidate held fixed.

Every gas evaluation starts from the accepted history, so repeated coupling
evaluations do not advance production, burnup, or irreversible history twice.
After the iteration, an independent gas evaluation at the endpoint certifies
the candidate: with that gas, the live coupled residual (the system the
implicit derivative linearizes) is evaluated once in the scaled solver
coordinates of the coupling termination. A step is accepted only if the Newton
and coupling solves succeed, that joint residual is finite and within
`coupling_tolerance`, the endpoint gas is admissible, and the physical,
local-root, contact, energy, gas, and increment checks pass. A small iterate
change alone never accepts a step; an exhausted coupling budget refuses the
step and the runtime retries with a smaller one. A refused attempt leaves the
accepted state unchanged.

Heat capacity enters through enthalpy differences. Gas conduction, solid
contact, and radiation are separate gap-conductance terms; radiative
conductance stays finite when the surface temperatures coincide. Newton trials
with negative gas-species amounts or singular or inverted fuel or cladding
deformation are refused before the constitutive and mixture evaluations.

### Fission gas and helium

Gas reservoirs and transfer counters are distinct; cumulative fission-gas
release is a derived ratio. `FissionGasSpeciesState`,
`FissionGasSpeciesTransfers`, and `FissionGasSpeciesStepResult` carry passive
Kr and Xe identities in `FISSION_GAS_TRACER_SPECIES` order through the same
total bubble and HBS update, so changing the yield mix does not relabel gas
already retained. `FissionGasStepResult`, `FissionGasTransfers`, and
`FissionGasDiagnostics` report the total-gas candidate, its transfers, and the
local-solve status.

`HeliumGasModel(radial_cell_count=...)` stores dissolved and trapped helium on
spherical radial cells. Each backward-Euler step uses the exact
modified-Helmholtz solution within each shell and a finite-pressure Henry
boundary; signed transfer allows fill-gas helium absorption as well as release.
The fission-gas mode count and the helium cell count are independent; one
grain-growth evaluation per gas candidate serves both models.

`HighBurnupStructureFormation` gives the restructured fraction from effective
burnup. `HighBurnupStructurePores` evolves pore number `N`, pore gas `A`,
integrated central gas variance `B` [atoms²/m³], and vacancies `W` [1/m³] with
one four-variable backward-Euler root, using a positive dimer-anchored
two-node Gauss–Radau reconstruction of the source cluster generator (see
[sources](nuclear_fuel_sources.md#printed-source-discrepancies)). Pore swelling
is `omega(T)*A + Omega*W` per reference fuel volume, with the gas atomic volume
`omega(T)` from Barani et al. Part II Eq. 17 and the Schottky volume `Omega`.
HBS boundary gas is a separate retained reservoir.

`advance_fuel_gas` evaluates the gas block at one site and returns
`FuelGasBlockResult`; `PreparedFuelRod.residual` takes that block as its fixed
gas input.

### Free volume and pressure

The communicating free volume is the plenum, radial gap, dishes and chamfers,
connected cracks, open porosity, and connected grain-boundary cavities. Closed
bubbles and HBS pores are retained storage. At one rod pressure,

```text
p × Σ(V_j / T_j) = n_total × R
```

over the compartments; uniform pressure assumes the gas communicates along the
rod and is not an axial-flow model.

## Observations

`FuelObservationPlan(prepared, grid, sensors, samples)` binds `FuelSensor`
definitions to times of a `FuelFixedGrid`. A sensor has a `FuelObservableKind`
(thermocouple or centerline temperature, in-pile or cold pressure, metal or
oxide-inclusive diameter, elongation, oxide thickness, hydrogen, incremental or
cumulative fission-gas release), a physical position, and an optional response
time, drift-valid window, and detector reference. Cold pressure and the two
diameter kinds are valid only where the computed rod temperature is within
`cold_tolerance` of the sensor's `cold_temperature`.
Incremental release after refabrication uses `FuelGasAncestry`.

`evaluate(result)` and `evaluate_states(states)` return a
`FuelObservationResult` with per-sample validity; `FuelSensorParameters`
applies detector gain, offset, and drift. `evaluate(result)` reports
`successful` only if the production run succeeded. `thermocouple_response`
integrates the first-order sensor lag exactly over each piecewise-linear
interval.

## Sensitivities

`FuelFixedGrid(times, prepared_id)` freezes accepted step times;
`replay_fuel_grid(prepared, grid, parameters=...)` re-solves every transition on
that grid without adaptivity or retries and returns `FuelReplayResult` with
per-step success and branch-change flags. The step controller's choice of grid
is not differentiated.

`fuel_parameterization(bundle, names)` selects named law scalars
(`FuelParameterName`) and returns a `FuelParameterization` and their nominal
values; each replacement value is checked against its constructor's domain.
`FuelSensitivityPlan(prepared, grid, parameterization, values,
parameter_capacity=..., maximum_working_set_bytes=...)` computes forward
tangents of the full state history. Each step differentiates the converged
coupled endpoint (field unknowns plus fuel-site hydrostatic stresses) with one
multi-right-hand-side solve, including gas feedback through temperature,
stress, and helium pressure. `FuelSensitivityResult` reports per-step
linear-solve and branch-change evidence: a contact, stick/slip, saturation, or
HBS branch change is reported, not smoothed.

## Calibration

`FuelCalibrationRod(observation, data, covariance, case=...)` pairs an
observation plan with data, a `CholeskyCovarianceAction`, and a
`ScientificCase`. `FuelCalibrationPlan(rods, parameterization, parameter_space,
campaign, prior_source_ids=...)` separates shared physical coordinates from
per-rod fill-pressure scales and sensor nuisances (`FuelCalibrationParameters`).
Its rods must be exactly the calibration cases of a `ScientificCampaign`, whose
locked-evaluation cases stay out of the fit.

- `prepare_fuel_least_squares(plan)` returns a composite least-squares problem
  with the prior for `phydrax.optim.composite_least_squares`;
  `include_prior=False` returns a data-only nonlinear least-squares problem.
- `prepare_fuel_posterior(plan)` returns a `PosteriorProblem`.

`signal_derivatives` selects how the sensor-signal Jacobian is applied:

- `FuelSignalJacobianActions()` (default) differentiates the replay at every
  Jacobian action and stores no Jacobian.
- `FuelDenseSignalJacobian(materialization=MaterializationPolicy(...),
  maximum_working_set_bytes=...)` stores the Jacobian of the raw signals with
  respect to the physical and fill-pressure coordinates, computed with one
  `FuelSensitivityPlan` multi-right-hand-side solve per accepted step.
  `materialization` bounds the stored matrix of all rods and refuses a larger
  one before allocation; `maximum_working_set_bytes` bounds each rod's tangent
  workspace. Requires `include_prior=True`.

Sensor gain, offset, drift, whitening, and the prior stay in the composed model
under both choices. The calibration example uses
`FuelDenseSignalJacobian(materialization=MaterializationPolicy(max_entries=64,
max_bytes=512), maximum_working_set_bytes=64 * 1024 * 1024)` with
`GeneralizedGaussNewton(curvature_materialization=MaterializationPolicy(
max_entries=64, max_bytes=512))` for eight search coordinates: one physical law
scalar, one fill-pressure scale, and gain, offset, and drift of two sensors.

### Identifiability

`evaluate_fuel_calibration_identifiability(plan, position, rank_policy=...)`
returns a `FuelIdentifiabilityReport` from the whitened data-only Jacobian at a
search position; priors do not enter the rank.
`evaluate_fuel_identifiability(whitened_jacobian, parameter_layout,
observation_layout)` analyzes a supplied Jacobian. The report holds the Fisher
information, normalized column correlations, weak directions, and condition
number. Temperature data alone cannot separate conductivity, initial-gap,
relocation, and diffusivity errors; correlated columns and weak directions show
such confounding.

## Ensembles

`FuelEnsembleMember(prepared, member_id=..., checkpoint_directory=...,
time_step=..., maximum_steps=...)` describes one independent rod run.
`FuelEnsemblePlan(members, maximum_working_set_bytes=...,
maximum_output_bytes=...)` checks the working-set and retained-output budgets
before running, groups members by `fuel_numerical_execution_id`, and
`run()` returns a `FuelEnsembleResult` with one `FuelRodResult` per member.
Each member keeps its own checkpoints and identities.

## Qualification profiles

`nuclear_fuel_profiles()` declares the candidate model family for capability
discovery. `nuclear_fuel_execution_profile(bundle, reduction=..., coolant=...,
axial_interaction=..., prepared_id=..., runtime_id=..., ...)` binds candidate
support to one prepared model and measured runtime evidence. Both profiles are
unreleased: they require the V1–V15 verification gates and the scientific
release gates (IFPE data rights, independent locked evaluation, ancestry and
fabrication split, precommitted observable criteria, runtime precision and
resources).

## Verification campaign

`python -m tools.nuclear_fuel_qualification` (run from the repository root)
executes the numerical verification cases:

| Case | Check | Reference |
|---|---|---|
| V1 | Constant-property cylinder conduction | Analytic cylinder |
| V2 | Nonlinear cylinder conduction | Kirchhoff transform |
| V3 | Generalized plane strain | Lamé closed-end tube |
| V4 | Finite-strain inflation | Rivlin tube |
| V5 | Free thermal expansion | Stress-free eigenstrain |
| V6 | Shrink-fit contact | Lamé shrink fit |
| V7 | Booth release | Booth sphere series |
| V8 | Gas pressure at unequal temperatures | Communicating-volume ideal gas |
| V9 | Paired gap heat exchange | Concentric-cylinder heat flow |
| V10 | Constant-stress creep | CREPR closed form and Norton tube |
| V11 | RZ thermomechanics | Manufactured solution |
| V12 | Layered versus RZ | Independent reductions |
| V13 | Whole-history balances | Energy and gas conservation |
| V14 | Fault rollback and restart | Accepted-checkpoint replay |
| V15 | Whole-history refinement | Richardson extrapolation |

Each case also runs a deliberate fault that its check must detect; a case
closes only when both the numerical check and the fault detection pass. V13
compares deposited energy with an independent integral of the prescribed power
and deposition fractions. V15 requires fine-grid contraction unless both
refinement differences are at or below `sqrt(float64 epsilon)` times the metric
scale, in which case the metric is reported as `roundoff` without an order
claim.

The supplemental `contact-coolant` case runs both reductions with each coolant
boundary under Coulomb friction. It checks that contact closes, that the
friction force stays within the Coulomb bound and is nonzero while closed, that
channel energy balances hold, that a checkpointed run resumed in a new
simulation reproduces the uninterrupted run bitwise, and that
`"strict-source-support"` refuses the synthetic bundle. It is reported under
`supplemental_cases`, gates `successful`, and does not enter the closure
matrices.

```sh
python -m tools.nuclear_fuel_qualification --smoke --output /tmp/fuel-verification.json
python -m tools.nuclear_fuel_qualification --cases V5 contact-coolant --output /tmp/fuel-cases.json
python -m tools.nuclear_fuel_qualification --input-reports /tmp/a.json /tmp/b.json --output /tmp/fuel-verification.json
```

`--smoke` uses coarse refinements and short histories (a 3480 s base at
3000 W/m with 60 s transients; `contact-coolant` covers 0–0.6 s instead of
0–1.6 s). The full design uses a 3.0×10⁷ s base at 20,000 W/m with 3600 s
transients. `--cases` selects cases (run in campaign order), which bounds
compilation memory when cases run in separate processes. Per-case artifacts
and checkpoints are kept in `--artifact-directory`, or beside `--output` as
`<output-stem>.artifacts`. `--input-reports` combines reports produced by the
same source build and environment without running physics. The command exits
nonzero if any numerical check or fault detection fails. A passing report does
not resolve the IFPE data-rights or locked-evaluation gates.

## Benchmarks

`python -m benchmarks.nuclear_fuel_performance` measures preparation, step
lowering and compilation, first and warmed steps, the full production history,
compiler memory, retained-array bytes, checkpoint and restart, and tangent
cost. Its fifteen capacity cases vary radial cells, axial layers, RZ axial
cells, fission-gas modes, helium radial cells (40, 80, 160), and sensitivity
parameter count independently. Work totals sum
`solver.coupling.thermomechanical_iterations`,
`thermomechanical_residual_evaluations`, `evaluations`, and
`gas_block_evaluations` over all attempts. Retained-array bytes are logical
payload, not measured resident memory.

```sh
python -m benchmarks.nuclear_fuel_performance --smoke --output /tmp/fuel-performance.json
python -m benchmarks.nuclear_fuel_performance --case-index 0 --output /tmp/fuel-case0.json
```

`--case-index` selects capacity cases 0–14; `--warmup` and `--repeats` set the
timing repetitions. Smoke results are not evidence of performance at full
scale.

## Appendix: implementation structure

This section describes internals that affect identities and compiled
artifacts; it is not needed to use the application.

- **Solver coordinates.** `FuelRodFields` order defines the solver coordinate
  layout. Unknowns are physical fields divided by fixed scales (temperature,
  contact pressure, gas pressure, friction force, free gas, coolant heat flux,
  channel enthalpy); the joint endpoint appends fuel-site hydrostatic stresses.
  The normalized spaces carry a computational coordinate identity (field and
  site roles, packing, finite-element form and topology roles, integer routes,
  dtype) that is distinct from the physical rod identity and enters prepared
  and runtime fingerprints.
- **Step loading.** One loading record per accepted state, time step, and
  bundle holds the interpolated conditions, step-averaged power and flux, the
  radial power advance, species production, and exposure. It is shared by the
  gas block, the thermomechanical block, and endpoint certification. Burnup
  differences are taken from the converted endpoint epoch so that a
  zero-fission cooldown has exactly zero exposure change.
- **Local roots.** Fuel and cladding constitutive roots return their complete
  terminal stress, creep, and plastic state, so candidate history uses the
  root point without re-evaluation. Finite-strain kinematics are prepared once
  per material point. Implicit differentiation uses the residual only.
- **Derivatives.** The joint endpoint derivative is one prepared
  linearization of the coupled residual at the converged point. Layered rods
  use banded and bordered solves; RZ rods use matrix-free block and Schur
  solves. Replay attaches `phydrax.nonlinear.implicit_root_output` to the
  converged primal, so automatic differentiation of replay and likelihood
  observables uses the coupled endpoint map without solving again.
- **Retries.** The production runtime evaluates the retry attempts of one step
  in a single scanned body; `attempt_executed` marks the attempts that ran, and
  unused capacity is not counted as an attempt.
- **Shared execution.** `FuelNumericalBinding` (from
  `FuelNumericalExecution.bind`) keeps a rod's numerical and bundle arrays as
  dynamic leaves; only array-free metadata is static. Result restoration
  checks the array layout before attaching the rod's identities.
- **Site state.** Fuel integration sites store `mechanics` (creep history,
  eigenstrains, `hydrostatic_stress`, Cauchy stress), `gas` (total and tracer
  gas state and transfers), and `helium` records; cladding sites store creep,
  plastic strain, fluence, oxide, and hydrogen.
- **Source slots.** `FuelSourceSlot` lists the fuel-site, cladding-site, and
  rod slots in evidence order; the `supported` array of `FuelSourceSupport`
  follows that order on its last axis.
- **Modules.** Gas code is split by topic: `_gas_state` (state records),
  `_intragranular`, `_grain_boundary`, `_hbs`, `_fission_gas` (total-gas
  model), `_helium`, and `_gas_block` (per-site block). Cladding laws are in
  `_cladding_properties`, `_cladding_deformation`, and `_cladding_surface`.
  `_geometry` builds layered and RZ bodies and traces with separate builders.
