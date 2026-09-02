# Shallow water

Phydrax implements one- and two-dimensional Saint-Venant flow with
upward-positive bathymetry on structured and mapped finite volumes. The authoritative
cell state is `(h, hu[, hv])`: water depth and horizontal discharge. Bed elevation
`b` is a prepared auxiliary field, and free-surface elevation is `eta = h + b`.

## Physical and dry-state contract

`ShallowWaterSystem` admits finite states with `h >= 0`. An exactly dry state has
`h = 0` and zero discharge. Primitive conversion, physical fluxes, and signal bounds
define dry velocity as zero and never divide by zero.

`ShallowWaterWetDryPolicy` separates numerical wetness and velocity regularization
from mathematical admissibility. Its small positive defaults suppress ill-conditioned
near-shore velocities; applications with dimensional depths should set physically
resolved thresholds explicitly. The policy does not add a thin film or clip accepted
cell averages.

The bed-inclusive mechanical-energy density reported by shallow-water observables is

`E = |m|^2 / (2 h) + g h^2 / 2 + g h b`,

with zero kinetic energy at the exact dry state.

## Well-balanced wet/dry interface method

`ShallowWaterHydrostaticHLLPlan` couples all pieces of the bed-step model:

1. Chen--Noelle subcell hydrostatic reconstruction;
2. dry-safe shallow-water HLL signal estimates;
3. one shared conservative transport flux;
4. separate owner and neighbour hydrostatic momentum corrections;
5. an exact dry/dry zero-mass-flux route.

The corrections have zero depth component. They are face-owned contributions, not a
cell-centered approximation to `-g h grad(b)`. This pairing preserves wet and partially
dry lakes at rest over discontinuous cellwise beds.

`ShallowWaterBathymetryPlan` requires exactly one explicit representation:
`cell_values=` or a static-physical-bed `evaluator=` sampled on stage coordinates.
Prepared beds are bound to geometry and precision identities. The hydrostatic HLL
operator supports Cartesian axes and arbitrary unit normals; its ALE route uses
`F·n - w_n U` and the same one-sided hydrostatic corrections. Missing/stale bed,
metric, or seam evidence fails before execution.

## Reconstruction

Piecewise-constant reconstruction is the monotone fallback. `MUSCLReconstruction`
reconstructs free surface and discharge while retaining cellwise bed traces. Faces
touching a dry cell fall back to the piecewise-constant equilibrium representation.
This preserves dry lake states and gives second-order accuracy for smooth fully wet
solutions on a flat or cellwise bed.

`ShallowWaterEquilibriumWENOZPlan` reconstructs free surface and discharge with
fifth-order WENO-Z weights while carrying bed traces separately. The optional normal
characteristic route records `characteristic_used`, eigenbasis conditioning, and the
explicit `equilibrium-componentwise` fallback for dry stencils. `ShallowWaterSystem`
still does not advertise a generic Roe capability.

## Stage positivity and time stepping

Wet/dry shallow water must use `PreparedFiniteVolumeRuntime` with
`FluxPositivityPlan`. Every SSPRK forward-Euler substep evaluates both the selected
reconstruction and the piecewise-constant hydrostatic fallback. One face factor blends
the shared flux and both bed corrections. A secondary global reduction closes any
remaining admissibility defect.

Direct `UnsplitFiniteVolumeSSPRK3Plan` and directional splitting reject the balanced
method because they do not own this stage-positivity contract. Runtime evidence records
fallback validity, activation factors, accepted shared fluxes, and accepted one-sided
bed-correction integrals.

## Coriolis forcing

`ShallowWaterCoriolisSource(f0, beta=...)` provides f-plane or beta-plane forcing for
two-dimensional shallow water. It is passed through `ConservationProblemIR` with its
canonical `source_id`. The source contributes no mass, and its SSPRK stability bound is
included in stable-step selection.

`GeostrophicBalancePlan` validates a user-declared reference with the same discrete
residual used by execution and stores its geometry-bound residual. The prepared
deviation operator evaluates `R(U) - R(U_eq)`, so a certified f-/beta-plane reference
is preserved exactly; it does not discover arbitrary equilibria.

`ShallowWaterNormalDischargeBoundary` supplies a complete hydrostatic trace and one
accepted outward mass flux. `ShallowWaterCharacteristicOpenBoundary` combines outgoing
interior and incoming exterior Riemann invariants, distinguishes sub/supercritical
flow, and rejects the near-critical ambiguous route.

## Minimal construction

```python
import jax.numpy as jnp
import phydrax as phx

shape = (128,)
grid = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(shape[0], periodic=True),),
    axis_names=("x",),
).prepare(jnp.asarray(((0.0,), (1.0,))))

system = phx.equations.ShallowWaterSystem()
discretization = phx.discretization.FiniteVolumePlan(
    grid, component_names=system.component_names
).prepare()
method = phx.discretization.FiniteVolumeMethodPlan(
    phx.discretization.MUSCLReconstruction(),
    phx.discretization.ShallowWaterHydrostaticHLLPlan(),
)
boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
problem = phx.equations.ConservationProblemIR(
    "wet-dry", "state", system, boundaries
)
bed = phx.discretization.ShallowWaterBathymetryPlan(
    cell_values=jnp.zeros(shape), field_id="flat-bed"
)
compiled = phx.equations.compile_conservation_problem(
    problem, discretization, method, bathymetry=bed
)
runtime = phx.solver.PreparedFiniteVolumeRuntime(
    compiled.dynamics,
    phx.discretization.FluxPositivityPlan(),
)
```

`compiled.dynamics.shallow_water_observables(state)` derives named depth, bed,
free-surface elevation, momentum, velocity, wet mask, and energy without changing the
authoritative state. `FiniteVolumeOutputPlan.write_snapshot(..., shallow_water=view)`
can store these fields beside the ordinary finite-volume checkpoint data.

## Supported capability

- one- and two-dimensional structured and mapped finite volumes;
- explicit cell-value or static-physical evaluator bathymetry;
- axis and arbitrary-normal/ALE hydrostatic HLL balance;
- piecewise-constant, equilibrium MUSCL, and equilibrium WENO-Z reconstruction;
- prescribed-normal-discharge and characteristic open trace policies;
- declared geostrophic reference preservation;
- wet and exactly dry cells with SSPRK stage positivity/retry;
- fixed-capacity multilayer hydrostatics and single-fraction MPM/Exner bedload physics;
- fixed-mask derivatives plus isolated shoreline saltation evidence;
- LPP-resolved float16/bfloat16 storage with float32-or-higher decisions/reductions;
- branchwise JAX JVP/VJP away from wet/dry, limiter, threshold, and event switches.
`PreparedBalancedShallowWaterLowering` supplies the common equilibrium split
`-div(m)` and `-div(m tensor u) - g h grad(eta)` to triangle/unstructured, SBP,
global-spectral, and DGSEM derivative owners. The four named lowering functions bind
the same prepared bathymetry/geometry identity, so constant surface and zero discharge
annihilate under each backend's own discrete derivative.


`HydrostaticLayerCoupling` certifies stable density ordering and a positive-semidefinite
hydrostatic energy Hessian. `MultilayerShallowWaterSystem` uses fixed layer capacity.
`BedloadSedimentPlan` and `ShallowWaterExnerSystem` implement the named noncohesive,
single-fraction Meyer--Peter--Muller/Exner route; suspended, cohesive, and multigrain
transport are not claimed.

Sub-float32 finite-volume storage requires an exact provider `PrecisionResolution`.
Flux/decision/reduction roles default to float32, and `quantize_and_validate` rechecks
finite/admissible/wet-mask evidence after the storage round trip. FP8/MX remains
fail-closed unless the LPP provider certifies the requested operations and format.

## Qualification and benchmarks

Machine-readable lake, dry-dam-break, and smooth-convergence checks are available
through:

```text
python tools/shallow_water_qualification.py --case lake
python tools/shallow_water_qualification.py --case dam-break
python tools/shallow_water_qualification.py --case convergence
```

`tools/shallow_water_benchmarks.py` separates JIT compilation from steady residual
throughput for piecewise-constant/MUSCL and fully wet/wet-dry regimes.

The balanced construction is independently implemented from the methods described by
Audusse et al., *A Fast and Stable Well-Balanced Scheme with Hydrostatic
Reconstruction for Shallow Water Flows*,
[doi:10.1137/S1064827503431090](https://doi.org/10.1137/S1064827503431090), and
Chen and Noelle, *A New Hydrostatic Reconstruction Scheme Based on Subcell
Reconstructions*,
[doi:10.1137/15M1053074](https://doi.org/10.1137/15M1053074).
