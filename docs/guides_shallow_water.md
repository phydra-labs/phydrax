# Shallow water

Phydrax implements one- and two-dimensional Saint-Venant flow with static,
upward-positive bathymetry on Cartesian structured finite volumes. The authoritative
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

Bathymetry is required when this interface is selected and rejected with ordinary
numerical fluxes. The initial execution contract supports static Cartesian structured
finite volumes only. Mapped, triangle, unstructured, moving, SBP, spectral, and DGSEM
bathymetric execution fail before JIT.

## Reconstruction

Piecewise-constant reconstruction is the monotone fallback. `MUSCLReconstruction`
reconstructs free surface and discharge while retaining cellwise bed traces. Faces
touching a dry cell fall back to the piecewise-constant equilibrium representation.
This preserves dry lake states and gives second-order accuracy for smooth fully wet
solutions on a flat or cellwise bed.

WENO and characteristic reconstruction are not currently supported by the balanced
method. `ShallowWaterSystem` intentionally does not advertise a Roe eigensystem, so a
generic Roe method is not a dry-safe shallow-water route.

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

This is transient rotating-flow support, not a geostrophically well-balanced spatial
scheme. Exact preservation of pressure--Coriolis equilibria is outside the current
contract.

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
bed = jnp.zeros(shape)
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

- one- and two-dimensional Cartesian structured finite volumes;
- periodic and ordinary state-producing physical boundaries;
- piecewise-constant and equilibrium-aware MUSCL reconstruction;
- wet and exactly dry cells;
- static discontinuous bathymetry;
- SSPRK3 stage positivity and retry;
- f-plane and beta-plane Coriolis sources;
- branchwise JAX JVP/VJP away from wet/dry and limiter switching surfaces.

Not supported: moving or mapped beds, unstructured bathymetry, prescribed-normal-flux
boundaries for the balanced method, WENO balance, geostrophic well-balancing,
characteristic open boundaries, reduced precision below float32, sediment, multilayer
systems, or differentiability at shoreline topology changes.

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
