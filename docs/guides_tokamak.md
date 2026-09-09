# Axisymmetric tokamak modeling

`phydrax.applications.tokamak` owns the axisymmetric device semantics absent from generic MHD, PIC, circuit, and control substrates. Its name is intentionally narrow: this package does not imply stellarator, inertial-fusion, edge/SOL, gyrokinetic, or disruption support.

## Magnetic convention

The native frame is right-handed cylindrical `(R, φ, Z)` with `R > 0`. Poloidal flux uses Wb/rad and satisfies:

```text
B_R = -(1/R) ∂ψ/∂Z
B_Z =  (1/R) ∂ψ/∂R
Δ*ψ = -μ₀ R Jφ
```

`TokamakMagneticConvention` records all signs, handedness, flux normalization, and an optional COCOS label. `TokamakConventionTransform` applies signs and the complete 2π conversion together. An adapter never infers a convention from data signs.

## Equilibrium interchange

`tokamak.interchange.import_eqdsk` reads a bounded, checksummed G-EQDSK resource. The caller supplies its complete source convention. The parser rejects malformed counts, unsupported trailing numeric fields, nonfinite arrays, and boundaries outside the grid.

The scoped IMAS boundary imports or exports exactly one equilibrium time-slice mapping. It does not initialize an IMAS database or actor. The installed IMAS release remains responsible for mapping its live IDS object to the exact supported record.

## Flux surfaces

`FluxSurfacePlan` prepares nested star-shaped surfaces by fixed angular rays. It freezes contour routes and produces enclosed volumes, face areas, major/minor radii, safety factor, and a conservative `MetricLinePlan`.

Initial support ends strictly inside the LCFS. Islands, stochastic regions, multiple magnetic axes, and X-point topology derivatives are unsupported. Prepared geometry is differentiable only after its discrete topology has been fixed.

## Core transport

`TokamakCoreTransportPlan` evolves:

- electron particle density;
- electron thermal energy;
- ion thermal energy.

The closure supplies integrated face conductances. This avoids hiding a physical-length assumption in normalized flux coordinates. Backward Euler is solved with the native tridiagonal-line solver. Results retain candidate and accepted states, solver evidence, and particle/energy closure. Negative states or failed solves roll back atomically.

Current diffusion is a separate conservative poloidal-flux operator. `PreparedTokamakTransportCurrentCoupling` requires the same flux-surface geometry and commits core/current state only when both participants succeed.

## Native equilibrium and circuits

`FixedBoundaryGradShafranovPlan` solves the prescribed-toroidal-current forward problem on a uniform R-Z grid with a sparse native linear operator. This is not yet a nonlinear pressure/F-profile reconstruction.

`CoupledInductancePlan` represents reciprocal active/passive winding matrices
and certifies magnetic energy, resistive loss, and backward-Euler numerical
dissipation. `AxisymmetricCoilResponsePlan` accepts governed response arrays or
constructs the regular boundary response of declared circular filament coils
from complete elliptic integrals. `FreeBoundaryTokamakPlan` couples those
currents to the boundary response and a quasi-static fixed-current plasma
solve. Plasma-induced circuit response beyond the admitted mutual-inductance
matrix is not inferred.

## Plant and measurements

`TokamakCorePlantPlan` exposes core transport through the transactional `AbstractDiscretePlant` runtime. Actuator bounds are enforced before commit. `TokamakShotRecord` composes governed `MeasurementAsset` values and requires one exact shot/time identity; `TokamakShotSplit` prevents a shot from crossing training and locked-evaluation roles.

All tokamak capability profiles are unreleased candidates. Synthetic examples establish numerical behavior only, not predictive machine accuracy or operational control readiness.
