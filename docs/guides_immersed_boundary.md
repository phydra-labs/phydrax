# Immersed-boundary coupling

Phydrax couples fixed-topology Lagrangian markers to uniform, unit-density MAC flow through one material-measure adjoint interpolation/spreading operator. The qualified marker assignment is the cubic tensor B-spline: four routes per axis, fixed route indices inside one routing program, nonnegative weights, partition of unity, and affine reproduction.

## Marker measure and force convention

`LagrangianMarkerSetPlan` owns stable marker IDs, reference positions, a static active mask, and positive material quadrature weights. Current positions and velocities are temporal state. Changing marker activity or cardinality is a topology event and requires re-preparation.

`PreparedMACMarkerTransfer.interpolation_operator(relation)` is J. Its Hilbert adjoint is S:

```text
S = W_E⁻¹ Jᵀ W_L
```

`W_E` is the MAC face-dual measure and `W_L` is marker material quadrature. `spread(relation, value)` accepts force density per marker measure; callers must not multiply marker quadrature first. Diagnostics expose partition, first-moment and gradient-sum defects, force, torque, and virtual-work residuals.

Route indices and masks are nondifferentiable. With piecewise geometry differentiation, weights remain differentiable while one routing program is fixed. A converged moving-geometry solve is accepted only when rebuilding its canonical routes yields the same indices and masks.

## Exact fixed and prescribed coupling

`MACImmersedBoundaryProjectionPlan` enforces outer-boundary closure, incompressibility, and prescribed marker velocity in one pressure-plus-marker solve. The public physical constraint is

```text
J u = U_b.
```

The returned marker multiplier is force density exerted by the body on the fluid. The fluid receives its spread; a body receives the exact negative marker-measure adjoint load. Pressure and marker constraints are solved iteratively; pressure transform solves may be used by ordinary pressure projection but are not exact coupled-IB solvers.

`MACImmersedBoundaryIMEXEulerMethod` evaluates prescribed geometry at the attempted time and applies a coefficient of dt. `MACImmersedBoundarySBDF2Method` uses exact IMEX startup and a coefficient of 2 dt / 3 on subsequent steps. Both commit pressure, multiplier, and fluid history atomically or retain the previous accepted state.

## Penalty CFD–DEM

`MACPenaltyIBCFDEMCouplingPlan` remains an explicitly approximate penalty family. It shares the same marker transfer but computes

```text
slip = J u − U_body
fluid force density = −penalty × slip
```

The body receives the opposite quadrature-integrated force and torque. Numerical validity and slip qualification are reported separately. `IBPenaltyPlan(require_slip_for_acceptance=True)` preserves strict acceptance by default. DEM contact subcycling remains confined to this penalty family.

## Free rigid bodies

`RigidMarkerMapPlan` binds body-frame markers to `PreparedRigidBodySet`. It rotates marker arms with SO(2) or SO(3), constructs rigid marker velocity, and exposes the paired generalized force/torque pullback. `MACRigidImmersedEulerMethod` performs a contact-free synchronized fluid/body velocity solve at one predicted accepted pose. It is separate from DEM subcycling.

## Deformable structures

`FiniteElementImmersedMarkerMapPlan` binds a fixed FE interpolation H and its material-measure adjoint H*. `MACDeformableImmersedBackwardEulerMethod` combines a `SecondOrderDifferentialSystem`, structural configuration update, fluid momentum, pressure, and marker no-slip in one nonlinear accepted step. Its energy ledger reports fluid kinetic energy, supplied structural energy, coupling powers, and their residual. Area or volume conservation is not claimed unless the selected structural model contains that constraint.

## Failure and differentiation

All exact paths fail closed on nonfinite state, truncated support, failed linear/nonlinear solves, divergence, gauge, slip, KKT residual, or route inconsistency. Mathematical solve differentiation is certified only for a successful primal/adjoint solve with fixed marker activity and routing. `rhs-only` differentiation intentionally freezes operator geometry.

## Current boundaries

The qualified scope is uniform 2-D/3-D, unit-density MAC flow with fixed marker topology. Variable-density coupling, mapped/ALE grids, AMR, distributed marker ownership, remeshing, contact/lubrication extensions, fluctuating hydrodynamics, divergence-free interpolation, and sharp embedded-boundary changes are outside this contract.
