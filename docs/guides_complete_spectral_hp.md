# Complete spectral and hp elements

Phydrax extends one finite-element architecture across adaptive tensor hp,
compatible tensor complexes, simplex/hybrid references, robust DGSEM, moving
meshes, nonlinear solvers, CAD/unfitted geometry, differentiable schedules, and
high-order persistence. These capabilities remain ordinary `CellMesh`,
`FiniteElementDiscretization`, `FiniteElementForm`, workset, constraint, operator,
and solver values.

## Native hp compilation

`compile_finite_element_problem` accepts either a prepared finite-element
discretization or `FiniteElementHPEpoch`. Epoch master-trace constraints become
homogeneous finite-element constraints automatically. Multi-field epochs use
`prepare_multi_field_finite_element_hp_epoch`, including fieldwise conformity,
component shape, and degree offsets.

Physical transfer uses quadrature, target mass, coupling, and declared metric
Jacobian through `physical_mass_projection`. Persistent semantic caches are keyed
by scientific identities rather than Python object identity.

## General h, p, and g adaptation

`AnisotropicHPattern` describes directional tensor splits.
`refine_anisotropic_hp_cells` activates the corresponding child paths while
preserving stable root/path identities. Forests can grow transactionally with
`resize_hp_forest` and reorder deterministically with `compact_hp_forest`.

`GeometryOrderAdaptation` transfers coordinate fields between geometry orders and
reports curvature evidence. `NIrregularMortarPlan` groups an arbitrary supported
set of child patches under one coarse trace.

## Robust conservation dynamics

Tensor GLL DGSEM supports explicit physical/periodic facet ownership, arbitrary-
normal numerical fluxes, diagonal mass inversion, state and weak-residual
linearizations, conservative SSP-stage entropy/positivity filtering, and one
two-pass LDG compressible Navier–Stokes path. Certificates continue to distinguish
periodic inviscid entropy evidence from uncertified physical-boundary and viscous
extensions.

`NodalDGConservationMethodPlan` supplies exact-mass weak DG on stable simplex
references, mixed triangle/quadrilateral mortars, curved coordinate fields,
tetrahedra, prisms, and rational linear pyramids. Nonpolynomial overintegration is
recorded as heuristic quadrature evidence.

Entropy-mortar defects can be derived from left/right states, entropy variables,
numerical flux, and entropy potentials through `derived_mortar_entropy_defect`;
callers do not need to assert a defect manually.

## Compatible tensor spaces

`TensorDeRhamComplex` constructs the exact algebraic sequence

```text
H1 → H(curl) → H(div) → L2
```

with explicit gradient, curl, and divergence matrices. `TensorPiolaMap` provides
covariant and contravariant physical mappings. `TensorDeRhamTransferPlan`,
`CompatibleTraceConstraint`, `CompatibleMortarPlan`, and
`CompatibleAuxiliaryMultigrid` retain commuting and trace roles across p changes.

## Simplex and hybrid references

`SimplexNodalFamily` supplies Modepy warp-and-blend triangle/tetrahedron nodes and
orthonormal modal tabulation. `HybridReferenceFamily` supplies
triangle-times-interval prism bases and a rational degree-one pyramid basis.
Mixed three-dimensional faces use canonical polyhedral connectivity and
two-sided conservative interface routes.

These references remain ordinary `CellMesh` and finite-element compiler values.

## Advanced solvers

The solver layer includes:

- matrix-free Newton linearization with Phydrax linear solves;
- nonlinear local condensation;
- nonlinear FAS cycles;
- restricted additive and multiplicative Schwarz;
- BDDC/FETI-DP trace coarse plans;
- adaptive eigenspace transfer and goal indicators.

`FiniteElementHPMultigridPreconditionerBuilder` exposes hp hierarchy preparation
through the normal preconditioner lifecycle.

## CAD and unfitted geometry

`LevelSetCutQuadrature` and `UnfittedAggregationPlan` handle embedded domains and
small-cut-cell aggregation. `ConservativeMovingInterfaceTransfer` applies a
physical-mass projection between interface spaces.

## Differentiable hp workflows

`FrozenHPAdjointSchedule` composes accepted transfers forward and raw dual
pullbacks backward. `RelaxedHPMarking` provides differentiable weights and a
separate deterministic safety projection. `MeshVaryingUQAggregator` projects
sample-specific meshes onto one reference event space before moments are formed.

Discrete topology promotion remains a derivative boundary unless the caller
explicitly chooses a relaxed approximate policy.

## Performance and persistence

- `PersistentSemanticCache` stores arrays and metadata under semantic IDs.
- `FusedMortarAction` and `FusedTensorTransfer` preserve transpose structure.
- `HPMixedPrecisionPolicy` prevents accidental precision narrowing.
- `HPWorksetMemoryPlan` computes a bounded workset capacity.
- adaptive VTK, XDMF, forest output, and full-runtime checkpoints preserve identities;
- the meshio finite-element importer preserves high-order coordinate DOFs.

## Current boundary

The distributed lowerer provides cost-aware ownership and explicit owned-local,
halo-update, interface, and contribution-sum phases. JAX named-axis collectives
are available; an MPI runtime and dynamic migration engine remain external.
Pyramid approximation is currently degree one, and mixed-cell viscous interfaces
remain outside the certified implementation.
