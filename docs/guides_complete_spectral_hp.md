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

## Robust DGSEM

The robust physics layer provides:

- characteristic inflow, outflow, slip-wall, and no-slip-wall states;
- entropy wall evidence;
- source/flux well-balance ledgers;
- BR1 viscous gradients, divergence, and mortar fluxes;
- action-aware overintegration;
- entropy, kinetic-energy, and skew split forms;
- troubled-cell evidence;
- conservative modal filtering;
- density/pressure positivity limiting;
- DG/subcell finite-volume projection, advancement, and reconstruction.

Entropy-mortar defects can be derived from left/right states, entropy variables,
numerical flux, and entropy potentials through
`derived_mortar_entropy_defect`; callers do not need to assert a defect manually.

## Moving meshes and temporal hp

`ALEMetricState` records coordinates, mesh velocity, Jacobian evolution, and the
temporal GCL defect. `MovingMortarMetricPlan` advances mortar geometry.
`LocalTimeSteppingPlan` supplies level steps and conservative refluxing.
`TemporalHPBudget` separates spatial, temporal, and algebraic error contributions.

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

`SimplexModalFamily` supports high-order triangles and tetrahedra.
`SimplexSBPPlan` constructs nodal derivative actions. `HybridReferenceFamily`
provides prism and pyramid nodes, while `HybridRefinementPlan` and
`HybridMortarPlan` provide child maps and polynomial interface reproduction.

These reference contracts do not create a second mesh runtime. Mesh-level hybrid
connectivity continues through the canonical `CellMesh` extension path.

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

`NURBSPatch` evaluates rational B-spline geometry with Cox–de Boor basis
recursion. `MultipatchContinuityPlan` constructs shared coordinates.
`TrimmedCADQuadrature` masks parameter quadrature through trim functions.
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
- `HeterogeneousSignatureSchedule` groups equal scientific signatures.
- `FusedMortarAction` and `FusedTensorTransfer` preserve transpose structure.
- `HPMixedPrecisionPolicy` prevents accidental precision narrowing.
- `HPWorksetMemoryPlan` computes a bounded workset capacity.
- adaptive VTK, XDMF, and forest output preserve epoch information;
- Gmsh and Exodus-array adapters build canonical meshes.

## Current boundary

This implementation intentionally excludes an actual multi-host communication
runtime, MPI partitioner, dynamic migration engine, and distributed failure
recovery. Partition-independent plans and serial restart data remain available,
but communication is supplied by external runtime work.
