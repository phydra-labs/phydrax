# Spectral elements

Phydrax spectral elements are high-order finite elements. They use the existing
`CellMesh` → `FiniteElementPlan` → `FiniteElementForm` → `WorksetProgram` compiler
rather than a second mesh, field, equation, or solver stack. Global Fourier,
Chebyshev, Legendre, and spherical spaces remain in the separate global spectral
API.

## Reference approximation

`ReferenceNodalFamily` prepares anisotropic tensor-product nodal families on
quadrilaterals and hexahedra. `SimplexNodalFamily` uses Modepy warp-and-blend
nodes and orthonormal triangle/tetrahedron modes. `HybridReferenceFamily`
provides triangle-times-interval prism bases and the rational linear pyramid.
`PreparedFiniteElementReference` binds each supported cell to explicit volume
and facet rules, requested actions, precision, dense tabulation, traces, and
optional tensor factors.

A field's coefficient representation is independent of conformity. Continuous
and discontinuous nodal Lagrange fields both use `point_value`; modal and moment
families retain their own representations. Geometry and field elements remain
independent.

## Quadrature and mass

Quadrature is explicit. The supported accuracy vocabulary distinguishes exact
polynomial rules, collocation, overintegration, and caller-supplied rules. An
n-point Gauss rule is exact through degree 2n−1. An n-point GLL rule is exact
through degree 2n−3. A p+1 GLL collocation rule therefore gives a diagonal
quadrature-induced mass, not the generic exactly integrated Qp mass.

`FiniteElementMassPolicy` distinguishes exact, collocated-diagonal, and row-sum
lumped mass. `CompiledFiniteElementProblem.weak_residual` always returns the weak
dual residual. `mass_inverted_rate` exists only through an explicitly prepared
mass policy; general temporal problems may instead use the DAE adapter.

## Execution

`FiniteElementExecutionPolicy` has independent axes:

```text
realization = matrix_free | sparse
local_kernel = auto | dense | partial | sum_factorized | collocated
accumulation = fast | deterministic | compensated
```

Dense execution is the numerical oracle. Sum-factorized and collocated actions
consume the same form, reference action, coefficient binding, runtime geometry,
and workset routes. Quad and hex cell and facet paths preserve exact transpose,
JVP, and VJP actions. Runtime coefficient values are excluded from kernel keys;
their support, entity, field, rule, side, shape, and axis layout remain part of
the compilation identity.

## Geometry and facets

Runtime metric data contain physical points, Jacobians, cofactors, weighted
measures, and mapped gradients. Tensor execution does not materialize one full
physical-gradient tensor per cell. Prepared facet metrics bind physical trace
points, surface measures, scaled normals, and explicit owner/neighbour trace
permutations.

High-order conforming H1 routes share vertex, oriented-edge, oriented-face, and
cell-interior coordinates. DG fields remain cell-local. One interface flux is
evaluated per physical face and scattered to both cells with opposite
orientation. Hexahedral faces support the complete quadrilateral rotation and
reflection group.

## DGSEM and entropy evidence

`TensorGLLSBPPlan` verifies positive norm weights, zero derivative of constants,
boundary extraction, and the element SBP identity. `MappedTensorMetricPlan`
prepares stationary mapped quad/hex metric terms and rejects failed determinant,
metric-identity, free-stream, watertight-face, or opposite-normal evidence.

`DGSEMConservationMethodPlan` combines:

- one symmetric consistent physical two-point volume flux;
- one typed arbitrary-normal interface flux;
- tensor GLL SBP data;
- compatible mapped metrics;
- explicit diagonal GLL mass inversion;
- deterministic, compensated, or fast accumulation.

`certify_dgsem_flux_compatibility` is separate from `ConvexEntropyPair`. It checks
the concrete physical fluxes against symmetry, consistency, entropy-potential,
and interface-dissipation identities. Physical boundary patches are explicit
and exhaustive. A stateful modal entropy/positivity filter can run after every
SSPRK stage while conserving geometry-weighted means. Compressible
Navier–Stokes uses one two-pass LDG path; viscous and physical-boundary entropy
status remains explicitly uncertified.


## General nodal DG

`NodalDGConservationMethodPlan` lowers ordinary weak-form DG through the same FE
action compiler. Exact cell-local mass matrices use Phydrax Cholesky
factorizations. Role-specific quadrature evidence marks nonpolynomial
overintegration as heuristic rather than exact. Supported executions include
triangles, quadrilaterals, tetrahedra, prisms, rational linear pyramids, curved
coordinate fields, mixed triangle/quad mortars, and conforming mixed
three-dimensional polyhedral interfaces.

## Transfers, multigrid, mortars, and distribution

Finite-element p-transfer keeps four operations distinct: primal prolongation,
raw dual pullback, pairing adjoint, and physical mass projection. P-level plans
feed the existing `phydrax.linalg` multigrid lifecycle. Weighted one-ring Schwarz,
tensor fast diagonalization, and low-order auxiliary correction are ordinary
Phydrax preconditioner builders.

`FiniteElementMortarPlan` is a two-sided interface object. It records left/right
interpolation, raw dual maps, pairing adjoints, physical mass projections,
orientation, quadrature, constant and polynomial reproduction, geometry
compatibility, and conservative integrated-flux evidence. Fixed-capacity hp
transactions retain accepted topology, lineage, degree tuples, deterministic
workset buckets, transfers, and rollback semantics.

Distributed plans use cost-aware shape/order estimates, explicit owned/halo
worksets, dependency/completion identities, exactly-once facet ownership,
deterministic reductions, and partition-independent global pairings. The
execution lowerer exposes owned-local, halo-update, interface, and
contribution-sum phases. JAX named-axis collectives remain the current backend;
no MPI runtime is claimed.

## Solvers and analysis

Spectral elements emit ordinary Phydrax spaces and operators. SSPRK stage
transforms, exact-time target schedules, portable full-runtime checkpoints,
streaming observables, accepted-step triggers, and bounded asynchronous
publication compose with existing linear, nonlinear, DAE, continuation, and
result substrates.

## Adaptive tensor hp epochs

`FiniteElementHPTopology` is a fixed-capacity refinement forest. Allocated
inactive parents retain their stable `(root, path)` identity, geometry, child
routes, and refinement depth, while `active` selects the current leaves.
`initial_finite_element_hp_topology`, `refine_tensor_hp_cells`, and
`coarsen_tensor_hp_cells` implement isotropic quadrilateral/hexahedral h changes;
per-axis p remains anisotropic. `balanced_hp_refinement_ids` closes requested
refinement under the 2:1 face rule before a candidate is built.

`FiniteElementHPEpoch` binds one active `CellMesh`, forest topology, inherited
geometry, deterministic degree buckets, nonconforming interface overlay, prepared
finite-element discretization, and trace constraints. Active cells are grouped by
degree tuple, so existing reference actions and kernels remain authoritative.
Curved children evaluate the accepted parent coordinate map rather than fitting
independent faces.

For H1 fields, a canonical master trace constrains p- and h-nonconforming cell
traces. For L2/DG fields, coarse-to-fine patches lower through asymmetric mortar
worksets with independent owner and neighbour widths. Hanging interfaces are
removed from the physical exterior domain.

## Adaptive decisions and transactions

`FiniteElementHPResidualJumpLedger`, `tensor_modal_decay_estimate`, and
`FiniteElementHPErrorEstimate` keep estimation separate from adaptation policy.
`finite_element_hp_decision` applies degree/depth bounds, active-cell and estimated
DOF budgets, and coarsening hysteresis. Balance-added cells are recorded separately
from the requested set.

Primal interpolation, physical mass projection, raw dual pullback, and pairing
adjoint remain separate transfer roles. `FiniteElementHPTransaction` pairs accepted
and candidate epochs with their lineage and transfers. The solver topology
transaction transfers declared state/history roles, certifies the fully prepared
candidate, and then promotes atomically or returns the accepted state unchanged.
Epoch forest and geometry data have canonical restart adapters.

## Adaptive solvers and distribution

Degree-bucket local elimination extends existing static condensation onto the hp
trace skeleton. `FiniteElementHPMultigridPlan` composes adjacent h/p transfers;
`FiniteElementHPSolverRefreshPlan` distinguishes reusable degree signatures from
route, metric, and skeleton refreshes.

Children inherit parent partition ownership. Adaptive owned/halo worksets and
mortar dependency plans are rebuilt from stable tree identities. This remains a
partition-independent planning contract, not dynamic repartitioning or an MPI
runtime.

## Nonconforming DGSEM

Conservative mortar evidence is not automatically entropy evidence.
`certify_dgsem_mortar_compatibility` separately checks mass compatibility,
constant reproduction, opposite normals, and the supplied entropy defect.
Nonconforming DGSEM flux ledgers reject a mortar without a passing certificate.

## Current limits

- h-refinement is isotropic with a 2:1 face-balance contract; anisotropic h and
  arbitrary n-irregular interfaces are not provided.
- Tensor entropy-certified DGSEM remains quadrilateral/hexahedral; general
  simplex and hybrid nodal DG does not claim the tensor SBP certificate.
- Pyramid approximation is currently the rational degree-one space.
- Nodal LDG viscosity is complete for the physical-boundary tetrahedron path;
  mixed-cell viscous interfaces remain unsupported.
- Mortar entropy compatibility is not inferred from ordinary L2 projection.
- Distributed phases do not constitute dynamic cell migration or an MPI
  runtime.
