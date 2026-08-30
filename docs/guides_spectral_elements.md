# Spectral elements

Phydrax spectral elements are high-order finite elements. They use the existing
`CellMesh` → `FiniteElementPlan` → `FiniteElementForm` → `WorksetProgram` compiler
rather than a second mesh, field, equation, or solver stack. Global Fourier,
Chebyshev, Legendre, and spherical spaces remain in the separate global spectral
API.

## Reference approximation

`ReferenceNodalFamily` prepares anisotropic tensor-product nodal families on
quadrilaterals and hexahedra. Gauss--Lobatto--Legendre nodes and weights come from
the canonical polynomial substrate. `PreparedFiniteElementReference` binds one
`FiniteElementSpec` to explicit volume and facet rules, requested reference
actions, precision, dense tabulation, traces, and optional tensor factors.

A field's coefficient representation is independent of conformity. Continuous
and discontinuous nodal Lagrange fields both use `point_value`; modal and moment
families retain their own representations. Geometry elements and field elements
remain independent.

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
and interface-dissipation identities. A complete entropy-stability claim also
records boundary, source, and viscous evidence. The initial conservation compiler
scope is periodic stationary mapped quadrilateral or hexahedral DGSEM; it does
not claim positivity preservation, shock limiting, ALE GCL, or viscous entropy
stability.

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

Distributed plans use explicit owned/halo worksets, dependency/completion
identities, exactly-once facet ownership, deterministic reductions, and
partition-independent global pairings. The current backend is JAX named-axis and
backend-neutral plan execution; no MPI runtime is claimed.

## Solvers and analysis

Spectral elements emit ordinary Phydrax spaces and operators. Linear and nonlinear
solves, block preconditioners, multigrid, DAE and second-order integration,
generalized eigenproblems, Ritz extraction, recycling, continuation, results,
accepted-step schedules, and checkpoints remain in `phydrax.linalg` and
`phydrax.solver`.

## Current limits

- Tensor reference actions cover quadrilaterals and hexahedra; simplex tensor-SBP
  execution is separate future work.
- High-order compatible H(div)/H(curl) tensor families are not provided.
- DGSEM is periodic and stationary-mesh only.
- Mortar entropy compatibility is not inferred from ordinary L2 projection.
- Distributed plans do not constitute an MPI mesh partitioner or communication
  runtime.
