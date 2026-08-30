# Fixed-mesh FEM solver acceleration

This guide covers linear-history initial guesses, exact FEM diagonals,
p-transfer, p-multigrid, collocated tensor kernels, staged DG traces,
quadrature policies, one-ring Schwarz, and low-order auxiliary operators.

The algorithms are Phydrax-native. The design was informed by
[libParanumal](https://github.com/paranumal/libparanumal), its matrix-free
[p-multigrid implementation](https://github.com/paranumal/libparanumal/blob/main/solvers/elliptic/src/ellipticPreconMultiGrid.cpp),
[multigrid smoothers](https://github.com/paranumal/libparanumal/blob/main/solvers/elliptic/src/ellipticPreconMultiGridLevel.cpp),
[overlapping Schwarz preconditioner](https://github.com/paranumal/libparanumal/blob/main/solvers/elliptic/src/ellipticPreconOAS.cpp), and
[tensor-product kernels](https://github.com/paranumal/libparanumal/blob/main/solvers/elliptic/okl/ellipticAxQuad2D.okl).

## Linear solve histories

`LinearSolveHistory` stores accepted solution vectors and their operator images.
For projection strategies it chooses coefficients that minimize the represented
RHS residual and reconstructs an initial solution from the paired solution
basis. Histories carry explicit operator-family, constraint, and nullspace
identities. Incompatible histories are rejected rather than reused by shape.

Updates are immutable and require an explicit acceptance decision. Initial
guesses are treated as algorithmic data and stop gradients by default; the
converged solve retains its existing differentiation policy.

Available strategies are:

- zero;
- last accepted solution;
- paired RHS/solution projection;
- fixed-capacity rolling history;
- stabilized polynomial extrapolation from accepted times.

The history design follows ideas described in
[Initial Guesses for Sequences of Linear Systems](https://arxiv.org/abs/2009.10863).

## Exact diagonal data

`CompiledFiniteElementProblem.exact_diagonal()` returns
`FiniteElementDiagonalData` with construction provenance and zero/negative
masks. Structurally sparse affine operators use direct coordinate storage. A
bounded coordinate-linearization fallback is available only when explicitly
enabled with a dimension cap.

No diagonal floor, absolute value, or regularization is applied implicitly.

## p-transfer

`FiniteElementPTransfer` separates:

- primal prolongation;
- raw dual pullback;
- pairing-aware adjoint;
- mass projection.

The generic transfer factory supports matching element cell/conformity families
with increasing degree. Supplied mass matrices are solved through Phydrax dense
factorization to construct a pairing-aware adjoint.

## p-level planning

`FiniteElementPMultigridPolicy` supports:

- every degree;
- approximately half degree;
- approximately half local DOFs;
- explicit decreasing orders.

`finite_element_p_multigrid_plan` validates level operators, smoothers, and
transfer spaces before creating the generic Phydrax multigrid hierarchy. Fine
levels may remain matrix-free; the caller supplies the coarse sparse operator
and preconditioner.

## Collocated tensor path

`CollocatedTensorProductOperator` implements quad/hex mass-diffusion actions
when nodal and quadrature axes coincide. It stores one-dimensional derivatives,
weighted mass data, and packed symmetric metric components. Unsupported or
overintegrated actions should use the generic tensor partial-assembly path.

The execution strategy follows the dataflow studied in
[Acceleration of tensor-product operations for high-order finite element methods](https://arxiv.org/abs/1711.00903),
but is implemented through JAX and `opt_einsum`, not translated OCCA kernels.

## DG derivative and trace staging

`CellDerivativeBatch` evaluates one set of local values/gradients.
`DGTraceBatch` constructs plus/minus `FacetJet` traces from packed side data so
multiple facet actions can share one staged derivative representation within an
operator invocation. Staged state-dependent data must never survive into a
later nonlinear residual evaluation.

## Quadrature accuracy

`QuadratureAccuracyPolicy` distinguishes:

- declared polynomial exactness;
- collocation;
- overintegration;
- explicit rule degree.

Polynomial-exactness mode rejects unknown coefficient or kernel degree.
`QuadratureEvidence` records role, selected degree, exactness, and aliasing
status. Nonpolynomial material laws require explicit integration evidence.

## One-ring Schwarz

`one_ring_patch_plan` builds fixed-mesh cell patches from interior-facet
adjacency. It computes reciprocal overlap weights so valid patch weights form a
partition of unity. `FiniteElementPatchPreconditioner` applies supplied local
inverse matrices and weighted additive scatter. Local matrix construction and
factorization remain explicit preparation responsibilities.

## Low-order auxiliary action

`LowOrderAuxiliaryOperatorPlan` binds high-to-low interpolation,
low-to-high anterpolation, and positive multiplicity weights.
`LowOrderAuxiliaryPreconditioner` transforms the residual, invokes any prepared
low-order Phydrax preconditioner, maps the correction back, and reapplies the
weights. The plan does not create or expose a mesh-generation API.

## Current scope

These paths are fixed-mesh and single-device. They do not add MPI/OGS,
partitioning, OCCA, parAlmond, mesh loaders, or a duplicate Krylov stack.
