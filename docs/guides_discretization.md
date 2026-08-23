# Discretization

Phydrax treats discretization as a composition of finite supports, field spaces,
measures, operators, transfers, and realization records. A discretization is not
synonymous with a mesh or a collection of points.

## Continuum semantics and finite coordinates

`phydrax.domain` remains the source of continuum labels, components, and functions.
`phydrax.geometry` remains the source of analytic, reconstructed, and mesh geometry.
`phydrax.discretization` binds those semantics to finite numerical coordinates.

The main distinctions are:

- `DiscreteTopology`: combinatorial organization without coordinates;
- `DiscreteSupport`: topology plus one geometric embedding identity;
- `DiscreteFieldSpace`: scientific field semantics, exact DOF layout, and linalg
  coordinate space;
- `DiscreteMeasure`: finite physical, probability, counting, or signed measure;
- `AbstractDiscretizationPlan`: symbolic construction and capability contract;
- `AbstractPreparedDiscretization`: prepared geometry, spaces, measures, and
  method-specific numerical state;
- `FieldTransfer`: source-to-target map with explicit conservation, nesting, and
  adjoint claims;
- `DiscretizationBundle`: complete approximation provenance for a computation.

Quadrature points are not assumed to be DOFs. A spectral basis is not assumed to be
its evaluation grid. A point cloud is not fabricated into a cell complex.

## Tensor grids and numerical axes

Numerical axis specifications live in `phydrax.discretization`. The axis declares its
primary entity: `UniformAxisSpec` stores point values, while `UniformCellAxisSpec`
stores interval values. `PreparedTensorGrid.cells()`, `.vertices()`, and `.faces(axis)`
return exact tensor entity layouts with their own shapes, coordinates, measures, and
boundary masks.

```python
import phydrax as phx

plan = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(64),),
    axis_names=("x",),
)
support = plan.prepare(((0.0,), (1.0,)))
dx = phx.discretization.DerivativeRequest(
    "dx",
    support,
    "x",
    derivative_order=1,
    accuracy_order=4,
)
finite_difference = phx.discretization.FiniteDifferencePlan(
    support,
    (dx,),
    field_name="u",
).prepare()
```

Stencil banks have fixed capacity but an explicit validity mask per row. Interior and
closure rows may therefore use different widths without evaluating inactive
coefficients. Derivatives may map between different entity shapes, such as cells to
normal faces, and expose rectangular transpose and pairing-adjoint actions.

`compile_semidiscrete_pde(problem, support)` lowers supported PDE expressions directly
to native finite-difference operators. No spectral axis composer is inserted.
`SeparableSpectralDiscretization` is reserved for Fourier, sine, and cosine axes;
polynomial axes use an explicit collocation method.

For uniform periodic grids, `periodic_finite_difference(support)` prepares the common
first/second-derivative calculus. Uniform bounded FD2 Laplacians can instead be
diagonalized from explicit boundary semantics:

```python
diagonalization = phx.discretization.diagonalize_fd_laplacian(
    support,
    {"x": ("dirichlet", "neumann")},
)
solve = phx.discretization.FDLaplacianSolvePlan(diagonalization)
result = solve.solve(rhs, boundary_values={"x": (lower_value, upper_flux)})
```

The diagonalization chooses the certified FFT/DCT/DST family from point-versus-cell
placement and the lower/upper boundary pair. Neumann nullspaces require an explicit
compatibility and gauge policy.

Every prepared stencil carries independent consistency, adjoint, and applicable
conservation reports. `ManufacturedConvergencePlan` measures total, interior, and
boundary rates separately, so a high interior order cannot hide a failed closure.

Conservative operators retain expression form and entity transitions:

```python
diffusion = phx.discretization.ConservativeDiffusionPlan(
    support,
    boundaries={"x": ("dirichlet", "neumann")},
).prepare(coefficient)

multigrid = phx.discretization.StructuredMultigridPlan(diffusion).prepare()
```

Point-primary reference grids can be embedded with `MappedTensorGridPlan`; discrete
curl metrics certify free streams before physical operators are exposed. Named mapped
or Cartesian blocks compose through `MultiblockGridPlan`, explicit orientations, and
norm-compatible mortar interpolation.

Regular stencil interiors lower to compact offset kernels with closure-only row banks.
This execution optimization does not change the canonical stencil ID. Fixed-capacity
AMR, distributed halo schedules, portable checkpoints, and discrete adjoints all retain
the same field/entity identities.

`GridSampling` remains a domain evaluation request and accepts `TensorGridPlan` or
individual axis specifications. `GridBatch` remains an evaluation batch rather than
a state-space declaration.

## Spectral bases

`ModalTransform` is the canonical weighted analysis/synthesis contract. It carries
physical/modal spaces, stable mode IDs, measure, normalization, and transform
identity. `OperatorSpectrum` separately binds one operator's modal values,
nullspace, degeneracy groups, and approximation provenance. A
`SpectralDecomposition` is their convenience pairing:

```python
decomposition = phx.discretization.SpectralDecomposition.from_eigenpairs(
    eigenvalues,
    eigenfunctions,
    measure,
    decomposition_id="surface-laplacian",
)
space = phx.discretization.SpectralDiscretization(decomposition)
```

One transform may support multiple operator spectra, including distinct
finite-difference and pseudospectral Laplacian symbols.
The same decomposition feeds graph/manifold kernels, method-of-lines simulation,
and stochastic noise construction.

## Cell complexes and cochains

`CellComplexTopology` stores one oriented entity set per degree and sparse consecutive
incidences. Construction validates boundary-of-boundary without dense matrices.

`CochainDiscretization` binds Hodge stars, primal and dual measures, boundary masks,
and one `DiscreteFieldSpace` per cochain degree. Graph cochain utilities use this
canonical discretization while retaining `GraphIR` as an execution view.

Triangle and segment geometry expose canonical cell-complex views through their
simplicial topology objects. `TriangleMesh.discrete_support()` binds this topology to
the mesh embedding identity.

## P1 finite elements

The first variational backend is scalar conforming H¹ P1 on affine triangles:

```python
plan = phx.discretization.P1FiniteElementPlan(
    vertices,
    faces,
    field_name="u",
)
space = plan.prepare()

weak_problem = phx.equations.VariationalProblemIR(
    "poisson",
    "u",
    diffusion=1.0,
    source=lambda points: forcing(points),
    dirichlet=lambda points: boundary_value(points),
)
compiled = phx.equations.compile_variational_problem(weak_problem, space)
solution, linear_result = compiled.solve()
```

Preparation constructs consistent mass and stiffness operators, lumped vertex
measure, P1 basis gradients, trace subsets, and sparse Dirichlet elimination. Volume
loads use a degree-two three-point triangle rule; Neumann loads use boundary-edge
midpoints. `p1_local_matrices` is differentiable with respect to vertex coordinates
under fixed topology.

`space.heat_dynamics(diffusivity)` returns mass-matrix-consistent semidiscrete heat
dynamics suitable for differential solvers.

## Finite volume

Structured finite volume binds cell-average fields and directional face-flux spaces to
an interval-primary tensor grid:

```python
grid = phx.discretization.TensorGridPlan(
    (phx.discretization.UniformCellAxisSpec(128, periodic=True),),
    axis_names=("x",),
).prepare(jnp.asarray([[0.0], [1.0]]))

space = phx.discretization.FiniteVolumePlan(grid, field_name="u").prepare()
system = phx.equations.ScalarConservationSystem(
    1,
    lambda state, axis, args: args["speed"] * state,
    lambda left, right, axis, args: jnp.full(
        left.shape[:-1], jnp.abs(args["speed"])
    ),
    system_id="linear-advection",
)
law = phx.equations.ConservationProblemIR(
    "transport",
    "u",
    system,
    phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
)
method = phx.discretization.FiniteVolumeMethodPlan(
    phx.discretization.MUSCLReconstruction(),
    phx.discretization.RusanovFluxPlan(),
)
dynamics = phx.equations.compile_conservation_problem(law, space, method)
```

Every internal integrated face flux is evaluated once and applied with opposite signs
to its neighboring cells. Geometry, physical systems, reconstruction, interface
solvers, boundaries, and time integration remain separate plans. See
[Structured finite volume](guides_finite_volume.md) for high-resolution systems,
wave propagation, mapped grids, projection, multiblock coupling, AMR, and
differentiability.

## Temporal and stochastic discretizations

`TemporalMesh` distinguishes an actual physical-time partition from requested output
sampling. Uniform path-integral slicing uses:

```python
slicing = phx.discretization.TemporalMesh.uniform(
    0.0,
    1.0,
    64,
    role="path",
)
```

Adaptive DAE solutions carry `RealizedTemporalMesh`, including fixed-capacity
accepted times, a validity prefix, the realized count, source plan ID, and requested
output-grid ID. Their result bundle contains both the incoming spatial/formulation
records and the realized internal-time record.

`SpatialNoiseBasis` lives in `phydrax.stochastic` and binds to an exact
`DiscreteFieldSpace.field_space_id`:

```python
noise = phx.stochastic.SpatialNoiseBasis.from_spectrum(
    space,
    lambda eigenvalue: 0.02 * jnp.exp(-0.05 * eigenvalue),
    rank=8,
)
```

`StochasticCouplingPlan` retains solver/observable coupling semantics and owns a
generic `DiscretizationHierarchy` whose levels contain complete bundles. Stochastic
realization identity, coupling identity, spatial field identity, noise basis identity,
and refinement identity remain separate.

## PINNs and operator learning

PINNs do not receive a fabricated spatial state discretization. `FunctionalSolver`
records parametric trial functions and residual terms in a bundle; integration and
collocation remain explicit term realizations.

`FunctionSamples` records separate `support_id` and `measure_id` values. Source and
query samples can therefore have distinct supports, measures, and graph views without
conflating sample coordinates with field DOFs.

## Refresh and adaptation

Prepared objects may refresh numeric geometry or coefficients only when topology,
DOF layout, and symbolic sparsity remain fixed. A topology or DOF-count change
requires replanning. Fixed-capacity masks remain part of topology and realization
identity, and inactive payloads must remain numerically inert.
