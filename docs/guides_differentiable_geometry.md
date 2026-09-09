# Differentiable fixed-topology geometry

Phydrax separates discrete topology discovery from differentiable coordinate
realization. Geometry and mesh topology are never differentiated. A prepared
realization is valid only while its explicit evidence remains accepted.

## Exact analytic sweeps

`Extrusion(profile, height)` lifts a two-dimensional region into a centered
local-z prism. `Revolution(profile)` interprets the profile coordinates as
`(radius, axial)` and revolves it around local z. Existing rigid transforms
place either result in world coordinates.

```python
import phydrax as phx

profile = phx.geometry.Circle((0.0, 0.0), 1.0)
cylinder = profile.extruded(2.0).compile()

torus_profile = phx.geometry.Circle((2.0, 0.0), 0.25)
torus = torus_profile.revolved().compile()
```

Extrusion preserves exact signed-distance semantics only when the profile is
an exact signed distance. Revolution additionally requires the complete
profile to remain in the non-negative radial half-plane. Query
`compiled.validity()` before using a state-dependent representation.
Revolution measure, uniform sampling, and boundary charts are intentionally
unavailable until radial-moment cubature is supplied.

## Runtime geometry validity

`CompiledGeometry.validity()` checks finite parameter leaves, every declared
`ParameterSpec.bounds`, and representation-specific conditions supplied by a
`GeometryValidityProvider`.

The disposition is one of `VALID`, `INVALID`, or `INCONCLUSIVE`. Restricted
representations without an executable provider are inconclusive. Global design
search accepts restricted representations only when their runtime validity can
be evaluated; invalid candidates are counted as invalid objective evaluations.

Field certificates remain semantic contracts. Runtime validity determines
whether those contracts apply at the current design state.

## Sparse voxel sampling

`VoxelGeometrySamplingPlan` evaluates a compiled boundary field on a prepared
`SparseVoxelGrid`. The resulting numeric values remain differentiable with
respect to the geometry state while voxel topology is fixed. Morton lookup,
active support, and narrow-band selection are discrete.

Sampling and multilinear interpolation do not preserve exact signed-distance
semantics. The returned certificate is explicitly approximate and
piecewise-smooth. An optional `ExactSDFEnclosureCertificate` can certify cell
sign only where its Lipschitz interval excludes zero; cells intersecting that
interval remain unresolved.

## Surfel geometry

`SurfelGeometryPlan` validates current positions, unit normals, scaled tangent
axes, and physical surface weights on stable point ownership. Those arrays
remain differentiable while activation, Morton ownership, overlap routes, ray
hit order, and footprint membership are fixed.

`SurfelGeometryCertificate` separates numeric validity from source authority.
It records position/normal accuracy, orientation scope, coverage scope,
footprint meaning, and optional curvature/error bounds. Local tangent patches
never acquire watertight or exact-SDF semantics implicitly. See
[Surfels](guides_surfels.md).

## Implicit point projection

`ImplicitPointProjectionPlan` binds fixed reference points to one compiled
boundary field. The primal map solves each point back onto the current zero set
inside a fixed trust region. Its derivative uses the normal gauge

```text
point_tangent = -field_parameter_tangent * field_gradient
                / squared_field_gradient_norm
```

The derivative is valid only when the returned root residual, field-gradient
margin, and trust-region evidence pass. Failed proposals return finite reference
points and zero accepted motion.

```python
projection = phx.geometry.ImplicitPointProjectionPlan(
    compiled,
    reference_boundary_points,
    trust_radii,
    source_id="body-boundary",
)
result = projection.realize(candidate_state)
if not result.accepted:
    # Reject the candidate or rebuild the topology epoch.
    ...
```

The normal gauge describes one mesh parameterization. It is not the derivative
of an external mesher and does not preserve arbitrary tangential CAD
correspondence. Use a chart-based provider when tangential material identity is
part of the model.

## Implicit surface discovery

`discover_implicit_surface` consumes a valid three-dimensional region and a
nonperiodic `PreparedTensorGrid`. Discovery is host-side and concrete:

1. evaluate the lattice sign pattern;
2. locate every sign-changing edge root;
3. split active-cell inside corners into connected components;
4. build manifold dual incidence;
5. fit regularized QEF vertices;
6. orient and validate a closed triangle topology;
7. freeze topology, anchor routes, face diagonals, and intersection pairs.

```python
grid = phx.discretization.TensorGridPlan(
    tuple(phx.discretization.UniformAxisSpec(17) for _ in range(3)),
    axis_names=("x", "y", "z"),
).prepare([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])

surface_plan = phx.geometry.discover_implicit_surface(
    compiled,
    grid,
    source_id="body-surface",
)
surface = surface_plan.realize(candidate_state)
```

Runtime realization preserves static connectivity and output shapes. It checks
geometry validity, the complete lattice sign pattern, root regularity and trust,
QEF solve status, cell containment, face orientation/area, and intersections
between nonadjacent triangles. Invalid output falls back to the discovery
vertices. `refresh_required` means the host must stop the current topology
epoch and rediscover.

Lattice values within the configured zero tolerance fail discovery. Shift or
refine the grid rather than relying on an arbitrary tie convention.

`ImplicitSurfaceRealization.to_triangle_mesh()` is host-only and requires an
accepted concrete realization. Ordinary dense-grid evidence is sampled/local;
it is not a global topology theorem. A `CertifiedImplicitTopology` is required
for a certified-topology claim.

## Finite-element mesh motion

`FiniteElementMeshMotionPlan` maps a fixed boundary-coordinate provider into a
full-dimensional vertex-coordinate FE mesh. The initial support envelope is:

- two- or three-dimensional full-dimensional meshes;
- triangle, quadrilateral, tetrahedron, or hexahedron cells;
- P1/Q1 vertex coordinates;
- fixed connectivity and entity IDs;
- one supplied coordinate for every topological boundary vertex.

Interior displacement is the graph-harmonic extension of boundary displacement.
The graph operator is fixed at preparation and solved through `phydrax.linalg`
with RHS-only differentiation. The plan validates signed coordinate Jacobians
at deterministic reference probes and rejects orientation reversal, small
Jacobians, excessive displacement, nonfinite values, rejected boundary maps,
or failed extension solves.

```python
motion = phx.discretization.FiniteElementMeshMotionPlan(
    discretization,
    projection,
)
realization = motion.realize(candidate_state)
context = phx.equations.FiniteElementExecutionContext(
    realization.runtime,
)
```

The execution runtime always contains finite accepted coordinates. On a failed
proposal it contains the base coordinates, while `realization.accepted` remains
false. State-design line searches must include `accepted` in both
`state_admissibility` and `state_realization`; fallback physics must never make
an invalid candidate acceptable.

Boundary entity membership is static during an epoch. Reclassifying a boundary,
changing connectivity, or rebuilding a volume mesh is a topology event. Phydrax
does not differentiate or automatically transfer state across that event.

## Learned design on a fixed physical realization

Learned parameterizations restrict an existing physical state/design problem;
they do not discover mesh topology or train a decoder.
`reparameterize_state_design(problem, decode, latent_template,
physical_template, *, decoder_id, realization_id, latent_bounds=None,
design_admissibility=None)` freezes the decoder and composes it through the
physical residual, objective, constraints, bounds, and state certification.
Templates fix both PyTree schemas, shapes, and dtypes. The realization identity
must continue to name the same geometry, mesh, and schema, and the optional
admissibility predicate must reject invalid decoded geometry before the
physical solver runs rather than substitute fallback geometry.

Original physical bounds remain composed constraints; latent bounds are
additional and never replace them. `StateDesignParameterization.response_vjp`
solves and accepts the physical state and transpose system before pulling the
response through the decoder. It forms neither a dense decoder Jacobian nor an
optimizer derivative. A stationary result for
`parameterization.problem` is stationary only over the frozen decoder image,
not the full coordinate, density, or topology space.

Solid mechanics supplies two exact lowerings over this foundation:

- `prepare_learned_shape_design` requires an existing `StateDesignProblem`
  whose state solver is `FiniteElementStateSolver` or
  `NeuralVariationalStateSolver`, an exact-schema `DesignState` physical
  template, and a mandatory realized-geometry admissibility predicate.
  Nontrainable schema parameters must remain equal to the template, and schema
  bounds are retained in addition to the original physical constraints.
- `prepare_learned_topology_design` requires a
  `TopologyMechanicsProblem` and a decoder returning the prepared cell-density
  array. Cells outside the prepared design mask remain exactly at their fixed
  densities before and after the existing density transform. The original
  filter, projection, material interpolation, volume constraint, load cases,
  branch gates, and FE authority remain in force. Despite the API name, the FE
  cell connectivity and prepared density-transform realization are fixed.

Shape lowering returns a `StateDesignParameterization`; solve its `.problem`
with the ordinary state/design methods and separately decide the required
physical reporting or reanalysis route. Topology lowering returns
`LearnedTopologyDesign`.
`solve_learned_topology_design(design, initial_states, initial_latent,
reanalysis_plan, initial_reference_state, *, method=None, termination=None,
args=None)` performs the latent solve—`ReducedMMA` by default—and then
mandatorily calls the existing `reanalyse_topology_design` protocol from its
`TopologyReanalysisPlan`.

`LearnedTopologyResult.accepted` requires latent solve success, accepted source
state/adjoint evidence, accepted reference transfer and FE mechanics
reanalysis, and the reference problem's volume constraint. Its
`latent_result` certificate remains a latent-coordinate certificate.
Independent reference FE reanalysis, not the decoder or a learned proposal, is
the final physical authority.

If a binary or otherwise hard design is required, implement extraction in the
reanalysis plan's transfer function. That transfer and the final reference
solve occur after the smooth latent optimization and outside every derivative.
Thresholding is never smuggled into the decoder VJP, and successful relaxed
mechanics cannot stand in for acceptance of the extracted design.

## Choosing another geometry route

Use fixed-topology B-Rep realization when stable CAD charts and source-face
identity are available. Use IGA when the simulation is already represented by
an exact supported spline parameterization. Use cut-cell, immersed, or embedded
methods when conforming mesh coordinates are unnecessary. Use implicit surface
realization for explicit surface output or fixed-topology boundary motion; it
does not create a conforming volume topology.

See `examples/differentiable_geometry_fem.py` for an executable surface and FE
workflow.
