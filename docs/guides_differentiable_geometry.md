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

## Choosing another geometry route

Use fixed-topology B-Rep realization when stable CAD charts and source-face
identity are available. Use IGA when the simulation is already represented by
an exact supported spline parameterization. Use cut-cell, immersed, or embedded
methods when conforming mesh coordinates are unnecessary. Use implicit surface
realization for explicit surface output or fixed-topology boundary motion; it
does not create a conforming volume topology.

See `examples/differentiable_geometry_fem.py` for an executable surface and FE
workflow.
