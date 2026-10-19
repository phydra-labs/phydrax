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
from phydrax.units import METER

profile = phx.geometry.Circle((0.0, 0.0), 1.0)
cylinder = profile.extruded(2.0).compile(coordinate_unit=METER)

torus_profile = phx.geometry.Circle((2.0, 0.0), 0.25)
torus = torus_profile.revolved().compile(coordinate_unit=METER)
```

Extrusion preserves exact signed-distance semantics only when the profile is
an exact signed distance. Revolution additionally requires the complete
profile to remain in the non-negative radial half-plane. Query
`compiled.validity()` before using a state-dependent representation.
Revolution measure, uniform sampling, and boundary charts are intentionally
unavailable until radial-moment cubature is supplied.

## Certified tubes and construction ranges

`Tube` supports finite straight segments and full exactly orthonormal circles,
not a general sampled centerline. The radius must be positive and strictly
below certified reach. For a circle of radius two the tube's open feasible
radius interval is `(0, 2)`; a straight segment has infinite reach and round
caps. Compilation binds radius in the explicit coordinate length unit.

The executable blocks below run sequentially in one process; later blocks reuse
the earlier imports and coordinate-request helper. Expected standard output: none.

```python {.executable}
import jax
import numpy as np
import phydrax as phx

phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
circle = phx.geometry.CircleCurve(
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.0
)
tube = phx.geometry.Tube(circle, 0.25, include_parameter_range=True)
tube_field = tube.compile(coordinate_unit=phx.units.METER)
assert tube.reach == 2.0
assert tube.parameter_range is not None
assert tube.parameter_range.interval == (0.0, 2.0)
path_bounds = tube.distance_bounds(np.array([[2.0, 0.0, 0.5]], dtype=np.float64))
assert path_bounds[0, 0] <= 0.5 <= path_bounds[0, 1]
```

The bounds enclose centerline distance, before subtracting tube radius. Invalid
radius requests retain `CertifiedParameterRange` on their refusal; read it with
`phx.geometry.refusal_parameter_range(error)` rather than parsing text.
Fillet/chamfer result refusals also retain `.parameter_range`. Offset and
deformation success records expose it only when requested. `attained_exactly`
describes a limiting endpoint, never permission to use the boundary;
`certified_inner_bound` is conservative and `unknown` means no admitted proof.
Chamfer scale preserves the authored setback ratio. No unknown end is promoted
to infinity, and resource refusals do not become feasible-range estimates.

## Host certification and coordinate tickets

Certified B-Rep queries use a request-owned coordinate-enclosure budget.
The following preparation uses the canonical ticket/budget owners explicitly;
these host objects are not JAX model inputs or process-global caches.
Subsequent examples in this section reuse `coordinate_request`.

```python {.executable}
from collections.abc import Iterator
from contextlib import contextmanager

from phydrax import HostBatchTicket
from phydrax.discretization import CoordinateEnclosureBudget


@contextmanager
def coordinate_request() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=1_000_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=100_000_000,
            maximum_memory_bytes=1_000_000_000,
        )
        with budget.activate():
            yield budget
```

`HostBatchTicket(work_upper, bytes_upper=...)` declares the parent host-work and
simultaneous managed-storage allowances. `CoordinateEnclosureBudget` borrows
that admitted `BatchTicket` and separately declares coordinate
`maximum_work_units` and `maximum_memory_bytes`; neither layer can grant more
credit to the other. Enter the ticket before activating its coordinate budget,
and retain that lifetime through the whole host preparation. Child ticket
reservations cannot overcommit the parent.

These declarations cap work and scratch storage, not geometric error.
`phydrax.discretization.CoordinateEnclosureResourceError` retains the refused
resource, limit, requested amount, completed amount, and available phase evidence.
`RationalEnclosureError` reports an unresolved rational denominator or chart
limit; it is not permission to approximate the source. Refusals propagate without
raising budgets or widening tolerances. Use JAX x64 for binary64 enclosure
publication and never replace a failed proof with fine/coarse agreement.

### Mass, edge, normal, and selection evidence

This source-only box needs no query tessellation to certify native properties.
The volume oracle is six cubic meters, independent of the boundary integration.
Native edge status is convex; geometric selection returns explicit unresolved
membership rather than a sampled yes/no decision.

```python {.executable}
lazy = phx.geometry.BRepTessellationPolicy(realize=False)
box = phx.geometry.brep_box(
    (0.0, 0.0, 0.0),
    (1.0, 2.0, 3.0),
    coordinate_contract=phx.SpatialCoordinateContract(phx.units.METER),
    tessellation=lazy,
)
with coordinate_request() as budget:
    properties = phx.geometry.brep_mass_properties(
        box, box.require_geometry(), subdivisions=1, coordinate_budget=budget
    )
    edges = phx.geometry.brep_edge_classification(
        box, box.require_geometry(), maximum_cells=4, coordinate_budget=budget
    )
    normals = phx.geometry.brep_normal_ranges(
        box, (0.0, 0.0, 1.0), draft_angle=0.1,
        maximum_cells=8, coordinate_budget=budget,
    )
    predicate = (
        phx.geometry.BRepPredicate("analytic_family", family="plane")
        & phx.geometry.BRepPredicate("facing", direction=(0.0, 0.0, 1.0), angle=0.1)
    )
    outlet = phx.geometry.select_brep_entities(
        box, predicate, maximum_cells=8, coordinate_budget=budget
    )
assert properties.volume_enclosures[0, 0] <= 6.0 <= properties.volume_enclosures[0, 1]
assert all(edge.status == "convex" for edge in edges)
assert len(normals) == 6
assert len(outlet.selected) == 1
assert not outlet.unresolved
```

Mass-weighted inertia additionally requires an explicit density and its native
mass/length-cubed unit, with a coordinate unit matching the source exactly.
Signed shell-volume queries retain negative cavity and zero-containing rows;
they do not inherit the mass-property positivity gate. Normal angles use
outward material-side normals, and threshold contacts/singular normal sets
remain explicit. Selections combine through three-valued `&`, `|`, `~`, and
`.neighbors()`. Bind a nonempty face selection with `.bind_region(...)` using
exhaustive recipe names, and explicitly choose inclusion for unresolved members.
See [recipe region binding](guides_planar_process_layout.md#bind-selected-boundary-regions).

### Sections, clearance, and contact kinds

Sections retain certified native curves and unresolved source parameter regions,
not a polygon fitted to intersection samples. Sphere/plane and cylinder/plane
sections use admitted analytic/native carriers. Nonrectangular model face trims
explicitly refuse the section wrapper; this restriction does not imply that
general clearance ignores trims.

```python {.executable}
import math


def sphere_region(center: tuple[float, float, float], radius: float) -> phx.geometry.SurfaceRegion:
    return phx.geometry.SurfaceRegion(
        phx.geometry.SpherePatch(
            center, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), radius
        ),
        [[0.0, -math.pi / 2], [math.tau, math.pi / 2]],
    )


first_sphere = sphere_region((0.0, 0.0, 0.0), 1.0)
second_sphere = sphere_region((5.0, 0.0, 0.0), 2.0)
with coordinate_request() as budget:
    clearance = phx.geometry.brep_clearance(
        first_sphere, second_sphere, coordinate_budget=budget
    )
    contact = phx.geometry.brep_section(
        first_sphere,
        phx.geometry.PlanePatch((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        coordinate_budget=budget,
    )
assert clearance.status == "certified"
assert clearance.distance_lower <= 2.0 <= clearance.distance_upper
assert contact.status == "certified"
assert contact.intersections[0].points[0].kind == "tangent_point"
assert contact.intersections[0].points[0].branch_count == 0
```

Clearance is minimum boundary distance, not penetration depth. General native
faces use BVH/parameter branch-and-bound with certified trim membership.
`unresolved_pairs` and `boxes_processed` preserve budget-limited evidence;
`distance_lower` is not a sampled guess and witnesses ground the upper bound.
Contact kinds distinguish crossings, tangent points/curves, and singular points.
An unproved indicatrix reports branch count `-1`, never a guessed branch number.

### Sewing preserves authored carriers

For existing native source models, choose `BRepSewingPolicy` with a fixed
`maximum_gap` and call `sew_brep` according to its source-model contract.
Positive-gap admission currently supports unique full-range parallel
equal-length line pairs, not arbitrary close curves. Retained pair-carrier
bounds, canonical coedge face-gap rows, and endpoint gaps are distinct evidence.
Partial/curved/ambiguous pairings refuse. Publication and native meshing replay
recompute bounds; an underclaim is not trusted. STEP source uncertainty is
capped at the policy maximum. No geometry moves and no retry widens the gap.

### Fillets, offsets, and defeaturing

Source identities, not mesh edges or display tags, select a blend. Inspect
`BRepBlendResult.status` before accessing `.model`; separately inspect
`sensitivity_status` before using its optional radius `.source`.
Perpendicular/circular/parallel-cylinder admitted radius maps transport all
native seam data. Oblique intersection-cap fillets retain a certified primal
but explicitly refuse linked radius sensitivity. Multiple incident selected
edges do not acquire a fabricated corner patch.

For the source-only box above, select a z-directed edge through its native
vertex incidence. Restore its local cylindrical blend through defeaturing;
the volume-change oracle is `(1 - pi/4) * radius**2 * edge_length`.

```python {.executable}
native_geometry = box.require_geometry()
vertical_edge = next(
    index for index, (first, last) in enumerate(native_geometry.edge_vertices)
    if np.array_equal(
        np.abs(np.asarray(native_geometry.vertex_points[last] - native_geometry.vertex_points[first])),
        (0.0, 0.0, 3.0),
    )
)
fillet = phx.geometry.brep_fillet(
    box, (box.edge_ids[vertical_edge],), 0.25,
    tessellation=lazy, include_parameter_range=True,
)
chamfer = phx.geometry.brep_chamfer(
    box, (box.edge_ids[vertical_edge],), (0.125, 0.25), tessellation=lazy
)
assert fillet.status == "certified"
assert chamfer.status == "certified"
assert fillet.model is not None
assert fillet.parameter_range is not None
blend_face = next(
    index for index, patch in enumerate(fillet.model.patches)
    if isinstance(patch, phx.geometry.CylinderPatch)
)
removed = phx.geometry.brep_defeature(
    fillet.model, fillet.model.face_ids[blend_face], operation="fillet", tessellation=lazy
)
assert removed.status == "certified"
assert removed.model is not None
volume_change = (1.0 - math.pi / 4.0) * 0.25**2 * 3.0
assert removed.volume_change_enclosure[0] <= volume_change <= removed.volume_change_enclosure[1]
assert len(removed.model.face_ids) == 6
```


Offsets now return a record: use `result.model`, never pass the result itself
where a `BRepModel` is required.

```python {.executable}
offset = phx.geometry.brep_offset(
    box, 0.25, tessellation=lazy, include_parameter_range=True
)
assert offset.evidence.volume_status == "certified"
assert offset.evidence.volume_enclosure is not None
assert offset.evidence.volume_enclosure[0] <= 13.125 <= offset.evidence.volume_enclosure[1]
assert offset.parameter_range is not None
offset_model = offset.model
```

Here each dimension increases by twice the offset: `1.5 * 2.5 * 3.5 = 13.125`.
Planar rounding evidence encloses rational intent and represented coefficients.
Sharp curved edges, vanishing/reversed edges, and unexcluded collisions refuse.
Smooth offset trees have explicitly absent planar rounding/volume evidence,
not zero-error certificates; query mass properties independently.

`brep_defeature(model, face, operation="fillet" | "chamfer" | "hole", ...)`
removes one source feature only in admitted local analytic families. Inspect
certified/unsupported/unresolved/collision status, volume-change enclosure,
gap contribution, Hausdorff enclosure, and surviving lineage before handing
the result to simulation. A bounding-box overlap with an unsupported curved
face is unresolved, not proof of collision freedom. Unaffected recipe face
keys survive; newly created sharp edges can make exhaustive edge naming
unsupported.

### Polynomial deformation and fixed topology

A lattice fixes identity boundary controls and binds only free interior
positions as dynamic length-unit leaves. The determinant certificate covers
the entire box; it is not a sample of Jacobians. The named taper/shear forms
have certificate-qualified inverses. Twist, bend, box-crossing patches,
restricted/multi-span splines, and unsupported implicit intersection carriers
refuse rather than being refitted.

```python {.executable}
proof_box = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
controls = phx.geometry.identity_lattice(proof_box, (2, 2, 2))
controls = controls.at[1, 1, 1, 0].add(0.125)
deformation = phx.geometry.PolynomialDeformation(proof_box, controls)
with coordinate_request() as budget:
    injectivity = phx.geometry.certify_deformation(
        deformation, maximum_cells=32, coordinate_budget=budget
    )
assert injectivity.status == "certified"
assert all(cell.determinant_range[0] > 0.0 for cell in injectivity.cells)
```

`deform_brep(...).parameter_map` feeds the existing
`FixedTopologyBRepSource(model, parameter_map=...)`; it preserves UV trims and
incidence while composing admitted carriers with coefficient enclosures.
The map acts on shared definition coordinates. Occurrence placements stay
declared, not converted to per-instance world-space motion.
`deformation_feasibility_scale` certifies the whole displacement-scale interval
and retains determinant-limiting cells; it never calls a singular supremum feasible.

### Certify a sketch before lowering

`Sketch.solve()` convergence is numerical evidence; unique-root certification
is a separate host step. This circle fixes center and radius with authored IDs.

```python {.executable}
from phydrax.geometry.design import (
    FixedPoint, Radius, Sketch, certify_sketch_solution,
    sketch_dimension_derivatives,
)

sketch = Sketch(
    [[0.2, -0.1]], circle_centers=[0], circle_radii=[1.0],
    constraints=(FixedPoint(0, (0.0, 0.0)), Radius(0, 2.0)),
    constraint_ids=("center", "radius"), feature_id="certified-circle",
)
with coordinate_request() as budget:
    certificate = certify_sketch_solution(sketch.solve(), budget=budget)
    derivatives = sketch_dimension_derivatives(certificate, budget=budget)
assert certificate.status == "certified"
exact = np.array([0.0, 0.0, 2.0], dtype=np.float64)
assert np.all(np.asarray(certificate.lower) <= exact)
assert np.all(exact <= np.asarray(certificate.upper))
certified_profile = certificate.to_profile()
assert certified_profile.certificate is certificate
assert len(derivatives) == 1
```

Rank-gap/backward-error evidence and component constraint IDs explain
under/over/redundant systems. Redundant rectangular systems can report DOF
without a square-system root certificate. Custom interval primitives refuse
explicitly. Dimension derivatives attach to regular certified endpoints, not
LM iterations. `solve_priority_sketch` accepts cumulative priority tiers only
after strict root inclusion; residual satisfaction alone does not advance a
tier. Keep its rejected/unsatisfied IDs and every attempted tier certificate.

## Learning from parametric B-Rep samples

The incidence graph owns immutable topology and entity identity; the prepared
sample plan evaluates realized JAX geometry. Attach mass/edge descriptors only
from the same model. Stable recipe keys, not revision-scoped row IDs, connect
supported entities across edits. The following small surrogate learns sphere
area from a pooled face-position feature, with an independent closed-form target.
It is a pooled B-Rep sample baseline, not a claim that a graph neural network
has learned topology changes.

The runnable B-Rep graph surrogate, `examples/brep_graph_surrogate.py`,
builds a five-member radius family through public design APIs, attaches
certified `brep_mass_properties` targets, and trains an Equinox readout of
incidence-graph messages. Its bounded plain-gradient loop needs no optimizer
dependency. One radius-gradient step is followed by independent B-Rep
certification at the proposed radius:

```python {.executable}
from examples.brep_graph_surrogate import run

surrogate_step = run()
assert surrogate_step["radius_after"] < surrogate_step["radius_before"]
assert surrogate_step["prediction_after"] < surrogate_step["prediction_before"]
assert surrogate_step["certified_after"] < surrogate_step["certified_before"]
```

Run it with `JAX_ENABLE_X64=1 python -m examples.brep_graph_surrogate` to print
the before/after radius, surrogate prediction, certified area, and training
loss. The graph topology is fixed; the derivative flows through sampled
coordinates and the trained readout, not host certification.

```python {.executable}
import jax.numpy as jnp

sphere = phx.geometry.brep_sphere(
    1.0, coordinate_contract=phx.SpatialCoordinateContract(phx.units.METER)
)
incidence = phx.graph.brep_incidence_graph(sphere)
compiled = phx.geometry.FixedTopologyBRepSource(sphere).compile(
    coordinate_unit=phx.units.METER
)
radius_index = next(
    index for index, spec in enumerate(compiled.schema.specs)
    if spec.parameter_id.name == "radius"
)
# The exact analytic family owns whole-sphere validity; B-Rep sampling owns chart features.
analytic_sphere = phx.geometry.Sphere(
    (0.0, 0.0, 0.0), 1.0, feature_id="sphere-design"
).compile(coordinate_unit=phx.units.METER)
analytic_radius_index = next(
    index for index, spec in enumerate(analytic_sphere.schema.specs)
    if spec.parameter_id.name == "radius"
)
with coordinate_request() as budget:
    sample_plan = phx.graph.prepare_brep_graph_samples(
        sphere, grid_size=3, coedge_samples=3,
        policy=phx.geometry.brep.BRepQueryPolicy(
            maximum_operations=100_000, maximum_points=10_000,
            maximum_scratch_bytes=1 << 20,
        ),
        budget=budget,
    )


def face_feature(radius: jax.Array) -> jax.Array:
    state = compiled.state.replace_at(radius_index, radius)
    samples = sample_plan.evaluate(compiled.kernel.realize(state))
    # Full sphere position charts: pole positions remain valid even when normals are singular.
    radius_squared = jnp.mean(jnp.sum(samples.face_grids[0][:3] ** 2, axis=0))
    return radius_squared[None]


radii = jnp.array([0.75, 1.0, 1.25, 1.5, 1.75], dtype=jnp.float64)
for radius in radii:
    assert bool(analytic_sphere.validity(
        analytic_sphere.state.replace_at(analytic_radius_index, radius)
    ).accepted)
features = jax.vmap(face_feature)(radii)
area_targets = 4.0 * jnp.pi * radii**2
fit = phx.ml.fit(
    phx.ml.linear.RidgeRecipe(alpha=1e-12), features, area_targets,
    feature_schema=phx.ml.FeatureSchema(
        ("pooled_radius_squared",), dimensions=(phx.units.AREA,)
    ),
    target_schema=phx.ml.TargetSchema(
        "continuous", names=("surface_area",), dimensions=(phx.units.AREA,)
    ),
)
assert bool(fit.valid)
assert incidence.model_id == sphere.model_id
np.testing.assert_allclose(
    jax.vmap(fit.model)(features), area_targets, rtol=1e-8, atol=1e-8
)


def design_objective(radius: jax.Array) -> jax.Array:
    predicted_area = fit.model(face_feature(radius))
    return (predicted_area / (4.0 * jnp.pi) - 1.0) ** 2


initial_radius = jnp.asarray(1.5, dtype=jnp.float64)
gradient = jax.grad(design_objective)(initial_radius)
candidate_radius = initial_radius - 0.01 * gradient
assert bool(jnp.isfinite(gradient))
assert bool(analytic_sphere.validity(
    analytic_sphere.state.replace_at(analytic_radius_index, candidate_radius)
).accepted)
candidate_realization = compiled.kernel.realize(
    compiled.state.replace_at(radius_index, candidate_radius)
)
assert bool(jnp.isfinite(candidate_realization.seam_residual))
assert bool(jnp.isfinite(candidate_realization.seam_bound))
assert bool(candidate_realization.seam_residual <= candidate_realization.seam_bound)
assert bool(design_objective(candidate_radius) < design_objective(initial_radius))
```

Training freezes the fitted model; design gradients still propagate through
realized samples to the radius. The same scalar radius is explicitly bound to
the exact analytic `Sphere` validity owner and the B-Rep chart state. Every
training radius and the candidate must pass whole-sphere validity; the candidate
also retains numerical seam acceptance. Generic restricted B-Rep
`CompiledGeometry.validity()` is inconclusive without a representation-specific
provider, so it is not substituted for this exact family proof. These
coefficient-only sample states are not newly published native B-Rep models.
This is one explicit candidate step within a fixed complete-sphere family,
not automatic topology optimization.

General consumers must honor `trim_status` and `normal_status` separately;
unresolved trim-band points are not inside masks and singular normals are not
unit vectors. Weighted `face_quadrature` supplies `FunctionSamples` for geometric
operators; coedge samples follow sense. Linked grids/p-curves/quadrature
transport with the realization, and topology changes require fresh preparation.
See the [graph API](api/operators/graph.md#b-rep-incidence-and-differentiable-samples).


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
margin, and trust-region evidence pass. Failed proposals return finite
reference points and zero accepted motion, but their forward and reverse
sensitivities are NaN. The finite rollback is diagnostic state, not a
zero-sensitivity physical solution.

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

1. evaluate the lattice sign pattern in one batched field call;
2. locate every sign-changing edge root with one vectorized ITP program;
3. split active-cell inside corners into connected components through
   lookup tables over the 256 cube sign configurations;
4. build manifold dual incidence;
5. fit batched regularized QEF vertices with native small solves;
6. orient (one batched gradient) and validate a closed triangle topology;
7. freeze topology, anchor routes, face diagonals, and the BVH overlap pairs
   of triangles' reachable dual-cell boxes, which is the complete
   self-intersection candidate set for every accepted realization.

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

## Neural implicit regions

`NeuralImplicitRegion(network, bounds, coordinate_unit=METER, ...)` represents the region
`{x in bounds : phi(x; w) <= 0}` of a scalar network `phi`, negative inside, over
explicit axis-aligned bounds. Every PARAMETER array of the network (by its
resolved array roles) becomes a design parameter `ParameterId(feature_id, path)`
of the compiled `DesignState`; FIXED arrays such as `fixed_field` data stay fixed
kernel data and never become design parameters, and networks with model state are
refused. The kernel rebuilds the network from the state, its fixed data, and its
static structure. Weights therefore train only through the geometry design-state
route (`with_parameters`, `with_state`, or an objective over the state). A
`GeometryDomain` wrapping the compiled geometry stays a fixed domain.

Construction checks sampled evidence at the current weights host-side and
refuses (`ValueError`) when a check fails:

1. a Lipschitz upper bound, constructed for a plain `MLP` as the product of layer
   Frobenius norms and activation Lipschitz constants (`CONSTRUCTED`), or declared
   through `lipschitz_upper_bound` (`DECLARED`); either is checked against
   sampled gradient norms;
2. an `evaluation_error` bound on evaluated field values;
3. sign margins: `interior_points` satisfy `phi <= -(sign_margin + error)`, while
   `exterior_points` and the lattice nodes on the bounds satisfy
   `phi >= sign_margin + error`;
4. a sampled topology: the Betti numbers and boundary components of the zero set
   resolved from the field signs on a uniform `discovery_resolution` lattice.

The evidence is sampled, not a proof. No covering argument bounds the field
between lattice nodes and sample points, so a zero-set component or sign change
between samples can go undetected. The `FieldCertificate` therefore reports
`SignReliability.LOCAL`, `ZeroSetAccuracy.APPROXIMATE`, and no
`topology_identity`, whether the Lipschitz bound is constructed or declared; it
carries the Lipschitz bound and evaluation error. Consumers that require reliable
sign, such as implicit curve and surface discovery, refuse these regions.
`BOUNDARY_NORMAL` is advertised only for networks whose declared regularity is at
least C1 (for example `tanh` or `sin`, not ReLU) when `gradient_margin` holds at
the lattice zero crossings. Measures, boundary sampling, and closest points are
not advertised.

```python
# `network`: a pretrained phx.nn.models.MLP(in_size=2, out_size="scalar", ...)
# whose zero set bounds the body.
region = phx.geometry.NeuralImplicitRegion(
    network,
    [[-1.5, -1.5], [1.5, 1.5]],
    coordinate_unit=METER,
    interior_points=[[0.0, 0.0]],
    exterior_points=[[1.4, 1.4]],
    sign_margin=0.05,
    evaluation_error=1.0e-12,
    gradient_margin=0.1,
    discovery_resolution=33,
    feature_id="neural-body",
)
geometry = region.compile(coordinate_unit=METER)
domain = phx.domain.GeometryDomain(geometry)

trained = geometry.with_parameters({region.parameter_ids[0]: updated_weight})
checked = region.recertify(trained.state).compile(coordinate_unit=METER)
```

`CompiledGeometry.validity(state)` rechecks every sampled margin at any state but
stays inconclusive away from the checked weights, because the sampled topology is
resolved only at those weights. `region.recertify(state)` reruns every check at
the trained weights and rejects the state when a margin fails or the sampled
`ImplicitRegionTopology` differs, for example when a disk splits in two or
vanishes.

## Finite-element mesh motion

`FiniteElementMeshMotionPlan` maps a fixed boundary-coordinate provider into a
full-dimensional vertex-coordinate FE mesh. The support envelope is:

- two- or three-dimensional full-dimensional meshes;
- triangle, quadrilateral, tetrahedron, hexahedron, prism, or pyramid cells;
- P1/Q1 vertex coordinates;
- fixed connectivity and entity IDs;
- one supplied coordinate for every topological boundary vertex (every vertex for
  the `PRESCRIBED` route).

`FiniteElementMeshMotionPolicy(route=...)` selects the interior extension:
`HARMONIC` (graph Laplacian, valid on convex planar domains), `LINEAR_ELASTICITY`
(Jacobian-stiffened elasticity that keeps small cells near moving inclusions rigid),
`WINSLOW` (inverse harmonic map whose barrier keeps maps onto a convex reference
valid), `MMPDE` (Huang–Russell moving-mesh PDE equilibrium for a `monitor(points)`
passed to `realize`), or `PRESCRIBED`. Linear operators are fixed at preparation and
solved through `phydrax.linalg` with RHS-only differentiation; nonlinear routes solve
through `phydrax.nonlinear` with implicit root derivatives in the boundary design and
monitor parameters. The shared `MotionValidityPlan` validates signed corner
Jacobians inside the traced step and rejects orientation reversal, small Jacobians,
excessive displacement, nonfinite values, rejected boundary maps, or failed route
solves; `motion.certify(coordinates)` is the host-epoch Bernstein certificate.

```python
motion = phx.discretization.FiniteElementMeshMotionPlan(
    discretization,
    projection,
)
realization = motion.realize(candidate_state, numeric_version="shape-design-7")
context = phx.equations.FiniteElementExecutionContext(
    realization.runtime,
)
```

The caller-supplied nonempty `numeric_version` identifies the proposed numeric
coordinate state and participates in runtime identity; it is not inferred from array
values or a display label. The execution runtime always contains finite accepted
coordinates. On a failed proposal it contains the base coordinates, while
`realization.accepted` remains false. State-design line searches must include
`accepted` in both `state_admissibility` and `state_realization`; fallback physics
must never make an invalid candidate acceptable.

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
mandatorily calls the existing `reanalyze_topology_design` protocol from its
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
