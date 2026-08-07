# Geometry substrate

`phydrax.geometry` is the representation-aware geometry layer. It separates
host-side construction from JAX execution and gives analytic primitives, simplicial
meshes, CAD B-Reps, CSG expressions, and reconstructed geometry one compiled
contract.

## Source, kernel, state, and realization

A `GeometrySource` is the authoritative construction object. Calling `compile()`
produces a `CompiledGeometry` containing:

- an immutable `GeometryKernel` that defines algorithms and topology;
- a dynamic `DesignState` containing all trainable numeric values;
- a `ParameterSchema` with stable, feature-scoped parameter identities; and
- a `GeometryTolerance` used by tolerance-sensitive queries.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

source = phx.geometry.Sphere(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    feature_id="body",
)
geometry = source.compile()

radius = phx.geometry.ParameterId("body", "radius")
larger = geometry.with_parameters({radius: 1.5})
volume_gradient = jax.grad(
    lambda value: geometry.kernel.measure(
        geometry.state.replace_at(geometry.schema.index(radius), value)
    )
)(jnp.asarray(1.0))
```

Compiled queries are JAX-safe and batch-preserving:

```python
points = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
inside = geometry.contains(points)
signed_field = geometry.boundary_field(points)  # negative inside
normals = geometry.boundary_normal(points)
```

Use `phx.domain.GeometryDomain(geometry)` only when the geometry must participate
in labeled domains, components, integration, or constraints. Construction and CSG
belong in `phx.geometry`; the domain layer is deliberately a thin adapter.

## Capabilities and field certificates

Every kernel declares a set of `GeometryCapability` values. Consumers can require
region queries, signed fields, boundary normals, measures, sampling, boundary
atlases, or seam diagnostics without checking concrete representation classes.

`FieldCertificate` records the actual numerical contract of the boundary field:
zero-set accuracy, sign reliability, distance semantics, regularity, validity
region, safe-step information, and parameter differentiability. Approximate CAD
and reconstruction fields therefore do not masquerade as exact analytic signed
distances.

## Analytic geometry and CSG

Analytic sources provide closed-form fields, measures, samplers, and boundary
atlases. Sources compose before compilation:

```python
left = phx.geometry.Sphere((-0.4, 0.0, 0.0), 1.0, feature_id="left")
right = phx.geometry.Sphere((0.4, 0.0, 0.0), 1.0, feature_id="right")

union = (left | right).translated((0.0, 0.0, 1.0))
intersection = left & right
difference = left - right
scaled = left.scaled((1.0, 2.0, 1.0))
```

Sharp CSG preserves exact set membership but generally yields a nonsmooth level-set
field at operation seams. Blend CSG provides a smooth approximate zero set and
reports that weaker contract through its field certificate.

## Simplicial geometry

`TriangleMesh` and `SegmentMesh` own canonical validated arrays and topology.
`TriangleTopology` provides half-edge twins, boundary loops, manifold checks, and
connected components. `TriangleBVH` and `TriangleMeshQueryIndex` accelerate closest
point queries. `MeshRegion` and `PlanarMeshRegion` lower watertight 3D meshes and
planar triangulations to the common geometry kernel.

`discrete_operators(...)` constructs matrix-free DDG incidence, mass, Laplacian,
and gradient operators from the same topology. Mesh adapters accept build123d,
meshio, PyVista, and trimesh inputs through the canonical import functions instead
of maintaining representation-specific query code.

## Boundary atlases and measure partitions

`BoundaryAtlas` is the common boundary-integration structure. A chart maps a
reference coordinate to a physical boundary point and supplies its physical
Jacobian, outward frame, trim domain, source entity identity, physical tags, and
seam ownership. Atlas metadata survives rigid transforms, scaling, selection, and
fixed-topology CAD reevaluation.

```python
atlas = geometry.boundary_atlas
selected = atlas.select(entity_ids=(0,))
partition = phx.geometry.BoundaryAtlasPartition(selected)
```

`BoundaryAtlasPartition` estimates one physical measure per chart and supports
fixed-size stratified sampling. `GeometryMeasurePartition` is the explicit simplex
partition for boundary segments or planar/surface triangles. Sampling APIs return
`SamplingResult`; bounded rejection exposes completion, acceptance, and proposal
counts rather than silently returning too few points.

## CAD B-Reps

`BRep(path)` imports STEP, IGES, and BREP files through OCCT.
`BRepModel` keeps stable vertex/edge/wire/face/solid incidence, one parametric
surface patch per face, trim loops, tessellation-to-face identities, and an import
report. Supported analytic OCCT surfaces remain analytic patches; other faces are
represented by rational tensor-product B-splines.

Rational spline evaluation uses the shared span-local B-spline kernel. Each
curve query gathers `degree + 1` controls; each surface query gathers only the
tensor product of the active controls in its two parameter axes. Expanded
nonuniform and repeated OCCT knot vectors are preserved. At an exact chart
endpoint the final polynomial span supplies the one-sided differential, so
surface Jacobians and boundary frames remain finite instead of collapsing to a
constant endpoint branch.

```python
import build123d as bd

model = phx.geometry.model_from_occt_shape(
    bd.Box(1.0, 2.0, 3.0).wrapped,
    linear_deflection=0.1,
)
source = phx.geometry.BRepSource(model)
geometry = source.compile()
print(source.model.report)
```

`FixedTopologyBRepSource` reevaluates an imported tessellation from trainable
patch parameters while preserving face topology, boundary entity identity, knot
vectors, degrees, and knot-span topology. Rational B-spline control points and
weights remain differentiable; moving knots across a query is intentionally not
part of this fixed-topology contract. A realization exposes the current
vertices, faces, atlas, and a differentiable seam residual. The validity region
requires unchanged topology, positive surface Jacobians, and compatible seams;
`BRepSeamCompatibility` makes the last condition an explicit design constraint.

## Sketches and geometric constraints

`Sketch` solves fixed-connectivity 2D line/circle systems with declarative
constraints such as `Coincident`, `Horizontal`, `EqualLength`, `Radius`, and
tangency. A solved sketch lowers to `PlanarMeshRegion`.

`DesignConstraintSystem` solves geometry-state constraints without coupling to a
particular representation. Available constraints include parameter targets and
equalities, point distances, interior/exterior clearance, measure and boundary
measure targets, boundary-point conditions, and B-Rep seam compatibility.

## Bounded global design search

`DifferentialEvolutionSearch` is intended for low-dimensional geometry design
problems whose residual objective is nonsmooth, multimodal, or poorly served by a
single local initialization. It searches the squared residual from
`DesignConstraintSystem` over an explicit finite box. Differential evolution is a
stochastic global heuristic: convergence reports population-fitness dispersion, not
a proof of global optimality or coverage of every basin.

Search bounds and physical schema bounds have different roles. `ParameterSpec.bounds`
describe physical admissibility and may be one-sided or absent. `search(..., bounds=...)`
defines the finite algorithmic box. Every trainable degree of freedom must have finite
lower and upper search limits, and those limits must remain inside any finite physical
schema bounds. Scalar limits broadcast across a parameter; array limits must match its
declared shape.

The root PRNG key is required. The initial population uses the typed
`phydrax.sampling` design substrate; Latin hypercube is the default, while scrambled
Sobol and the other supported reference designs may be selected explicitly. The
current state is inserted into the first population member. Generated candidates are
reflected into the box before both evaluation and storage.

```python
import jax.random as jr
import phydrax as phx

geometry = phx.geometry.Sphere(
    center=(0.0, 0.0, 0.0),
    radius=1.0,
    feature_id="body",
).compile()
center = phx.geometry.ParameterId("body", "center")
radius = phx.geometry.ParameterId("body", "radius")

system = phx.geometry.DesignConstraintSystem(
    geometry,
    (phx.geometry.ParameterTarget(radius, 1.5),),
)
search = phx.geometry.DifferentialEvolutionSearch(
    32,
    100,
    design=phx.sampling.SobolDesign(scrambled=True),
)
global_result = system.search(
    search,
    key=jr.key(0),
    bounds={
        center: ((-0.25, -0.25, -0.25), (0.25, 0.25, 0.25)),
        radius: (0.25, 2.5),
    },
)

# Local refinement is a separate, explicit phase.
local_result = system.solve(initial_state=global_result.state)
optimized_geometry = geometry.with_state(local_result.state)
domain = phx.domain.GeometryDomain(optimized_geometry)
```

`DesignSearchResult` preserves the final population and objectives, best-objective
history, exact generation and objective-evaluation counts, invalid-evaluation count,
resolved bounds, root key, design signature, and termination reason. Its global
`converged` flag is independent of `ConstraintSolveResult.converged` from the optional
local phase. Keeping the phases separate makes extra evaluations and failure modes
observable.

Global search currently requires
`geometry.field_certificate.validity_region == "all_space"`. Restricted regions such
as fixed-topology B-Rep validity cannot yet provide an executable per-candidate
validity predicate, so search fails before objective evaluation rather than silently
crossing an uncertified region.

## Reconstruction with provenance

Point-cloud, planar, terrain, and LiDAR reconstruction are explicit pipelines:
`reconstruct_planar_region`, `reconstruct_surface_region`,
`reconstruct_dem_region`, and `reconstruct_lidar_region`. They return a
`ReconstructedGeometrySource` carrying an immutable `ReconstructionReport` with
input/output counts, algorithm parameters, watertightness, winding consistency,
recentring, warnings, and an input digest. Invalid reconstruction raises
`ReconstructionFailure` with the same report; approximation is never hidden behind
a primitive constructor.

## Core API

::: phydrax.geometry.CompiledGeometry

---

::: phydrax.geometry.GeometrySource

---

::: phydrax.geometry.FieldCertificate

---

::: phydrax.geometry.BoundaryAtlas

---

::: phydrax.geometry.TriangleMesh

---

::: phydrax.geometry.BRepModel

---

::: phydrax.geometry.FixedTopologyBRepSource

---

::: phydrax.geometry.Sketch

---

::: phydrax.geometry.DesignConstraintSystem

---

::: phydrax.geometry.DifferentialEvolutionSearch

---

::: phydrax.geometry.DesignSearchResult

---

::: phydrax.geometry.ReconstructionReport
