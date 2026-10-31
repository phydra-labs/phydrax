# Native CAD and meshing

Phydrax owns source-defined B-Rep construction and meshing. The example below constructs a rectangular prism from its planar profile and extrusion, asks the native provider for tetrahedra, and requires the accepted result to retain its source-bound certification, region-boundary evidence, and provider identity. The element-measure sum is independently compared with the analytic prism volume; a valid mesh alone is not a source-fidelity claim.

Run the workflow in an environment with the CPython 3.13 Phydrax wheel and its matching meshcore wheel installed:

```sh
python examples/native_brep_workflow.py
```

The native qualification package separates executable cases by source and consumer contract. Its default corpus includes curve-provider acceptance, rational/analytic B-Rep surface and solid routes, labeled-image material generation, mapped reference geometry, periodic quotient construction, P1 transfer, common refinement/remap, native bisection, predicates/triangulation, and native certified boundary layers. The `native-cold-fe-fv-continuation` case covers fresh-process restoration of the source, accepted mesh, and physical field; FE solves in writer and reader; native bisection with conservative/linear field-transfer checks; and a separate-process FV runtime-state restore followed by an accepted Euler-step continuation. T-spline CAD-to-IGA includes local refinement and a PDE residual oracle. Every case fails on an unmet oracle, missing certification, wrong provider identity, or unavailable native prerequisite; native requirements are not converted into skips.

```sh
python -m tools.meshing_qualification --scenario native-corpus
python -m tools.meshing_qualification --scenario native-brep-workflow
python -m tools.meshing_qualification --scenario native-image-material
python -m tools.meshing_qualification --scenario native-cold-fe-fv-continuation
python -m tools.meshing_qualification --scenario native-tspline-cad-analysis
python tools/check_native_dependency_isolation.py --python 3.13 --all-native
```

The installed isolation gate first verifies that the base Phydrax wheel refuses a native mesh without meshcore, then installs the matching native wheel and runs the positive qualification package from a temporary copy outside the source checkout. Scenario inputs are constructed in code rather than loaded from checkout fixtures; cached bytecode is excluded, and both Phydrax and meshcore must resolve from installed wheel artifacts. Optional geometry-engine imports are rejected. With `--all-native`, returned scenario names must match the canonical native registry exactly: absent, extra, failed, missing-dependency, or skipped results fail the gate. Matching wheels are built when neither `--wheel` nor `--meshcore-wheel` is supplied; prebuilt artifacts must be supplied together and have matching release versions.

External engines remain explicit optional comparisons. For example, `--scenario gmsh` or `--scenario planar-bands` runs the Gmsh comparison and reports missing dependencies separately; neither is part of the engine-free native gate. Native routes do not fall back to external geometry engines. Source approximations, exact CAD definitions, and mesh realizations retain distinct identities and fidelity claims.

Continuity is source-owned. Built-in analytic carriers declare C1 continuity
over valid parameter ranges; custom `AbstractCurve` and `AbstractSurfacePatch`
implementations must establish it explicitly rather than inherit a positive
claim. C1 continuity alone does not establish regularity or injectivity of the
physical map. Exact rational and radical source coefficients retain their
algebraic identity in fingerprints without rounding them to display values.

Native CAD operations are available through `phydrax.geometry.brep`:
`boolean_brep`, `sew_brep`, `brep_fillet`, `brep_chamfer`, `brep_offset`,
and `brep_defeature`, together with their typed results and failure evidence.
Assembly definitions and uses are authored through `BRepAssemblyGraph`;
occurrences retain exact placements and are expanded only when needed.
Boolean operations return source geometry without generating triangles unless
`BRepBooleanPolicy(tessellation=...)` explicitly requests a realization.

Supported native analytic intersections, including unequal-radius
perpendicular cylinders, preserve coupled source-defined branches even when
the curves are non-rational. Face-cell construction retains their source
occurrence lineage.

Fully covered exact equal-radius perpendicular cylinders are either proved
disjoint by normal separation or returned as typed singular unresolved regions.

T-spline B-rep intersections use cell-local rational Bézier source pieces and
native interval enclosures. Transverse graph branches against planes, cylinders,
spheres, NURBS surfaces, and other T-spline surfaces retain their source
`IntersectionCurve` and Krawczyk evidence, so CAD trimming and native surface
meshing consume the original face charts rather than fitted polylines. Tangent,
singular, or non-graph cases return typed unresolved parameter regions instead
of publishing sampled intersection geometry.

For bounded-root assembly vertices, affine placement transforms the stored
numeric source realization; the source-root enclosure remains separate
evidence, not a replacement point.

The focused cylinder-intersection and bounded-root placement checks are:

```sh
pytest -q \
  tests/unit/geometry/test_brep_boolean.py::test_nonrational_quadric_boolean_publishes_complete_branches_and_pcurves \
  tests/unit/geometry/test_meshing_domain.py::test_brep_bounded_root_placement_preserves_numeric_source_realization
```

Native B-Rep tessellation is an optional derived realization. A
`BRepTessellationPolicy` requests a physical linear chord-deflection bound and
a maximum continuous oriented-normal angular bound in radians;
`trim_samples_per_edge` is a minimum boundary scheduling density, never a cap.
Native surface generation preserves face IDs, exact UV charts, source revision,
and occurrence-specific ancestry. Continuous source-to-triangle bounds are
checked before publication; source CAD identity remains unchanged, and
unresolved or unsupported carriers refuse with an explicit reason rather than
returning sampled triangles as certified.

For volume-boundary fidelity, source UV remains two-dimensional while target
corners retain the full three-dimensional parent-reference coordinates and
cell-local face index. Revalidation maps those exact corners through the
selected local reference face; polyhedral targets require explicit actual-face
triangle ancestry.

For a direct smoke recipe, run:

```sh
pytest -q tests/unit/geometry/test_brep_tessellation.py::test_direct_native_tessellation_smoke_and_source_identity
```

`OffsetCurve` retains its base parameterization and exact signed distance.
Positive distance uses the right normal for planar curves, or tangent crossed
with the authored direction for space curves. A vanishing normal cannot be
certified as a regular offset. `RuledSurface(first, second)` uses a shared rail
parameter and linear interpolation between the two three-dimensional rails;
its transverse parameter lies in [0, 1].

CAD authoring points, vectors, and polygon vertices accept ordinary tuple/list
inputs as well as arrays; their owners perform the same one-time conversion and
validation. Explicit `CellBlock` global IDs likewise accept tuples without
changing the canonical block identity.

Numerical offset distances remain differentiable. Publishing a concrete edited
distance as a new source requires constructing a new carrier so its exact
distance and numerical realization agree. Structurally affine offset chains
retain all authored operations while evaluating iteratively; they do not
require nested derivative stacks or an arbitrary nesting limit. Algebraic
p-curve recovery uses the actual source surface map and does not invent a
finite bounding rectangle for an unbounded plane.
Endpoint validation reuses the exact affine source coefficients, including
rational-valued radical expressions from normal normalization. It does not
replace a deep offset chain with sampled endpoints.
Native-period endpoint binding covers both isoparametric directions of an offset
revolution. Conic and B-spline endpoint frames preserve exact real coefficients,
including rational-valued radical norms and nonrational exact radii.
Endpoint-to-vertex gap limits are checked against outward distance bounds; a
finite bound is not an exact-incidence proof. Root and source bindings remain
independently validated.
Meshing-domain admission replays the shared BRep vertex's exact UV endpoint
witnesses when numerical trim endpoints differ. A closed circular trim therefore
does not need a positive welding tolerance to accommodate the floating-point
realization of one full turn. This proof is available only after validating the
domain's source surface, p-curve, traversal, and occurrence bindings.
Supported parallel-cylinder chamfers retain constructive spatial intersection
roots at cap endpoints, together with the original conic-root rail and p-curve
bindings. Their endpoint incidence is established from those sources, not from
overlapping numerical point boxes. Exact single-coordinate source equations
tighten ruled-surface chart bounds without clipping them to a requested domain
or relaxing the geometric tolerance.
Closed-cylinder and closed-cone fillets retain exact conic-root torus trims,
including algebraic source centers and radii, instead of treating rounded
coefficients or quarter-turn angles as exact source data. The fixed-topology
radius map realizes affine and periodic p-curves numerically; cap endpoint
selection is prepared from exact source bounds before JAX tracing. Compiled
radius derivatives preserve the cap height while changing the contact-circle
radius. Chart and orientation validation remains active during realization.
Closed-sphere chamfers retain the seam's conic angular root and the exact
arc-length-setback/radius phase shift. Their ruled-surface contact rail is a
mapped curve on the authoritative sphere, not a circle inferred from a rounded
latitude. Shared endpoint admission requires exact line incidence and complete
containment in the unique source intersection; overlapping boxes are insufficient.
Face chart bounds use the contact enclosure, not the larger root-isolation domain.
Periodic mapped-curve lifts retain chart-image containment and require a full
native-period region before admitting a shifted periodic representative.
That region must use the patch's declared period; a nearby floating-point span
does not acquire full-period authority from a rounding enclosure.
Branch endpoint binding requires complete source-chart membership, not just a
matching projected parameter. Exact p-curve correspondence establishes UV
provenance; periodic sheet discrimination then retains the authored representative.
Arc-local scalar reparameterization does not change the root's source UV frame.
CAD import stages retain the proved face-gap bounds over each declared parameter
box and refuse a requested gap smaller than those bounds.
An affine B-spline p-curve can use the same exact lift and gap proofs as a line
only after a whole-span polynomial identity proves the common parameterization.
The original spline remains the authoritative carrier. Equal endpoint positions
or equal geometric images alone do not establish pointwise edge/face agreement.
Rational-valued radical coefficients use the same canonical coefficient payload
as their exact rational values. Genuinely irrational coefficients remain distinct
from their rounded numerical realizations. Complete carrier identity also binds
its declared structure and numerical arrays; external interchange that changes
those representations does not promise an unchanged source revision.

`patch.bounding_box(parameter_box)` validates the source UV rectangle and returns
outward physical bounds. It uses the canonical algebraic source enclosure where
available; interval evaluation of numerical coefficients is admitted only when
those coefficients represent the source exactly. Unsupported exact coefficient
families refuse rather than substitute sampled extrema.

`brep_mass_properties(model, subdivisions=8)` computes source-defined volume
enclosures in the cube of the model's length unit, separately for each solid
occurrence. `subdivisions` initializes the interval integration grid; it is not
an accuracy claim or a work ceiling. Consume `volume_enclosures` together with
the per-occurrence `status` and `reasons`. The `volume` midpoint is a numerical
convenience, not an independent certificate. Unsupported or unresolved source
integration retains an infinite enclosure and a NaN midpoint rather than
substituting a sampled or tessellated volume.

## Native interval evaluation for source charts

`prepare_interval_function(function, dimension)` defaults to `execution="auto"`.
Supported batched jaxprs lower once to the native meshcore interval executor;
the prepared function then evaluates each batch without Python interpretation
of its jaxpr equations. `PreparedIntervalFunction.execution` reports whether
that preparation selected `"native"` or `"host"`. The host evaluator remains
available with `execution="host"` as an independent enclosure reference.

Use `execution="native"` to require a complete native lowering. Unsupported
JAX primitives raise `IntervalExtensionError` during preparation rather than
silently running through the host interpreter. `auto` retains the host route
for jaxprs that contain primitives without native interval extensions. Native
polynomial, sine/cosine, square-root, and derivative programs use outward
rounding and are checked against the host enclosure contract.

One local cylinder-chart map measurement evaluated 4,096 interval boxes with
`CylinderPatch.evaluate` at `batch_capacity=64`. After two warm-ups, the median
of seven timed passes measured 0.681 microseconds per box on the native route
and 3.356 microseconds per box on the host route (about 4.9 times lower in this
sample). This measures the chart map's interval execution used by certification,
not end-to-end certification; it excludes tracing and preparation and is not a
benchmark claim or a performance guarantee.

## Placement evidence, normal ranges, and fitted intersections

STEP imports may request `CadImportPolicy.maximum_placement_deviation` to admit
rounded source frames. The reader retains the source pose bound on assembly
uses, composes it to each occurrence path, and multiplies the rotation bound by
an outward local-solid extent before publishing `BRepPlacementEvidence`. The
reported point-deviation bound includes the translation error; a request larger
than the declared limit is refused. A zero limit continues to require exact
placement representation.

`brep_normal_ranges(model, direction, draft_angle=..., angular_tolerance=...)`
returns one face-scoped angle enclosure in radians. It uses continuous surface
derivative or analytic normal enclosures over the source trims, refining cells
that can affect an angle extremum or requested threshold. Revolved line and
circle profiles lying in a plane through their axis use the meridian normal,
which stays regular where a profile meets the axis (disc caps, cone apices,
sphere poles). Cells whose normal is undefined, such as a profile crossing the
axis, are subdivided to 1/256 of the face parameter range and then reported in
`unresolved_cells` with spatial boxes in `singular_points`; they contribute the
full `[0, pi]` range and are never replaced by a point sample.

`CadExportPolicy(intersection_approximation=CadCurveFitPolicy(...))` requests a
fitted B-spline edge and its two source p-curves for an otherwise unrepresented
intersection branch. The policy's `tolerance` bounds spatial error and both
surface lifts; `parameter_tolerance` separately bounds UV error and defaults to
the same scalar. Export proceeds only when continuous whole-domain bounds cover
those errors, branch-local axes, nonlocal separation, and relevant trim
separation. The export report records this as a geometry-changing approximation
loss. Without the explicit fit policy, the non-rational intersection branch is
refused. A fitted carrier only interpolates its endpoints, so the vertices it
touches are no longer exact source roots in the exported geometry; their
root-supported endpoint bindings are released with them.

## Periodic seams, poles, and recovered p-curves

Closed periodic faces round-trip through STEP, IGES, and BRep text without
rounding their native turn. A seam used on both chart sheets, a whole-turn ring,
and a collapsed cap pole are recognized from exact period evidence: an affine
isoline at an exact multiple of the binary period. IGES has no line p-curve
entity, so a degree-one B-spline p-curve counts only when it is exactly that
isoline over the authored edge range. Each coedge of a shared full-turn edge
binds its endpoints to its own face, using the declared period of that face's
surface or else the p-curve's own native turn. Pole vertices keep their file
coordinates as exact source values.

When a file omits p-curves, import recovers only exact source maps. A circle
that is a height section of a cylinder, with the cylinder's own phase, maps to
the isoline `(t, h)`. A circle in a plane exactly parallel to a planar face maps
to its orthogonal chart circle. Its normal offset is not hidden: publication
certifies it as a face gap that must lie within the import tolerance. Other
combinations are refused rather than fitted.
