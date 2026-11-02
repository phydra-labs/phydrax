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

Sketch root preparation propagates fixed points, fixed circle radii,
coincidence, horizontal and vertical coordinates, and midpoints only when the
represented midpoint is exact. It retains every original residual. A reduced
candidate is accepted only when both the complete residual and the native
full-system least-squares stationarity test meet the declared tolerance.
Otherwise the same native least-squares owner solves the unreduced problem from
the original reference layout. In particular, conflicting weighted constraints are not
silently treated as hard equalities. This preparation does not prove root
uniqueness, and no gauge or least-motion objective is added.
For two positive distance loci with distinct fixed centers, an exact rational
discriminant and nonzero reference orientation establish a transversal
algebraic branch. Its floating realization is checked against the full
residual. Tangent or unresolved constructions are seeds only; this local branch
evidence does not establish global root uniqueness.
When a constructed branch is exactly rational, its algebraic point, not its
rounded numerical realization, may anchor a subsequent two-circle
construction. Non-rational constructed points are not used as exact anchors.
For an axis-aligned line through a fixed endpoint and a positive distance to a
fixed center, an exact discriminant and nonzero reference-side witness
similarly identify a transversal branch. Other line/circle relations remain in
the full residual.
`SketchSolution.preparation` retains the reference, seed, fixed-coordinate
facts, exact branch witnesses and whether the reduced root was accepted.
Call its `verify` method with the sketch, dimension state and solve tolerance
to replay its premises; this is separate from `root_evidence` and does not
establish global uniqueness.
For distance-triangulated components, `preparation.rigid_clusters` records an
exact positive-discriminant witness for every added point and a nonzero
reference orientation. Rigidity is conditional on simultaneous satisfaction of
the recorded distances; this is not a root-existence certificate. A cluster
with exactly one fixed anchor can be reduced to one orientation-preserving
angle when every declared incident distance and every reference offset is
exactly represented, its branch witness is valid, and it shares no other fixed
or constructed coordinates. All original residuals remain, and a failed reduced
root attempt falls back to the original full system. Other clusters remain
numerical; no gauge is imposed.

`SketchBranchWitness` identifies an ordered triple of points.
`SketchContinuationReference` retains a converged layout and explicitly
declared witnesses; `continue_sketch` checks their exact represented-coordinate
orientation signs and numerical rank after a dimension edit. A rejected edit
retains the reference unchanged. Matching endpoint signs do not prove that the
continuous edit path was event-free or that the root is globally unique.
`SketchContinuationResult.dimension_sensitivity`, `dimension_jvp`, and
`dimension_vjp` require an accepted transition, independently replay its
one-step checkpoint, and validate endpoint evidence before applying the
fixed-branch implicit derivative. Standalone
`Sketch.dimension_sensitivity` instead describes the local derivative of its
supplied regular root, even if that root was an unaccepted diagnostic attempt.
Derivative actions refuse a represented line-angle branch cut, collapsed
carrier, rank change, or rejected branch transition rather than reporting a
smooth derivative there.
Sketch problem identity includes constraint incidence and branch selectors,
including arc topology. Previous identities are not reused for the changed
representation; compiler keys likewise follow the current residual code and
captured values.
Sketch arcs retain directed endpoint and center incidence, including selected
major sweeps. Exact native profile lowering requires exact equality of the
represented endpoint radii and zero proved endpoint gaps; its planar model
retains bound native conic roots on the incident edge and coedge carriers.
Numerical residual convergence alone is not promoted to an exact circular
profile.

A `SketchContinuationReference` retains its source sketch as well as the
accepted root. For portable restoration, put it in a lifecycle `Composition`
entry with role `model-state` and persist the composition through the existing
model-archive API. Restoration reconstructs the sketch and independently replays
the retained one-step solve checkpoint, then checks the accepted output,
preparation, root, and branch evidence. This detects output changes against an
unchanged checkpoint; it does not authenticate a wholesale replacement of both
the source and checkpoint. Restoration does not require the original Python
objects.
Run `python examples/native_sketch_continuation.py` for the complete witnessed
dimension edit and archive-restoration workflow.

Run `python tools/sketch_preparation_benchmarks.py --counts 4 8 16` only
without competing benchmarks. It records growing axis-constraint and
distance-triangulated cluster scenarios, with preparation, Jacobian work, solve
time, traced retained host memory, and prepared array bytes separately. The
measurements are observations, not correctness or speed claims.

Exact radical normalization uses a deterministic 64-bit primality decision for
word-sized prime residues, including residues left after removing small factors.
Larger integers retain complete exact factorization without an operation quota;
no general complexity bound is claimed.

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

`edit_brep_occurrence_placement` selects an exact assembly occurrence path and
returns a new source model and occurrence correspondence. Uses are identified
within their owning definition; a path edit copies shared ancestry rather than
moving other instances. Exact host placements are not differentiable design
parameters. A changed placement invalidates the old tessellation and other
world-dependent evidence. Rebuild projection, contact, embedding and affected
coupling against the new coordinates before accepting the result.
The correspondence preserves topological incidence; a reflection's world
handedness change is recorded separately and does not replace normal or
Jacobian revalidation.

For repeated motion, an in-hand `SourcePreparation` can create a preparation
for the new meshing domain with `for_placement_domain`. It carries only exact
local spline spans when the source dependencies and occurrence inventory match;
placed spans and world-dependent proofs are constructed anew. Unknown or
changed local dependencies take the cold path.

The geometry realization benchmark includes repeated spline-backed assembly
motion and records local span extraction and reuse separately from placement,
world tessellation and proof, contact search, interface coupling and total
work. Run performance measurements without competing benchmarks.

Supported native analytic intersections, including unequal-radius
perpendicular cylinders, preserve coupled source-defined branches even when
the curves are non-rational. Face-cell construction retains their source
occurrence lineage.

Fully covered exact equal-radius perpendicular cylinders are either proved
disjoint by normal separation or returned as typed singular unresolved regions.

Planes that cut a straight cylinder only partly publish exact boundary branches.
A plane normal to the axis yields the arcs its face rectangle leaves, including
arcs across the cylinder's seam, and a plane parallel to the axis yields trimmed
generator lines; every branch end fixes the exact edge value it lies on.
Exterior tangencies are exact typed contacts rather than branches: a tangent
plane along a generator, a face edge touching the circle from outside the face,
and any touch outside a trim or edge range. A face edge touching the circle from
inside the face is an exact double root with multiplicity evidence: the arc
gets one node at the contact, and the edge vertex there keeps the replayable
tangency through the CAD archive. A circle through a face corner ends its arcs
on that corner's source vertex, and a source edge lying on a cylinder generator
is one edge shared with the cylinder through its exact lift instead of a
branch. A plane holding a cone's axis direction cuts exact trimmed section arcs
from it, circle and straight edges meet planes and cones at exact radical
roots, and branches whose partner faces are coaxial straight revolutions (a
cylinder and a cone sharing a circle edge) join at that edge's roots once the
faces are proved to meet only on it.

Supported trimmed intersections of curved faces from different operands can be
traced as graphs over a straight revolution's angle. Qualified examples include
cylinder-cylinder, cylinder-cone, cone-cone, and sphere against a cylinder or
cone; admission still requires the exact source and graph certificates. Each
trim crossing is a Krawczyk-certified node on the exact trim
coordinate, bound to the curve-surface root of that trim edge, and each seam
crossing of the other face is a node with an exact deck. The remaining refusals
return typed unresolved regions: a turning point of the traced graph inside the
trims, junctions whose curved partner faces are not coaxial straight
revolutions, planes through a cone's axis, and tangent contacts of a section
with a trim.

Discovery also proves separation of coaxial cylinder and torus supports from
their exact radial ranges. This excludes disjoint annular surfaces whose
Cartesian boxes overlap, without assuming that faces of the same imported
solid are disjoint and without introducing a general torus-intersection route.

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

For B-spline and T-spline carriers, each triangle's normal bound comes from the
exact Bernstein form of the normal direction `N_u x N_v` over every knot span
the triangle meets (`N_u`, `N_v` are the derivative numerators over the positive
denominator). Every source normal on a span sub-box is a convex combination of
the restricted control vectors, so the largest angle between a control vector
and the triangle normal bounds the deviation when all of them are acute. This
keeps the tangent components coupled; the componentwise Jacobian-box product is
retained as a second enclosure and the smaller upper bound is reported. The
span enclosures of one cell are computed once and shared by its chord and
normal bounds.

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
For a rooted sphere-revolution pole, the offset preserves the conic-rooted
latitude carrier and rebinds its native-period endpoint witnesses to the
offset surface; the rounded latitude is not used as exact pole authority.
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

`brep_mass_properties(model, relative_tolerance=1e-6)` encloses each solid
occurrence's volume, in the cube of the model's length unit, as the flux of
`x / 3` through its source faces (`x . (S_u x S_v) / 3` per chart). Affine faces
with line p-curves and polynomial B-spline faces spanning their chart integrate
exactly in rationals (Bernstein triple products); planar faces with other
p-curves integrate their Green area along the p-curves; other curved faces
integrate over parameter cells split at every smoothness break, either as
Green strips swept from the chart edge or as chart cells classified against
the source trim. Inside a smooth cell the second-order midpoint/Hessian
enclosure `f(c) A + sum f_ii A h_i^2 / 24 +- rad(f_uv) A h_u h_v / 16` applies
(intersected with `f(box) A`); trim-crossing chart cells hold their
zero-extended range. Certified edge-face and vertex gaps add the flux of their
closing strips and fans. Batched rounds refine the widest cells until the
half-width is at most `relative_tolerance` times the positive lower bound.
Cell programs run on the native interval executor when they lower completely,
otherwise on the host reference interpreter. Consume `volume_enclosures` with
the per-occurrence `status`, `reasons` and `exact`; the `volume` midpoint is a
numerical convenience. Unsupported carriers or trims, and requests below the
binary64 parameter or rounding resolution, keep an infinite certified enclosure
and a NaN midpoint; `reached_enclosures` keeps the tightest enclosure reached as
evidence. `BRepSource` reports the same owner's result as `interior_mass`
(`ExactMass` for exact volumes, otherwise `EstimatedMass` with the certified
half-width as uncertainty, at `measure_relative_tolerance`) and raises
`BRepMeasureRefusal` carrying the full evidence when an occurrence is refused.

`brep_shell_volume_sign(model, shell)` certifies the orientation of one closed
shell with the same flux enclosure, refined until it excludes zero. A shell
whose sign stays unresolved raises `BRepMeasureRefusal` rather than being
oriented from a sample.

## Native interval evaluation for source charts

`prepare_interval_function(function, dimension)` defaults to `execution="host"`,
the independent host enclosure reference. With `execution="native"`, supported
batched jaxprs lower once to the native meshcore interval executor, and the
prepared function then evaluates each batch without Python interpretation of
its jaxpr equations; `PreparedIntervalFunction.native_program` holds the
lowered program and `execution_reason` records the selection.

`execution="native"` requires a complete native lowering. Unsupported JAX
primitives raise `IntervalExtensionError` during preparation rather than
silently running through the host interpreter. Native polynomial, sine/cosine,
square-root, and derivative programs use outward rounding and are checked
against the host enclosure contract.

Direct and affine-placed native torus charts bind outward intervals for their
authored exact coefficients when preparing point, Jacobian, and Hessian
programs. Source covers and interpolation fidelity also include the gap
between exact source corner enclosures and their numerical mesh realizations;
nonbinary radii or placements are not silently replaced by rounded shadows.

One local cylinder-chart map measurement evaluated 4,096 interval boxes with
`CylinderPatch.evaluate` at `batch_capacity=64`. After two warm-ups, the median
of seven timed passes measured 0.681 microseconds per box on the native route
and 3.356 microseconds per box on the host route (about 4.9 times lower in this
sample). This measures the chart map's interval execution used by certification,
not end-to-end certification; it excludes tracing and preparation and is not a
benchmark claim or a performance guarantee. End to end, the source-fidelity,
domain-coverage, implicit-chart, and interval tests that use these enclosures
ran at parity in one local comparison (76 tests: 78.0 s host, 80.1 s native
with a warm compilation cache), so host remains the default.

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
sphere poles). A cell whose normal is undefined is subdivided, at knot lines
first, until a positive lower bound on `|S_u x S_v|` makes it regular or it is
certified to contain a singular point: it meets a degenerate edge p-curve
whose exact pole collapse the model proved (or a collapsed clamped spline
boundary row), or a Krawczyk box isolates the unique zero of two cross-product
components at an exact rational point where the third vanishes exactly.
Certified cells are returned in `singular_cells` with `singularities` evidence
and exact `singular_points` and contribute the full `[0, pi]` range; regular
cells are then not refined further, because they cannot change that range.
Cells still undecided at binary64 parameter resolution are returned in
`unresolved_cells` with a typed `unresolved_reasons` entry. No cell is decided
by a point sample.

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

B-spline pole identities (`control_point_ids`) are format-dependent. BRep text
names each pole by its u-major/v-fast record ordinal, and STEP and IGES carry no
pole identity. When re-importing the written record would change a surface's
pole identities, the export report declares a `face:<i>.pole-identity` loss.

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

Signed shell integration belongs to the native measure implementation and is
shared by Boolean shell nesting and interchange orientation repair. Its
directed interval sign proof is independent of a sampled volume estimate.
For an exact straight cylinder, the Green strip's angular integral is analytic;
the remaining source p-curve integral is interval-enclosed on individual
certified intersection charts, retaining the requested volume accuracy.

STEP retains shared conic basis references as one curve-bank definition and
exports a shared bank row as one referenced conic. Independently authored
coincident circles remain distinct. A closed chain on that same definition,
with consistent oriented vertex incidence and contiguous parameter intervals
covering exactly one native turn, retains symbolic endpoints at the aggregate
seam. Intermediate decimal trims are unchanged. An open chain, a partial or
multiple turn, mismatched incidence, or a gap between intervals does not acquire
native-turn authority; numerical proximity to a multiple of pi is not evidence.
This is the existing closed-edge representation convention applied to its
authored subdivisions, not a tolerance repair. Preserving shared references and
seam roots changes the decoded representation and therefore may change its
model/source revision; it does not merge distinct source occurrences or promise
historical representation IDs.

Boolean chart arrangements use the same exact affine-source recognizer as
curve/surface correspondence. A clamped degree-one B-spline with equal
weights is therefore an affine line, including its actual knot-parameter
scale; unequal weights are not silently treated as affine parameterization.

`repair_brep_source` is an explicit geometry- and interpretation-changing
operation, separate from exact sewing and STEP import repair. It retains the
input model and returns a separately identified source revision with entity
correspondence and certified displacement evidence. Its completion policy
identifies the bounded interpretation of gapped source trims, including the
authored incidence, sheets, and winding; whole-region displacement is relative
to that completed interpretation, not an asserted filled region of an
otherwise undefined gapped loop. The tolerance is a caller-supplied upper
bound in the model's length unit; use the original source's admitted tolerance,
not a larger Boolean or meshing tolerance. Repair refuses unsupported,
ambiguous, unproved, or out-of-budget constructions instead of welding nearby
entities or silently fitting curves. The repaired result is not a lossless
import of the original source.

When a file omits p-curves, import recovers only exact source maps. A circle
that is a height section of a cylinder, with the cylinder's own phase, maps to
the isoline `(t, h)`. A circle in a plane exactly parallel to a planar face maps
to its orthogonal chart circle. Its normal offset is not hidden: publication
certifies it as a face gap that must lie within the import tolerance. Other
combinations are refused rather than fitted.

Torus faces recover latitude and meridian circles. A circle whose rounded
center, axis, and radius match a coordinate line of the torus maps to that
affine isoline, including rotated or shifted chart phases at exact quarter
turns; publication then certifies the face gap of the recovered p-curve, so
decimal rounding in the file is bounded rather than assumed away. Torus seam
edges lift exactly onto the closing sheet, and p-curves that differ from their
source chart by whole native periods are accepted as the same image.
