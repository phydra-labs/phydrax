# File resources, interchange, and native persistence

Phydrax separates physical file handling, external-format conversion, and native
scientific persistence. A file extension is never sufficient evidence of scientific
meaning.

## Ownership

- `phydrax.interchange` owns admission of external bytes and resource sets, explicit
  format profiles, provider boundaries, and semantic conversion evidence.
- `phydrax.lifecycle` owns native checkpoints, results, model records, revisions,
  repositories, retention, and rollback.
- `phydrax.export` owns deployment products such as ONNX and IREE while using the
  common publication substrate.
- The canonical JSON-plus-NPY array archive is a native container implementation. It
  is not a generic replacement for standards such as HDF5, NetCDF, VTK, DICOM, or
  SEG-Y.

VTK, VTU, XDMF, NIfTI, Touchstone, and similar products are external projections.
They must not be the only copy of restart state.

## Resource carriers

`BoundedResource` retains a small exact resource in memory. `open_bounded_resource`
provides a context-managed seekable stream for larger single files while retaining an
exact `ResourceManifest`. Both routes:

- walk beneath an explicit trusted root by directory descriptor;
- reject traversal, symbolic links, and special files;
- enforce encoded-byte bounds;
- hash the exact source bytes;
- detect file or path-component changes during admission.

`BoundedResourceSet` and `ResourceSetManifest` extend the same rules to directory
stores and multi-file artifacts. Member paths are canonical, regular, deterministic,
and individually checksummed. Aggregate limits cover bytes, members, and nesting.

`BoundedArchive` preflights external ZIP-compatible containers. It rejects duplicate
canonical names, traversal, links, encryption, unsupported compression, excessive
expansion ratios, and member or aggregate size violations. This route is used for
external containers such as FMUs and NPZ-shaped inputs. Native array archives retain
their stricter canonical encoding.

## Structural decoders

The interchange layer provides reusable structural admission without assigning domain
meaning:

- finite duplicate-free JSON;
- bounded UTF-8 or ASCII text;
- XML with DTD and entity semantics disabled;
- pickle-free NPY and NPZ with header, dtype, shape, rank, and element preflight;
- HDF5 object, link, dataset, dtype, filter, attribute, and logical-byte inspection.

NPY decoding returns a read-only view backed by the immutable resource bytes;
request an explicit NumPy copy if mutation is required. NPZ decoding preflights
the complete inventory, checksums, headers, and aggregate element limits before
decoding members individually. It does not retain a second mapping of every
decompressed member.

Native array archives hash and count canonical NPY payloads before publication,
then stream the same encoding into stored ZIP members and verify each second-pass
digest. Source mutation refuses publication without replacing the destination.
Archive bytes, collection digests, and deterministic member metadata are unchanged.
Lifecycle NPZ exports stage directly to a file; explicitly authorized full support
bundles use a bounded, descriptor-verified temporary snapshot rather than retaining
the complete archive as bytes. Checkpoint restore hashes chunks incrementally through
a one-MiB disk-spilling spool, loads each admitted shard once, and returns read-only
host arrays.

Format adapters remain responsible for units, frames, axes, topology, field
association, validity, uncertainty, support, status, and provenance.

## Publication

External writers stage output, close it, enforce encoded bounds, reopen and validate
the actual encoded representation, synchronize it, and only then publish it.

`PublicationReceipt` records the durable byte identity of one file.
`ResourceSetPublicationReceipt` records every member and the aggregate identity of one
immutable bundle generation. Bundle publication is exclusive by default. Explicit
replacement uses a native atomic directory exchange and fails closed on hosts without
that capability; the displaced generation is removed only after the exchange commits.

This distinction matters for VMFB bundles, operator artifacts, XDMF/HDF5 products,
PVD/PVTU collections, SONATA, Zarr, and similar resource sets. Atomic replacement of
individual files is not a transaction over a bundle.

## Semantic conversion evidence

`AdapterReport` describes scientific conversion, not filesystem behavior. It records:

- source and target profiles;
- coordinate and representation mappings;
- preserved capabilities and fields;
- assumptions;
- explicit losses and waivers;
- refusal or failure status.

Resource manifests and publication receipts compose with the report but do not replace
it.

## Format capabilities

`format_capabilities()` returns a deterministic immutable catalog of implemented
profiles. Each `FormatCapability` declares:

- an exact `AdapterFormatProfile`;
- scientific domain;
- extension hints;
- carrier kind;
- implemented directions;
- optional provider dependency;
- preserved capabilities and known losses.

The catalog is introspection only. It is deliberately not a plugin registry or a
`load(path)` dispatcher. Ambiguous formats require an explicit profile. In particular,
`.msh`, `.xml`, `.h5`, and `.json` do not identify one scientific contract.

## Native CAD and explicit external interoperability

`read_cad(path, policy, trusted_root=...)` performs native suffix admission for
STEP (`.step`/`.stp`), IGES (`.iges`/`.igs`) and the supported externally defined
OCCT BRep text profiles (`.brep`/`.brp`). The selected reader owns its entity
graph, malformed/reference/depth/byte budgets, units, placements and coverage
report. The extension is a routing hint, not validation.

```python
policy = phx.interchange.CadImportPolicy(
    phx.SpatialCoordinateContract.si(),
    phx.interchange.ResourceLimits(
        max_bytes=16 * 1024 * 1024,
        max_depth=64,
        max_nodes=100_000,
        max_attributes=1_000_000,
        max_losses=1024,
    ),
)
imported = phx.interchange.read_cad(
    "part.step", policy, trusted_root="/trusted/cad"
)
model = imported.model
print(imported.coverage, imported.report)
```

STEP and IGES units come from their format graphs; `read_cad` refuses a source
unit override. BRep text has no embedded length unit and requires the caller's
`source_length_unit`. A native file decoder does not invoke OCCT algorithms.
`BRepModel` retains exact source identity separately from derived tessellation
identity, and `MeshingDomain.from_brep(model)` consumes exact native carriers,
not the query triangulation.

`save_brep_archive`/`load_brep_archive` own canonical native CAD lifecycle
persistence. Do not use an external `.brep` extension to imply that archive
format. Format-specific `read_step`, `read_iges`, `read_brep_text` and their
writers remain explicit entry points when the format profile is already known.

Explicit intersection-curve exports use a self-describing bounded
representation, not a relaxed import of the original source. The descriptor
retains source and approximation identities, unchanged export distance and
parameter limits, and a certified surface-Jacobian Lipschitz bound over the
whole source-to-fit UV homotopy. Its declared carrier-gap allowance follows
`distance_limit + surface_jacobian_lipschitz * parameter_limit`; achieved
residuals cannot enlarge those frozen limits. STEP carries that length bound
as its representation-context uncertainty together with the complete
descriptor. IGES carries the descriptor separately in Start text: Global
parameter 19 is coordinate discernibility (minimum resolution), not a
general edge-to-surface gap allowance.

Readers validate the complete declaration and independently certify the
actual emitted carrier-to-surface gaps before admitting stored gap bounds.
A marker, source provenance, or larger ordinary format uncertainty alone
never authorizes bounded admission. Import and export report the explicit
interpretation loss and artifact SHA under a new representation identity;
the original source's `1e-8` default relative correspondence policy remains
unchanged.

Within one admitted coordinate request, format-independent export preparations,
the complete bounded-representation descriptor, and independently earned
whole-coedge bounds may be reused for the same source owner and complete
fitting/tolerance premises. Preparation and descriptor reuse freshly bind the
actual carrier arrays, model/geometry state, host-side report, policy, and
approximation evidence; declared IDs alone do not authorize reuse. Their retained
storage remains charged to that request. A new request, restored owner, changed
source, or changed export policy does not inherit an unearned proof. Every emitted
artifact still receives its independent final import correspondence proof.
For a real curve carrier the unused fallback vertex point is not a residual-proof
premise. For a collapsed point carrier, that point remains content-bound and a
changed point requires a fresh proof.
Correspondence reuse canonicalizes a decomposition box only after a fresh
whole-range outward p-curve enclosure is entirely contained in the supplied box.
Overlap alone keeps the supplied box distinct. Generating checks and new proof
worksets still use the original supplied box. A retained complete residual bound
must satisfy the current requested tolerance; partial outcomes are never reused.
The full live carrier definitions, closed range, algorithm and interval cap stay
bound to the request-local cache.
Within each emitted-carrier correspondence query, rational quotient jets share
their lower-order hulls, and the first- and second-order chain rules share the
same full-interval surface Jacobian. Center value/derivative and full-interval
first/second derivative bounds remain independently present in the unchanged
mean-value/Taylor proof; no state is borrowed from another artifact or proof.
An outward center residual interval wholly outside the requested tolerance in
any component disproves the whole-domain bound. This early refusal is used only
when the supporting surface piece contains the complete center UV enclosure;
mere overlap cannot authorize it. It produces unresolved evidence, never a
positive certificate, and leaves the original tolerance and work caps unchanged.
When the three fitted carriers have identical knots and identical nominal and
outward-enclosed homogeneous denominator coefficients on a source span, their
numerator columns share one componentwise rational-jet evaluation. The resulting
3D/UV slices preserve the individual directed-rounding bounds. Unequal knots,
domains, or denominator coefficients retain the independent evaluation route;
the private grouped coefficients live only within that certificate and their
storage is admitted before construction.

Mapped STEP assemblies retain shared native solid definitions. Nested
`MAPPED_ITEM`/`REPRESENTATION_MAP` placements compose in occurrence order.
Mapping origins, operators, assembly usage frames, inverses, and compositions
all obey the same zero-deviation default: certified proper orthonormal
binary64 axes and unit scale. No scaled/reflected operator is rounded into a
rigid placement. Rounded oblique source frames require the explicit near-rigid
declaration described below, including nominal quarter turns emitted with
inexact direction ratios by an external writer.


STEP imports retain authored product and representation-item names, RGB surface-fill
colors, and presentation layers in `CadImportResult.entity_metadata`.
`StepEntityMetadata` binds canonical `StepEntityLabels` rows to exact `BRepEntityId`
values. Unsupported presentation styles are ignored by design in the coverage
report; they never become guessed RGB colors. IGES and BRep text imports explicitly
set this STEP-specific field to `None`.

`select_step_entities(model, metadata, "name", "inlet")` produces the same
selected/rejected/unresolved record as native geometric selection. Names require
uniqueness by default and report all conflicting entities; colors and layers
select exact sets. Solid labels expand to faces through explicit topology.
The returned selection uses the common `bind_region` API and recipe naming keys
to bind native meshing controls. Pass `metadata=imported.entity_metadata` to
`write_step` to preserve labels, colors, and layers on export.

Presentation records are not part of the model identity payload: attaching or
changing metadata on the same model leaves `model_id` and entity IDs unchanged.
Editing labels in a source STEP file does change its whole-file source revision,
and consequently its model ID. Source provenance is not a geometry-only digest.

For an existing successful STEP import containing a unique authored `"inlet"`
face name, this lookup preserves all nonselected and unresolved identities.
The prerequisite is real metadata on `imported.model`; this is not a standalone
file fixture.

```python
metadata = imported.entity_metadata
if metadata is None:
    raise ValueError("The imported source must carry STEP entity metadata.")
inlet = phx.interchange.select_step_entities(
    imported.model, metadata, "name", "inlet", family="face"
)
assert len(inlet.selected) == 1
assert not inlet.unresolved
```

Use `inlet.bind_region(...)` with exhaustive recipe names rather than replacing
the source identity by a label string. A duplicate-name conflict retains all
matching IDs; only an explicit nonunique lookup requests the entire set.


`StepRepairPolicy` defaults to no repairs. Its explicit `add_missing_seams`,
`reverse_inverted_shells`, and `orient_face_senses` switches authorize only
geometry-preserving topology changes. STEP-specific readers accept
`repair_policy=...` and `coordinate_budget=...`; enabled repairs require an
explicit coordinate-enclosure budget. They never move a carrier, fit a repair,
or grow a tolerance. Unsupported and unresolved repair families refuse.
Every applied repair records its entity and certified reason in the adapter
report, whose status is `AdapterStatus.REPAIRED`.

The bounded default orientation audit uses certified signed shell-volume
enclosures and certified nesting roles. Outer shells require positive signed
volume; cavity shells require negative signed volume. Certified inverted
shells refuse with `inverted-shell` instead of silently importing inside-out
solids. When the bounded default audit cannot resolve orientation, the import
remains admitted and its report explicitly records `orientation_unresolved`;
an opt-in shell repair refuses rather than guessing.
The policy declares `audit_maximum_work_units` (default 2,000,000) and
`audit_maximum_memory_bytes` (default 16 MiB) for this bounded default audit.
They limit work and storage; they do not change a geometric tolerance.

`StepRepairPolicy.subdivisions` defaults to one. Missing-seam repair supports
coaxial full-circle cylinder bands and natural sphere/torus chart seams;
face-sense repair supports closed convex linear shells, not arbitrary curved
or nonconvex shells. Explicit repair budgets may reserve more subdivisions,
but they cannot turn unresolved sign/nesting into an accepted reversal.
The [API reference](api/interchange.md#exact-step-repairs-and-orientation-audit)
lists repair switches and bounded audit semantics.


STEP occurrence placements keep a zero displacement limit by default.
Set `CadImportPolicy.maximum_placement_deviation` in target length coordinates
and pass `coordinate_budget=...` to a STEP reader to admit rounded proper
frames. Direction ratios define the source frame through normalization and
orthogonal projection; rational arithmetic and directed square-root bounds
enclose that frame, including oblique axes. Composition and inverse poses
retain rotation and translation enclosures. Nonunit scale, reflecting frames,
and degenerate direction sets refuse.

`BRepOccurrence.placement_evidence` publishes entrywise affine-coefficient error
bounds, rotation and translation norm bounds, native carrier extent, and the point-displacement
bound. It is absent on zero-defect occurrences. The limit is checked against
every occurrence and never enlarged. Positive placement displacement is not
the same as authored edge/face incidence gaps in sewing: occurrence
materialization, Boolean combination, and sewing currently refuse with
`placement-deviation`, identifying the occurrence and its bound. Placement
queries remain available. Native archives preserve the evidence; external CAD
export refuses rather than dropping it.

Optional live OCCT shapes and independent engine comparisons use only
`phx.interchange.model_from_occt_shape`, `import_occt_brep`,
`persist_occt_shape`, `read_occt_shape`, and
`prepare_occt_projection`/`PreparedOcctProjection`. Install
`phydrax[cad-occt-interop]` deliberately for these external operations. They are
not native decoding/query fallbacks and are not exported by the geometry facade.

The live-shape comparison representation does not supply native exact edge and
p-curve authority. Before creating `BRepSource`, explicitly decode its published
artifact with `read_cad` and a unitful native import policy. The optional Gmsh
adapter exports that native model through the native BRep text writer in the
declared source coordinate unit. It binds external tags to native source rows
by geometry and boundary-incidence verification, recording model/revision and
original/export digests; equal entity counts or exploration order are not IDs.

`tools/check_native_dependency_isolation.py` installs the Phydrax wheel and its
required in-house `phydrax-meshcore` wheel into a fresh environment with ordinary
dependency resolution. Its isolated consumer rejects installed external engines
and meshio, round-trips native CAD codecs and persistence, checks native exact
box queries, and measures a CAD-derived native cell mesh. This is evidence for
the named executed workflows, not proof that every native numerical route or
every external CAD entity has been implemented.

The consumer's process guard inspects `subprocess`, `os.system`, `os.exec*`
and `os.posix_spawn` launches. It refuses a named external meshing engine
whether invoked directly, by resolved path, or through `env`, a shell `-c`
command, or a recognized launcher such as `nice`, `timeout`, `taskset`,
`systemd-run`, `mpirun` or `uv`. Launch text it cannot read literally (shell
expansions, `env -S`, shell standard input) refuses. Arguments of other
programs are data, and processes started later by an admitted program are not
observed; the guard is evidence for these launch forms, not a sandbox. The
installed source-build fingerprint must equal the staged checkout's, computed
over the same provider-source tree that runtime provider lookup selects.

SciPy remains permitted numerical infrastructure. Its statistics/distance
imports can load the bundled Qhull module, so the consumer blocks Qhull geometry
constructors rather than misclassifying a numerical module import as geometry
execution.

## DXF planar profiles

`read_dxf(path, policy, trusted_root=...)` and `decode_dxf_bytes(data, policy)`
read bounded ASCII drawings into a `DxfImportResult` containing a `PlanarProfile`,
its declared coordinate unit, and an `AdapterReport`. `DxfImportPolicy` declares
byte, group, and entity limits before parsing. `$INSUNITS` or the caller's
`source_length_unit` must declare the drawing unit; conflicting declarations
refuse rather than triggering implicit conversion.

Supported source entities are `LINE`, `ARC`, `CIRCLE`, `ELLIPSE`,
`LWPOLYLINE`, and control-point `SPLINE`. Bulges retain circular arcs, and
quadrantal ellipses retain rational quadratic conics. Spline controls, positive
weights, and clamped knots remain authored coefficients. Fit-point-only splines
refuse because interpolation would invent geometry. Non-XY entities, nonzero
widths or thickness, periodic unclamped splines, and unsupported geometry-bearing
entities refuse. Unreferenced sections and drawing annotations are accounted for
as ignored by design.

Loop closure retains authored vertices. Source topology certificates decide
orientation and containment; open or ambiguous loops refuse with their endpoints.
General-angle circles and ellipses retain analytic carriers and independently
certified endpoint displacements. `maximum_junction_gap=0` is the default. A
positive caller declaration admits only junctions within that fixed bound, reports
the displacement, and publishes recomputed B-Rep gap evidence during extrusion.
It never snaps a vertex, fits a curve, or grows the declaration.
Successful nonzero-gap imports retain an `AdapterLoss` and a waiver grounded
in the caller's fixed `maximum_junction_gap`; zero-default refusals remain
unwaived. The report records both the declaration and actual outward rows.
`coordinate_budget` is borrowed through endpoint/topology certification and
does not become stored profile state.


`encode_dxf(profile, coordinate_unit=..., policy=...)` returns bounded ASCII
bytes. `write_dxf(profile, path, coordinate_unit=..., policy=...)` publishes them
atomically and returns a `DxfExportResult` with durable byte identity and conversion
evidence. Lines and circular carriers export analytically; rational conics and
splines export as control-point splines without refitting. An unrepresentable
coefficient refuses rather than becoming a rounded replacement.

### Decode, extrude, and round-trip an exact profile

This in-memory ASCII drawing declares millimeters and a closed four-by-two
rectangle. No user file or optional CAD engine is needed. Expected standard
output: none.

```python {.executable}
import phydrax as phx

phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
header = "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n"
rectangle = (
    "0\nLWPOLYLINE\n90\n4\n70\n1\n"
    "10\n0\n20\n0\n10\n4\n20\n0\n10\n4\n20\n2\n10\n0\n20\n2\n"
)
drawing = (
    header + "0\nSECTION\n2\nENTITIES\n" + rectangle + "0\nENDSEC\n0\nEOF\n"
).encode("ascii")
dxf_policy = phx.interchange.DxfImportPolicy()
decoded = phx.interchange.decode_dxf_bytes(drawing, dxf_policy)
assert decoded.coordinate_unit.unit_id == phx.units.MILLIMETER.unit_id
assert not decoded.profile.holes
solid = phx.geometry.brep_extrusion(
    decoded.profile, (0.0, 0.0, 3.0),
    coordinate_contract=phx.SpatialCoordinateContract(decoded.coordinate_unit),
    tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
)
assert solid.require_geometry().edge_vertex_gap_bounds is None
encoded = phx.interchange.encode_dxf(
    decoded.profile, coordinate_unit=decoded.coordinate_unit, policy=dxf_policy
)
restored = phx.interchange.decode_dxf_bytes(encoded, dxf_policy)
assert restored.coordinate_unit.unit_id == decoded.coordinate_unit.unit_id
assert restored.profile.outer.vertices == decoded.profile.outer.vertices
assert restored.profile.holes == decoded.profile.holes
```

File readers require `trusted_root`. Use `write_dxf` for atomic publication;
retain both its receipt and semantic report. `ProfileCurveUse` preserves
directed general-angle analytic conics, while unsupported legacy revolution
predicates refuse them. Extrusion and planar-face construction recompute
endpoint/face correspondence bounds rather than trusting imported declarations.


## Mesh and visualization files

Meshio-backed routes use explicit `MeshFileProfile` values. The declared matrix covers
provider-supported Abaqus, ANSYS, AVS-UCD, CGNS, DOLFIN XML, Exodus, FLAC3D, Gmsh,
H5M, HMF, Kratos MDPA, MED, Medit, Nastran, Netgen, Neuroglancer, OBJ, OFF, PERMAS,
PLY, STL, SU2, SVG, Tecplot, TetGen, UGRID, legacy VTK, VTU, WKT, and XDMF profiles.

Install `phydrax[mesh-file-interop]` for these optional file-codec routes.
Meshio is loaded only when an explicit codec boundary is executed; native
arrays, CAD decoding, geometry queries and mesh generation do not require it.
Retained meshio codec execution is external interoperability, not an in-house
numerical meshing implementation.

Single-file mesh exports are written to staging, decoded again through the declared
codec, compared against coordinates, connectivity, and fields, and then published.
Resource-set formats such as XDMF and TetGen require a resource-set writer rather than
silently publishing only one member.

Use geometry-appropriate VTK-family products:

- VTU for unstructured cells;
- VTP for surfaces, lines, or particles;
- VTI, VTR, or VTS for structured grids;
- PVTU for partitioned pieces;
- PVD for temporal collections;
- XDMF/HDF5 for large temporal arrays.

The currently qualified writer path is intentionally narrower than the entire VTK
family. Unsupported profiles fail explicitly.

## Native checkpoints and results

Durable accepted state uses lifecycle records. `CheckpointManifest` binds exact array
shards, payload digests, byte counts, analysis identity, numeric revision, execution
identity, diagnostics, and the completion boundary. `ResultManifest` binds named
fields, payload checksums, units, sampled semantics, evidence, and diagnostics.

Disposable caches may use atomic canonical archives and treat corruption as a miss.
They do not become lifecycle evidence merely because they contain arrays.

Exact-resume Equinox products remain template-bound. Their state bytes are
content-addressed and their JSON manifest is the commit boundary. Portable model
artifacts continue to use explicit architecture recipes and safe leaf codecs.

## Deliberate direct-I/O exceptions

Direct file primitives remain only where the surrounding component owns the physical
format or the path is an isolated provider workspace:

- the native array archive owns its ZIP and NPY encoding;
- lifecycle repositories own provider-specific POSIX and object-store transactions;
- HDF5 output plans own appendable inspection-series groups and completion markers;
- external-runtime workers read and write only their bounded private staging roots;
- third-party providers receive immutable staged files when they cannot consume a
  descriptor-backed stream;
- packaged worker bytes and distribution metadata are trusted installation-identity
  inputs;
- cache files use atomic publication and treat corruption as a miss.

Application-level external paths, native checkpoints, deployment bundles, mesh
exports, and rights-governed references do not qualify for these exceptions.


## Adding a format

A new file format requires all of the following:

1. A real Phydrax consumer and native target contract.
2. An exact format/profile and implemented direction.
3. A carrier and finite resource policy.
4. Magic or schema validation independent of the extension.
5. Bounds before allocation, decompression, or provider invocation.
6. Explicit mappings for units, frames, axes, identities, validity, and status.
7. An `AdapterReport` covering preservation and loss.
8. Resource or publication evidence.
9. Corruption and capacity behavior.
10. A real or standards-faithful fixture and an actual-codec round trip for writers.

Opaque vendor databases, Python pickles, framework checkpoints that import arbitrary
code, and hidden network fetching remain outside the core decoder surface.
