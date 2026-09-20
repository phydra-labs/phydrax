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

## Mesh and visualization files

Meshio-backed routes use explicit `MeshFileProfile` values. The declared matrix covers
provider-supported Abaqus, ANSYS, AVS-UCD, CGNS, DOLFIN XML, Exodus, FLAC3D, Gmsh,
H5M, HMF, Kratos MDPA, MED, Medit, Nastran, Netgen, Neuroglancer, OBJ, OFF, PERMAS,
PLY, STL, SU2, SVG, Tecplot, TetGen, UGRID, legacy VTK, VTU, WKT, and XDMF profiles.

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
