# Meshing

`phydrax.meshing` owns mesh construction, adaptation, certification, provider
execution, and topology transitions. Solvers consume the resulting native
carriers; they do not own external meshing sessions.

Run `python examples/meshing_native.py` for certification, refinement, exact
linear field transfer, and a finite-element solve on the refined carrier.

## Certify an existing mesh

```python
import numpy as np
import phydrax as phx

mesh = phx.discretization.CellMesh.from_triangles(
    np.asarray(((0., 0.), (1., 0.), (0., 1.))),
    np.asarray(((0, 1, 2),), dtype=np.int32),
)
result = phx.meshing.certify_cell_mesh(
    mesh, phx.SpatialCoordinateContract.si(),
)
assert result.audit.passed
```

A `CellMeshingResult` contains the canonical `CellMesh`, independent
`CellGeometrySpec`, physical coordinate contract, quality and audit reports,
compliance, staged trace, provider/runtime identity, and derivative mode.
A failed audit or compliance check is not a successful result.

`CellGeometrySpec` belongs to `phydrax.discretization`, not FEM. Its coordinate
nodes and element routes can represent geometry of a different polynomial order
from a solver's unknown field. Corner topology remains separate from curved
geometry nodes.

Audit `quality_scope` distinguishes `vertex_geometry` from `corner_cells`.
Native certification rejects unsupported high-order geometry rather than
presenting corner quality as curved-element validity. Gmsh separately evaluates
the full element's minimum Jacobian determinant. Supplied geometry or semantic
evidence that cannot survive canonical reordering is rejected, not silently
rebound. Association residual tolerances remain provider-owned.

## Scopes and organization

A `MeshingScope` names a source, its exact revision, entity kind and degree,
authoritative entity set, and persistent IDs. Boolean scope operations require
matching identity domains. Stale revisions are errors, not empty selections.

- `MeshPatch`: geometric boundary support.
- `MeshZone`: exclusive solver/material organization.
- `MeshLabel`: overlapping semantic grouping.
- `MeshAttribute`: entity-associated numerical or categorical data.

Geometry associations record source entities, target IDs, residuals, and
resolution/ambiguity evidence. Persistent IDs are not positional array indices.

## Controls and providers

`SurfaceMeshingSpec`, `SurfaceRemeshingSpec`, and `VolumeMeshingSpec` separate
physical requests from backend options. Cell-family policies, fill strategies,
size controls, features, regions, layers, and periodic constraints are explicit.
A control's presence in the native contract does not imply every provider can
honor it. Provider preflight must reject unsupported combinations before work.

Uniform, curvature, and proximity controls compile into a resolved size field.
`MeshMetricField` represents an SPD anisotropic metric. Native normalization
applies size bounds, anisotropy limits, and graph gradation. These operations
prepare requests; they do not certify that an external generator met them.

Gmsh is optional (`pip install 'phydrax[meshing-gmsh]'`). Use `GmshProvider`,
`GmshOptions`, and a revision-checked reopenable `BRepModel` or solid `BRepSource`; the old
`CADFEMMeshingPolicy`/`mesh_brep_for_fem` API has been removed. Sessions are
host-side resources and must not enter JAX state. Use a context manager for an
explicit `GmshSession`; its cleanup does not suppress the original exception.
Open CAD faces use `BRepModel`; `BRepSource` retains its closed-solid invariant.
Layer support is an explicitly qualified whole-solid straight normal sweep,
including experimental structured hexahedra—not arbitrary boundary-layer cores.
Required mixed families must actually occur in the generated result; permitting
mixed recombination does not guarantee that triangles will remain.

`NativeImplicitProvider` separates surface discovery from fixed-route
realization. Differentiate the prepared realization while its topology and
route remain valid, not through host-side discovery or audit decisions.

`ManifoldProvider` performs union, difference, and intersection of closed,
oriented triangular `SurfaceModel` operands in the same physical frame.
It preserves source-face ancestry and cell tags through the backend's face
relations. Semantic selections/interfaces require explicit transfer and are
rejected rather than silently dropped. Empty output is reported as a failure
because `CellMeshingResult` requires a nonempty carrier. Boolean topology changes
are nondifferentiable.

## Interchange

`MeshArrayArtifact` is the neutral host interchange representation.
`export_mesh_array_artifact` and `import_cell_mesh(artifact)` preserve native
block names, per-degree persistent IDs, geometry-node/vertex ordering,
attributes and units, zones/labels, coordinate contracts, and source revisions.
Pass explicit geometry-node IDs to preserve non-vertex node identities.

External `export_cell_mesh` uses meshio formats, which cannot encode the full
native contract. It requires `allow_lossy=True` and enumerates missing metadata
and any codec-level data losses; numerical fields are checked by reading the
written representation. A successful file write alone is not a lossless round
trip.

Shared reference-node ordering lives in the discretization layer. External
connectivity order must be converted there rather than duplicated in solvers.

## Adaptation and optimization

```python
transition, transfer = phx.meshing.refine_triangle_mesh(
    result.mesh,
    np.asarray(result.mesh.blocks[0].global_ids),
    result.coordinate_contract,
)
stencil = transition.vertex_stencil
assert stencil is not None
values = stencil.apply(result.mesh.vertex_global_ids, result.mesh.coordinates[:, 0])
```

`CellMeshTransition` binds a source mesh to an audited target, `MeshLineage`,
and optional vertex interpolation stencil. A `FiniteElementTopologyTransaction`
consumes that candidate and transfer evidence; rejection preserves the accepted
state. Unknown parentage must remain unknown, never guessed from new positions.

`TargetMatrixOptimizationPlan` and `optimize_cell_mesh` optimize fixed-topology
simplex coordinates with fixed-node constraints and an inversion barrier.
`optimize_cell_geometry_coordinates` supports user-defined high-order objectives;
the objective and acceptance policy must encode the required curved-element
validity. Derivatives of numerical objectives do not differentiate topology
changes or host-side acceptance.

## Polyhedral storage

Polyhedral topology uses packed incidence offsets/values. Exact-width
`PolyhedralBlock` groups coexist with standard cell blocks. Padding belongs only
in explicitly bounded execution worksets, not authoritative connectivity.
`prepare_polyhedral_worksets` rejects allocations beyond its entry budget.

## Assemblies, distribution, and coupling

`MeshPart` retains a certified cell carrier, prepared tensor grid, point cloud,
or spline carrier. `MeshAssembly` organizes named parts without tessellating
compact carriers. Part identities include numerical geometry: moving a mesh
invalidates old distribution and coupling evidence even when topology is fixed.

`CellPartition` is the solver-neutral cell ownership contract. `MeshDistribution`
adds exact global IDs, owned rows, ghost residence, and dependency routes.
`lower_finite_element` produces native FE phase/workset plans;
`lower_finite_volume` produces a grid-revision-bound Cartesian FV decomposition.
The old FEM-specific partition class is removed, without an alias.

Conformal and periodic overlays validate node bijections. Periodic vector
traces include the isometry's rotation. `ContactCoupling` is frozen node-to-node
normal contact, not a collision-search engine. `OversetCoupling` provides donor
interpolation and its transpose with explicit holes and partition-of-unity
weights; it does not claim conservative overlap remapping.

## Learned proposals remain untrusted

Marking, size, metric, and coordinate proposals bind an exact certified source.
`project_mesh_proposal` applies deterministic bounds, protected-entity,
gradation, SPD, and displacement constraints. `prepare_mesh_proposal` invokes
trusted native refinement or optimization and returns a
`MeshProposalTransaction`. Safety audit and compliance are separate evidence;
rejection or rollback retains the accepted source.

The native size/metric proposal route performs one triangle-refinement step.
It does not claim anisotropic adaptation or that requested final sizes were
achieved. Use an appropriate external metric provider for those requests.
Mesh promotion does not implicitly transfer PDE fields: consume the exposed
transition/transfer through the solver topology transaction.

## Additional optional backends

| Provider | Dependency | Supported execution boundary |
| --- | --- | --- |
| Mmg | `phydrax[meshing-mmg]` | Affine simplex metric adaptation through Mmg2D/MMGS/MMG3D |
| fTetWild | `phydrax[meshing-ftetwild]` | Robust surface-to-tetrahedron generation; sampled boundary-envelope evidence |
| Poisson | `phydrax[meshing-poisson]`, Python below 3.13 | Open3D screened Poisson reconstruction from oriented points |
| OpenVDB | Native `openvdb` Python binding; conda-forge provides it | Existing sparse voxel field to isosurface, with explicit background semantics |
| Omega_h | MPI-enabled Omega_h plus the packaged native bridge | Collective simplex metric adaptation, ownership, and ghost residence |
| VoroCrust | Source-built mesher plus the packaged extraction bridge | Surface sampling and explicit face-defined Voronoi cells |
| TIOGA | MPI-enabled TIOGA plus the packaged native bridge | Overset hole cutting and donor/receptor interpolation between affine cell parts |

External topology-changing providers are nondifferentiable. Mmg and Omega_h
output IDs identify the new revision; unknown ancestry is not replaced with a
nearest-neighbor transfer. fTetWild's boundary check samples vertices and
centroids and is not a continuous Hausdorff certificate. Poisson reconstruction
does not invent CAD associations. OpenVDB must know how inactive and
out-of-domain voxels are extended.

Native bridge sources and CMake definitions are packaged under
`phydrax/meshing/providers/native`. They link separately installed upstream
libraries; importing `phydrax` does not compile or launch them.
For VoroCrust, build the upstream checkout, then configure the bridge with
`VOROCRUST_SOURCE_DIR` and `VOROCRUST_BUILD_DIR` pointing to that checkout/build.
Pass the `vc_mesh` and `phydrax-vorocrust` executable paths to
`VoroCrustProvider`. Its radius control is the backend sphere-sizing bound, not
a guaranteed edge length. No material identities are inferred from seed colors.
Backend vertices that differ only within `relative_merge_tolerance` times the
output bounding-box diagonal are normalized to the lowest source vertex ID;
coordinates are not averaged. Collapsed alias faces and merged vertex counts
are reported in compliance. Set the tolerance to zero for exact aliases only.

TIOGA distributes complete named parts among MPI ranks and preserves part-local
ID namespaces. It does not partition one part or support curved/polyhedral donor
cells. Runtime-specific MPI launcher flags belong in deployment configuration,
not mesh semantics.

Run native benchmarks with
`python -m tools.meshing_benchmarks --resolution 8 16`.
`python -m tools.meshing_qualification --provider gmsh` exercises the real
curved-CAD path; the command also supports Manifold, Mmg, fTetWild, Poisson, and
VoroCrust. VoroCrust accepts explicit `--executable` and `--extractor` paths.
`examples/meshing_omega_h.py` exercises serial or MPI adaptation.

The local Homebrew MPI launcher required the explicit deployment setting
`HWLOC_SYNTHETIC='pack:1 core:10 pu:1'` and launcher arguments
`--bind-to none --map-by slot` to avoid an upstream hardware-topology startup
crash. This is not injected by either provider.

The QA extra installs SciPy and CAD typing packages. Native interfaces missing
from upstream declarations use narrow, runtime-checked protocols: the CAD edge
downcast and OpenVDB module, grid, accessor, and NumPy polygon outputs are
explicitly typed. OpenVDB remains lazily loaded. These boundaries require no
type ignores or unchecked casts and are exercised against the real libraries.

## Provider prerequisites and feasibility

VoroCrust source is available at
[sandialabs/vorocrust-meshing](https://github.com/sandialabs/vorocrust-meshing).
Build with CMake and an OpenMP-capable C++ compiler. On Apple Silicon, Homebrew
GCC can supply OpenMP. The Sandia website's “Coming Soon” source notice is stale;
the GitHub source avoids its binary-download CAPTCHA.

Prime is an extraction feasibility decision, not an implemented provider.
The documented [Part API](https://prime.docs.pyansys.com/version/stable/api/_autosummary/ansys.meshing.prime.Part.html)
exposes zone/topology queries, not a direct cell-connectivity array route.
The [FileIO API](https://prime.docs.pyansys.com/version/stable/api/_autosummary/ansys.meshing.prime.FileIO.html)
can export Fluent meshes, LS-DYNA, and MAPDL CDB, so file-mediated extraction is
feasible in principle but requires format-specific semantic auditing.
[Prime Server](https://prime.docs.pyansys.com/version/stable/getting_started/index.html)
requires a licensed matching Ansys installation on supported Windows/Linux
systems; installing the Python client alone does not supply the server.
