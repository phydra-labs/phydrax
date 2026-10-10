# Meshing

`phydrax.meshing` owns mesh construction, adaptation, certification, provider
execution, and topology transitions. Solvers consume the resulting native
carriers; they do not own external meshing sessions.

## Evidence vocabulary and current closure state

Native meshing capability is reported in four separate lanes:

- **implemented** means that the named source/request/route exists and is
  admitted by its public contract;
- **tested** means that the named focused checks ran successfully;
- **qualified** requires a retained, route-bound campaign artifact with its
  source, policy, runtime, tolerance, and resource identities; and
- **released** requires the separate authenticated release decision.

A safe refusal, a passing unit test, an optional-provider result, or a generated
catalog entry does not promote the next lane. The W02--W15 labels below are
implementation-workstream identifiers, not API or persistence schema versions.

| Workstream surface | Implemented/current test evidence | Qualification boundary |
| --- | --- | --- |
| W02 curve/planar/surface, W03 PLC volume, W05 imperfect-surface envelopes, W06 periodic combinations, W10 structured/quad/hex, W11 polyhedral, W12 lifecycle/distribution, W13 overset, W14 solver/design | Public implementation is present and the final focused owner checks reported clean. W05 exercised sheet, dirty, disconnected, and marked-adaptation envelope workflows under the unchanged 20,000,000-work request. W06 additionally exercised rotational-volume H(div)-3, mixed rotation/screw H1/H(div)/H(curl), and installed-Gmsh periodic STEP routes under unchanged limits. | No final W15 qualification or release artifact is inferred from those checks. Exact admitted combinations are the generated capability profiles; their Cartesian product is not supported. |
| W04 implicit/image/multimaterial | The current focused matrix is 83/83. | This is test evidence, not a retained final qualification campaign. |
| W07 native CAD | Native source-only `realize=False` import, three-circle OCP loft preservation, BRep topology/volume, cavity/shared-shell, unattached-edge, periodic-rational, surface-array, and scaled-oblique cases passed 16 targeted checks; the isolated loft route completed in 64.08 seconds. The overlapping-sphere partition publishes a real 0.001-policy mesh whose collapsed-pole radial splits retain exact rational chart coordinates, and the curved-void realization meets its 0.001 deviation through per-source-interval trim ribbons that are refined locally. | This is focused implementation/test evidence, not a final W15 qualification artifact. BRep text refuses implicit curved branches rather than fabricating them. Complete STEP/IGES entity coverage, every intersection curve, singular arrangements, and the independent writer campaign remain separate gates. |
| W08 metric/remesh | The focused surface-metric matrix is 32/32 and the tetrahedral-metric matrix is 40/40. Rational-trim archive/continuation and curved-sphere smokes passed, as did the 12 canonical positional archive-recipe checks. Dense accepted adapted targets reopen through the existing `accepted_target` role, bound to certification input, report, associations, and one canonical source-to-target lineage event; unchanged zero-displacement whole targets reuse certified endpoint measures. | This is focused implementation/test evidence, not a final W15 qualification artifact. It does not admit approximate carrier/source substitution, relaxed size, quality, fidelity, or resource limits, or generic topology generation. |
| W09 boundary layers/high order | Native advancing layers, immutable-cap core fill, and bounded high-order paths are implemented. The corrected exact-x-orbit source `curved-periodic-narrow-gap-exact-x-orbits` (revision `2afa0d24…`, source digest `059b334c…`) passed the complete unchanged 120-second gate; the final two repeats took 106.2345 and 108.5906 seconds (process walls 111.40 and 113.67 seconds). H(div) commuting/continuity defects were about `1e-14`, FV defects zero, and PDE defects about `1e-15`. | This is focused route qualification, not final W15/release evidence. The original ADVANCING period-x=1 source remains an immutable negative case: its authored top trace has exact x residuals `-101121/2361183241434822606848` and `-6620711477/77371252455336267181195264`. No tolerance increase, snapping, or source rewrite converts that original case into a positive. |
| W15 corpus/tooling/release | The bounded tooling/corpus checker matrix is 46/46. | Final qualification and benchmark artifacts have not been produced. The like-for-like performance and authenticated leadership/release gates remain open and no superiority claim is made. |

Every source-facing tolerance has one owner and one physical meaning. Predicate
filters certify signs or return unresolved; source-fidelity bounds measure
continuous two-sided geometric error; coverage tolerances bound independently
computed measure defects; association tolerances classify an already
authoritative source entity; and solver tolerances govern algebraic residuals.
They are not interchangeable slack. Construction and certification retain the
authored source revision, exact source rows and scientific IDs. Resource limits
are hard preflight/commit bounds; exhaustion returns typed evidence and leaves
the accepted mesh unchanged. Periodic support requires an exact declared
isometry and quotient-orbit data—nearby coordinates or post-hoc node matching
never create periodic topology.

### Native package identity

The native shared library is the separately packaged distribution
`phydrax-meshcore`; it is not a renamed third-party mesher and is not embedded
in the main `phydrax` wheel. `phydrax[meshcore]` pins the identical
Phydrax release. Loading verifies, in order, the public C header's ABI digest,
every bound symbol, the release, the configured source digest, and the build
configuration; the loader also hashes the bytes of the selected installed
library. Qualification runtime records therefore distinguish `source_digest`,
`binary_digest`, and `configuration`. An isolated rebuild may
have the same source and ABI identities but a different binary digest; neither
identity is qualification or performance evidence.

The retained integration receipt identifies source digest `96982a2a…` and ABI
digest `01d7a8cf…`. The resolver-installed wheel has binary digest `ca26383c…`;
the independently configured isolated build has binary digest `48b55b54…`.
Those binaries share the recorded source/ABI contract but are distinct build
artifacts. The binary difference is not a numerical difference, qualification,
or performance result.

`PHYDRAX_MESHCORE_LIBRARY` explicitly selects a shared library. Otherwise the
loader resolves the installed `phydrax_meshcore` distribution. A missing
library is an unavailable optional dependency, while a release, ABI, symbol, or
identity mismatch is a fail-closed `MeshcoreUnavailableError`; neither path
silently substitutes SciPy/Qhull or an external mesher.

## Workflows

Each script is a complete, bounded public-API workflow that checks its own
evidence and prints a JSON summary:

- [`examples/meshing_native.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/meshing_native.py):
  certification, bisection, exact linear field transfer, and a finite-element
  solve on the refined carrier.
- [`examples/adaptive_bisection_heat.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/adaptive_bisection_heat.py):
  adaptive 2D and 3D heat solves with Zienkiewicz-Zhu recovery, Dörfler marking,
  and `NATIVE_BISECTION` refinement.
- [`examples/adaptive_device_simplex.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/adaptive_device_simplex.py):
  solve, mark, refine, and coarsen on the device inside one capacity bucket,
  then commit one certified mesh.
- [`examples/anisotropic_metric_adaptation.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/anisotropic_metric_adaptation.py):
  recovered Hessian, `L^p` metric, `NATIVE_METRIC_2D` adaptation, and re-solve.
- [`examples/ale_conservative_remesh.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/ale_conservative_remesh.py):
  mesh motion assessed by `MeshMotionMonitor`, remeshing on its decision, and a
  conservative second-order remap through the common refinement.
- [`examples/cad_high_order_curving.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/cad_high_order_curving.py):
  native sphere source and constrained chart triangulation, independent
  coordinate degrees 2, 3, 4, 6 and 10, and a P1 consuming field.
- [`examples/boundary_layer_core_mesh.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/boundary_layer_core_mesh.py):
  native advancing prism layers, an immutable PLC core, combined-domain
  certification, and diffusion without Gmsh or OCCT.
- [`examples/delaunay_voronoi.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/delaunay_voronoi.py):
  exact meshcore Delaunay, constrained Delaunay, Voronoi, and power diagrams.
- [`examples/native_surface_meshing.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/native_surface_meshing.py):
  source-bound sphere triangulation, embedded-manifold H1 metrics, an exact
  zero-mean constraint, and a converging Laplace--Beltrami solve.
- [`examples/native_tetrahedral_meshing.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/native_tetrahedral_meshing.py):
  a two-material PLC with a cavity, constrained tetrahedral generation,
  independent region-volume checks, and a manufactured diffusion solve.
- [`examples/native_multiblock_meshing.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/native_multiblock_meshing.py):
  two source-bound transfinite blocks, one conforming pure-quad carrier, and a
  manufactured Q1 convergence check.
- [`examples/native_polyhedral_meshing.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/native_polyhedral_meshing.py):
  nonconvex two-material restricted-power cells, VEM/FV consumers, conservative
  common-refinement transfer, and an atomic composition rebind.
- [`examples/meshing_omega_h.py`](https://github.com/phydra-labs/phydrax/blob/main/examples/meshing_omega_h.py):
  serial or MPI metric adaptation through the persistent Omega_h worker.

The following repository integration recipes deliberately exercise
qualification-only source/archive helpers in addition to public numerical
facades; their existence does not make those helpers public API:

- `native_quad_hex_meshing.py` covers affine dual quad/hex and separately
  declared mapped balanced/frame-grid cases. Its relaxed positive size policy
  is not a pass of the original zero-tolerance request.
- `curved_native_transfer.py` covers a curved nested edit, learned marking
  proposal, physical state transfer, independent reanalysis, and fixed-epoch
  JVP/VJP checks without claiming learned superiority.
- `hp_metric_order_adaptation.py` reports bounded h/p/metric trials and ordered
  timing samples. Shared process peak RSS is not candidate allocation cost.
- `tools/native_moving_overset.py` covers interpolation and separately
  conservative transfer through motion, rebind, PDE update, and a temporary
  durable restart.
- `tools/layer_core_lifecycle_continuation.py` is a consumer of an existing
  content-addressed layer/core source archive; it requires the archive path and
  expected content ID and is not a standalone generator.

The code blocks of this guide form one sequence: later blocks reuse the
`result` certified in the first.

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
Where present, `surface_source` retains the actual `SurfaceSourceCharts` graph:
raw source UV, patch and cell scientific IDs, domain and authentic boundary
support belong to the same generated source, not a reconstructed outcome proxy.
Source archive reopening and stale/corrupt chart refusal are separate from a
successful raw-coefficient bisection theorem or complete surface PDE lifecycle.

`CellGeometrySpec` belongs to `phydrax.discretization`, not FEM. Its coordinate
nodes and element routes can represent geometry of a different polynomial order
from a solver's unknown field. Corner topology remains separate from curved
geometry nodes.

Affinity is a property of the complete source map, not its Python element
class or declared degree. Source-backed admission prepares the owning exact
coefficient bank, proves that all nonzero coordinate-polynomial terms have
total degree at most one, and checks the source's nearest-even corner images
against the carrier's actual binary64 bits. This work remains charged to the
active original allowance. An affine restriction or exact source is not
replaced by a new corner-only geometry merely because that predicate succeeds;
adaptation and motion must retain the original representation/ancestry or use
an explicit owning geometry transition.

Nested edits may retain a `RestrictedCellGeometryElement`: its coordinate map
is the original basis composed with the complete stored child reference chain.
Publication evaluates that chain and the root expression in exact arithmetic,
then publishes nearest-even binary64 corner images. Coarsening restores the
recorded parent basis and coefficient bank rather than numerically inverting or
interpolating the child map. Accepting the element does not bypass mapped
validity or replace the map with the corner carrier.

`BarycentricCellGeometryElement(source_element, barycentric_weights)` is a
different, full affine-P1 coefficient action. Its source is the canonical P1
element or another retained full action; each finite square matrix includes
every source column, including column zero. Actions retain their ordered source
stack and authored weights, even when row sums are not one. They are not
converted into Cartesian restrictions, normalized, or flattened into an
archive-only matrix. `CellGeometryRestrictionSource` separately records
scientific parent ancestry; that ancestry does not make a full coefficient
action a Cartesian reference chart. Its optional `block_source_blocks` names the
authored source block owning each restricted block. Bisection groups cells by
that owner and their composed action: a cell whose action is the identity on
its root rejoins the owner block, and other cells form
`<owner>/coefficient-action/<element_id>` blocks. A restriction without the
record presents each root as `source-cell/<id>`. Coefficient actions do not authorize an
extension of arbitrary curved source geometry, nor do they bypass publication
validity, embedding, coverage or source-fidelity checks.

**Curved global embedding.** Positive Jacobians alone do not prove that a
curved mesh is embedded. For mapped volume meshes the certificate proves a
boundary-degree theorem. Every closed cell map is certified orientation
preserving and locally injective. Cells sharing a vertex, edge or facet have
exactly identical restricted maps. The curved boundary is injective. Every
boundary shell has zero exterior degree. Under these premises every point has
at most one preimage, so interior cell pairs are never intersected pairwise.
Adjacent curved boundary facets, and adjacent cells of curved surface meshes,
touch along their shared entity and are separated by a certified tangent-cone
contact lemma rather than by box separation. The premises are recorded in
`GlobalEmbeddingCertificate.boundary_degree` (`MappedBoundaryDegreeEvidence`).
An unproven premise falls back to the pairwise route, never to acceptance.

**Exact construction sources.**

`CellGeometrySpec.exact_source` is the single exact-construction slot. It
accepts the owning exact power source, its restriction or linear-action source,
or the owning exact PLC source and authenticated convex-source image; each
source must match the geometry's declared cell family. Direct bindings use
`CellGeometrySpec.power(mesh, source)` or `CellGeometrySpec.plc(mesh, source)`;
retained action/convex graphs preserve their actual parent geometry and source.
There is no separate power-only slot or compatibility representation.

For native PLC tetrahedra, a constrained vertex names an original source edge
or triangle and its construction parameters. The original domain identity and
revision, source rows, and sparse `int64` scientific entity IDs remain
authoritative. Native `int32` row selectors are indices into those banks, not
scientific IDs. Exact source coordinates drive validity, embedding, coverage,
fidelity, measures, and metric sampling; binary64 coordinates are verified
nearest-even execution carriers. A carrier already exactly on its declared
closed source entity remains its own exact coordinate.
An exact-source predicate execution view may use binary64 only after proving
every source fraction equals that binary64 value exactly. A nonrepresentable
bank retains its original exact owner; this optimization changes neither
source arrays/SCI identity nor caps. A nearest-even carrier alone is not that
proof and cannot authorize predicate source normalization.
Boundary tags likewise bind original scientific entity IDs, not coincident
source/mesh row positions. Permuting the original boundary SCI bank requires
the corresponding identity-based remap; it does not change the source geometry
or authorize a relabeled physical boundary.

`ExactPlcCellGeometryConvexSource` retains the complete original parent mesh
and exact geometry, exact rational vertex supports, and original parent-cell
scientific IDs. It authenticates original corner laws, exact parent-reference
membership and target nearest-even carrier bits for admitted degree-one
tetrahedron/hexahedron/pyramid or cut-polyhedron images. Shared physical
coordinates do not substitute for those SCI bindings. Parent source work/bit
allowances remain unchanged; convex topology, mapped validity and complete
coverage still require independent certification. This source mechanism alone
does not qualify the original mixed/polyhedral adaptation or motion campaigns.

Scoped exact clipping/corner preparation checks include a nonorthogonal
pentagonal prism, two-material original-domain maps, embedding/coverage/exact
volume and nonsimple/immutable-face/capacity refusals. A separate exact box
clip of an original tetrahedron retained complete plane witnesses and exact
volume with ten hexahedral corner rows. These are bounded component proofs,
not general-scale all-hex source quality, cold PDE or varying-frame acceptance.

Bounded boundary refinement carries the construction witness through the
atomic native edit. A bound below the certified deviation, zero-tolerance
nonrepresentability, a fixed boundary, or insufficient capacity refuses the
edit with evidence rather than moving the source or discarding ancestry.
Nested refinement, coarsening, metric adaptation, and source archives retain
the exact source owner. Rebinding arbitrary coefficients does not create new
source authority.

B-Rep-derived `MeshingDomain` views retain their explicit geometry or model
owner. Archive validation rebuilds them through the same B-Rep construction
route, retaining authoritative physical edge queries, endpoint bounds, and
original vertex banks; composing patch pcurves is not a replacement authority.
The owner is dynamic fixed-role state, not a static hidden array container.
Restoration still validates exact registered types, complete field sets, and
every scientific field. A source-only archive may hold an owning source record
or `(source, specification, options)` tuple. A checkpoint mapping instead
requires its actual certification inputs, report, and associations, plus the
whole registered `MeshPart` when generation roles are declared.
`MeshingAcceptedEpoch(carrier, declarations, fields, transition=...)` binds an
actual accepted carrier and complete declared field banks. The closed
`accepted_epochs` role retains ordered history and exact source/target
transition bindings; FE and FV fields may share one actual part without fake
parts. The separate closed `neurofluid_transport` role requires the actual
typed transport checkpoint and complete compartment source/result theorem,
not a generic mapping or invented field declarations. These codec role
contracts are not a full scientific Image/PDE continuation proof.
`MeshingFieldIndexBinding.coefficient_bank` retains the complete ordered
scientific coefficient identities in an immutable array-free bank, with the
field-space identity stored once and every ordinal/component/coefficient ID
explicit. It replaces the old repeated coefficient-identity tuple; it is not a
digest, shape inference or moved numerical field. Actual `index_values` remain
unchanged. Prepared-space validation rejects missing, duplicate, reordered or
foreign identities, and canonical recipe/epoch/archive IDs honestly change.
This bounded encoding does not raise the original archive limits or establish
an unexercised full cold lifecycle.
The original sweep-hex epoch-zero binding has exercised this codec gate with
the complete scientific coefficient bank and all FE/FV fields under its original
limits. Its next adaptation still capacity-refuses; first binding is not
refinement, history continuation, full cold or sweep-hex qualification.
Cold theorem comparison validates complete original and fresh reports,
embedding/coverage outcomes, limits, schedules, bindings and derived identity
dependencies while retaining both genuine receipts separately. Historical
source-expression work/peak observations are not scientific source content:
fresh proof cost need not equal historical cost, but neither receipt/report
nor its IDs are rewritten to force equality. This comparison contract does
not assert that an unexercised first-write or fresh-cold workflow passed.

Unprepared IGA source owners are admitted only through the registered canonical
axis/basis constructors with complete fields replayed. Original nonuniform
rational weights retain their raw binary64 bits and gauge; restoration checks
assembly geometry/source identity and rejects stale geometry. This source
archive admission does not admit prepared IGA runtimes, JAXPRs, operators or
callbacks and does not normalize weights. Scoped source archive and ellipse
insertion/elevation/orientation derivative checks are distinct evidence, not
full IGA physics or CAD qualification.
Scoped prepared IGA consumers also exercise curved physical queries and
derivatives, outside-source refusal, nonuniform-pairing raw transpose/Hilbert
adjoint and stale runtime/order/source rejection. A public homogeneous-trace,
matrix-free sum-factorized diffusion plan compiled and executed at zero state.
That zero-state integration check is not a nonzero manufactured-PDE accuracy
proof or global CAD reader qualification.

Native placed-source queries retain the authored physical frame, including
admitted orthogonal reflections, instead of substituting a local-coordinate
proxy. Orthogonality admission does not normalize the authored coefficient
bits. Full-angular torus distance lower bounds use that frame and a continuum
bound; a sampled nearest point is not a global minimum certificate. Scoped
proper/reflected curved, live-parameter, affine, legacy and torus query checks
do not qualify heavy CAD partition/continuation workflows or every singular
closest-point case.

Prepared native spline bounds retain the anchored authored point law and true
homogeneous rational derivatives, including mixed second derivatives. A finite
first-derivative bound does not imply a finite Hessian at a C0 seam: an infinite
Hessian remains honest evidence, while a separately bounded trim ribbon may
still be certified. Constant seam/boundary charts and source-normal/tangent
queries use the same source owner; changed rational weights renew prepared
evidence rather than reusing an old cached law. Scoped seam, rational-jet,
cache-renewal and ribbon checks passed without source normalization. They do
not establish the original periodic STEP reader, full-period or cold-reader
campaigns.

The existing coordinate-enclosure owner retains the same source-span bank
through shared-curve angular refinement, source-bound trim-chain certification
and carrier interpolation. Preparation and quotient/restriction work remain
charged to the original allowances; retained storage includes actual backing
allocations, not just array views. This does not create a separate cache or
reset the work or scratch budget.

CAD interchange coverage is evidence, not an unchecked metadata label.
`CadCoverage` requires an immutable unique entity ledger, nonnegative integral
counters, finite fit error with the actual fitted-entity count, and positive
units. `CadImportResult` identity includes that coverage record without
rewriting the imported model or scientific source identity. Malformed IGES
decode retains its bounded resource manifest, including failures reached
through spline dependency chains. Scoped STEP read/write and coverage
validation evidence does not establish complete CAD construction/Boolean
capability or the independent writer campaign.

`CadExportResult` admits a successful provider output only with an actual
publication receipt, a valid adapter report and immutable coupled certificates.
Approximation payloads require declared-loss semantics; they cannot appear
under a lossless report. An explicit failed report remains a failure.
The scoped native empty-STEP output admission checks do not establish rational
writer success or the complete independent STEP/IGES profile and writer matrix.

Pyramid-to-tetrahedron restrictions can be genuinely rational. Their support,
field-content integration, and source coverage retain the actual quotient.
Removable apex denominators are canceled only through an exact change of
variables. Those integrals retain exact integration status; otherwise bounded
polynomial-series integration publishes an explicit remainder enclosure and
non-exact status. Planar source coverage uses full-expression plane identities
and oriented boundary integrals, not a corner-polyhedron volume or sampled
support. Exact coefficient work is charged independently of process-global
basis-cache warmth; exceeding a declared budget refuses before the operation
and leaves the source geometry and accepted state unchanged.

`GeometryAssociation` keeps source representation and revision separate from
mesh entity IDs. Indexed piecewise-linear associations carry explicit source
dimensions, 64-bit identifiers, and `GeometrySourceEntityRole` values for
vertices, edges, facets, and regions. Role is independent of geometric
dimension: a planar region and a volume facet both have dimension two but
different canonical source keys. Indexed PLC rows require explicit roles;
unclassified rows have no role, dimension, or index. Their canonical source keys
must agree with all three metadata fields. Indexed implicit associations name
the scalar source's single `implicit-zero-set` boundary; `MAPPED_REFERENCE`
associations name the exact image entities of an authored reference mesh.
Neither is relabeled as a B-Rep or piecewise-linear approximation.

Indexed `SURFACE` associations retain authored corner/curve/patch dimensions,
64-bit definition indices, and declared occurrence paths in their identity.
Index meaning belongs to the source domain's authoritative tables; it is not
inferred by parsing an entity ID or matching a physical coordinate. A surface
association cannot name a volume-region stratum. Source coverage charts and
physical-cell material correspondence remain separate evidence.

`PlcAssociationTransfer.source_vertices` is the authoritative source bank, not
the triangulated support domain's vertex array. A support domain may contain
Steiner vertices without creating new scientific source corners. For an
identity-preserving rigid planar or volume source translation, call the successor
transfer's `transition_source_associations(accepted, predecessor, target_mesh,
geometry=target_geometry, translation=shift)`. The transition verifies source
incidence, original authority indices, physical frame and scientific mesh
entity correspondence, then proves the actual successor maps lie on the
declared source. It returns revalidated patches, zones, labels and
parent-linked associations; incompatible source edits are refused.
For fixed-topology planar translation, the transition retains the accepted
association's exact target IDs, row order and occurrence paths, including sparse
protected-edge scopes. Proving complete internal source strata does not permit
expanding those scopes; successor source-revision identities, support,
parameters, orientations and parent evidence are rederived.


Native parametric surface results carry their original `SURFACE` authority
through serial `NATIVE_BISECTION` with a `SurfaceAssociationTransfer` built on
`PreparedSurfaceSourceSupport`. Topology lineage, never coordinate matching,
names each target cell's scientific root cell and its exact root-reference
corners. Every successor is published only after its fidelity request is bound
to the root's re-verified chart chain: the children of every root must tile its
reference triangle exactly, and each chord inherits the root's two-sided chart
bound plus its exactly measured deviation from the restricted root chord.
Provider remeshing routes refuse surface source transfer.

Changed image material sources rebuild their authoritative domain through
native PLC source preparation. Coalesced source facets may be polygons; they
are not reinterpreted as a flat list of triangles. Region renewal certifies
the moved carrier against this new domain and retains the prepared source
and its work/allocation evidence in `RegionMeshingRenewal.source_preparation`.
Unchanged source renewals retain their existing authority and still refresh
geometry-bound material coverage.

`evaluate_cell_quality(mesh, geometry=geometry)` evaluates the owning scalar
coordinate map. Measures integrate its signed Jacobian with a degree-aware
reference quadrature; shape, radius/sliver, condition, and metric statistics
sample its physical Jacobian images. Boundary angles and warpage use mapped
tangents and face normals. Rational and embedded measures remain sampled
quadrature, not exact integrals. Omitting `geometry`, or using the separate
`coordinates` override, explicitly requests straight-corner quality; supplying
both authorities is refused.

Audits always evaluate their actual `CellGeometrySpec`. Their `quality_scope`
is `mapped_coordinate_cells` for mapped coordinate elements and
`vertex_geometry` for vertex-defined cells. `sampled_valid` is sampled frame
evidence, not a whole-map validity or global-embedding proof. A supplied
evaluation must bind the current geometry layout and match every recomputed
quality output. Its actual `MeshMetricField` owner remains a dynamic,
nontrainable child with the original vertex identities; it is not a static
numerical payload. These canonical evaluation/report/archive representations
and identities change with the new source-map and metric ownership.

Geometric validity is certified by
`phydrax.discretization.certify_cell_geometry_validity`: the Jacobian
determinant of every mapped cell (Pk simplices, Qk quadrilaterals and
hexahedra, prisms, rational pyramids through collapsed coordinates, and
star-decomposed polyhedra) is converted to Bernstein form and
adaptively subdivided until it is proven above the policy floor
(`CERTIFIED_VALID`), proven below it (`INVALID`), or the `CellValidityPolicy`
budget is exhausted (`UNRESOLVED`). A twisted trilinear hexahedron with positive
corners but an inverted interior is therefore invalid, and curved geometry is
certified rather than rejected. A polyhedron that is not star-shaped about its
center stays `UNRESOLVED`; that is not a validity disproof.

Polygon cells are certified without assuming star-shapedness. A planar polygon
is `CERTIFIED_VALID` iff its vertices are finite and distinct, every edge clears
`relative_determinant_floor` times the polygon diameter, its boundary is simple
(`polygon_simplicity_2d`: non-adjacent edges are disjoint and adjacent edges meet
only at their shared vertex), it is counterclockwise, and twice its signed area
exceeds the policy floor. An embedded polygon must instead lie within
`CellValidityPolicy.relative_planarity_tolerance` of its Newell plane (relative
to the polygon diameter), be simple in its dominant-axis projection, and have a
squared vector area above the floor. Repeated vertices, self-intersecting
boundaries, clockwise planar loops, and nonplanar embedded loops are `INVALID`;
an undecided simplicity, orientation, planarity, or area decision leaves the cell
`UNRESOLVED`, never valid. The segment and polygon predicates are documented
with the [exact predicates](api/geometry.md#exact-predicates-triangulations-and-diagrams).

`CellMeshAuditPolicy` assigns each audit check a `CellMeshAuditDisposition`
(`REJECT`, `RECORD`, or `SKIP`). Topology checks run on the welded complex in
which vertices within `coincident_vertex_tolerance` of each other (relative to
the bounding-box diagonal, found through a Morton-sorted grid) are merged:
coincident vertices, collapsed and duplicate cells, non-manifold facets, edges,
and vertices (disconnected cell stars), inconsistent facet orientation,
watertight boundary, and self-intersection of the cells of planar and surface
meshes or the welded boundary of volume meshes, through BVH candidate pairs and
exact-or-filtered orientation predicates. By default every check is `REJECT`
except `watertight_boundary`, which is `SKIP`.

Each report partitions its dispositioned checks into `evaluated_checks` and
`skipped_checks`, by finding name in check order (`unused_vertices`,
`unused_geometry_nodes`, `coincident_vertices`, `collapsed_cells`,
`duplicate_cells`, `nonmanifold_facets`, `nonmanifold_edges`,
`nonmanifold_vertices`, `inconsistent_orientation`, `open_boundary`,
`self_intersection`, `invalid_geometry`); a default report skips only
`open_boundary`. `check_counts` gives the finding count of every evaluated check
and every `unresolved_*` entry. `audit.passed` is `True` iff `issues` is empty:
no evaluated `REJECT` check found anything, the geometry, quality, and evidence
bindings held, the quality thresholds and association requirement were met, and
no rejected check was unresolved. It certifies nothing about a skipped check,
and `RECORD` findings appear in `recorded` without failing the audit.

Uncertain predicates are never reported as a pass: self-intersection pairs whose
predicates stay uncertain are not counted as clean but name
`self_intersection_predicates` in `unresolved`, as do unresolved validity
certificates (`geometry_validity`), an exceeded weld candidate capacity
(`coincident_vertex_capacity`), and an exhausted intersection candidate budget
(`self_intersection_capacity`). Each becomes an `unresolved_<name>` finding under
`CellMeshAuditPolicy.unresolved` (`REJECT` by default; `SKIP` is refused). Under
`RECORD` the audit may pass, but the check stays named in `unresolved` and
`recorded`. Connectivity above `maximum_connectivity_entries` is rejected with
`connectivity_capacity_exceeded`; `maximum_coincidence_candidates` bounds
welding and `maximum_intersection_candidates` bounds the streamed polygon-edge
and triangle-pair broad phases. Exhaustion fails closed without materializing an
unbounded pair table. Supplied geometry or semantic evidence that cannot survive
canonical reordering is rejected, not silently rebound. Association residual
tolerances remain provider-owned.

`evaluate_finite_volume_quality` reports skewness and non-orthogonality of every
interior face of a full-dimensional mesh.

## Scopes and organization

A `MeshingScope` names a source, its exact revision, entity kind and degree,
authoritative entity set, and persistent IDs. Boolean scope operations require
matching identity domains. Stale revisions are errors, not empty selections.
Source-closure replay reconstructs mesh-backed scopes from the unique owning
`CellMesh` with the same mesh ID and numerical revision in that closure. It
rederives local and global inventories (and validates an explicit projection
when present); missing or ambiguous owning revisions refuse checkpoint
publication rather than trusting stored scope inventories.

Native envelope source closures retain the original `RawTriangleSoup` or
declared `SurfaceModel`, explicit wrapping permissions, repaired realization,
preserved volume carrier and directed repair evidence. Cold restoration
reconstructs the owning audit and wrapping theorem; changing only reported
distance bounds or topology counts cannot authorize the repaired source.
Original and repaired identities remain distinct. A source-only archive receipt
does not establish accepted volume coverage, field transfer or physical
continuation; those require their own original consumer gates.

- `MeshPatch`: geometric boundary support.
- `MeshZone`: exclusive solver/material organization.
- `MeshLabel`: overlapping semantic grouping.
- `MeshAttribute`: entity-associated numerical or categorical data.

Geometry associations record source entities, target IDs, residuals, and
resolution/ambiguity evidence. Persistent IDs are not positional array indices.

## Controls and providers

`SurfaceMeshingSpec`, `SurfaceRemeshingSpec`, and `VolumeMeshingSpec` separate
physical requests from backend options. Cell-family policies, fill strategies,
size controls, regions, patches, layers, and periodic constraints are explicit.
A control's presence in the native contract does not imply every provider can
honor it. Provider preflight rejects unsupported combinations before work.

`NativeMeshingOptions` accepts this closed selector set; strings not listed here
are not aliases:

| Route | Exact admitted source/output boundary |
| --- | --- |
| `planar_constrained_delaunay` | piecewise-linear planar loops/holes/features → affine triangles |
| `curve_arc_length` | declared parametric curves/networks → affine intervals |
| `parametric_surface` | revision-bound patches, trims, seams, and poles → source-associated triangles |
| `implicit_surface` | fixed-lattice or adaptive implicit discovery → affine surface triangles |
| `implicit_restricted_delaunay` | established-reach analytic implicit profile → affine tetrahedra |
| `implicit_adaptive_tetrahedral` | enclosure-complete regular scalar source → affine tetrahedra |
| `plc_tetrahedral` | oriented PLC with declared regions/cavities/features → affine tetrahedra |
| `image_material_tetrahedral` | occupied-cell label interpretation and compartment complex → shared-interface tetrahedra |
| `surface_envelope_tetrahedral` | explicitly repaired/wrapped envelope source → affine tetrahedra; not exact dirty-input conformity |
| `periodic_delaunay` | translational quotient point orbits → affine triangle/tetrahedron quotient carrier |
| `plc_restricted_power` | PLC plus declared sites/weights → bounded polyhedra |
| `layer_core` | accepted native layers plus immutable cap/core PLC → mixed prism/tetrahedron carrier |
| `structured_transfinite` | explicit conforming transfinite blocks → quadrilateral/hexahedron carrier |
| `sweep` | pure-quad profile plus explicit extrusion map/stations → hexahedra |
| `planar_dual_quad` | subdividable planar PLC boundary → affine pure quadrilaterals |
| `plc_dual_hex` | subdividable affine PLC boundary → affine pure hexahedra |
| `plc_hex_dominant` | affine PLC with explicit mixed-family policy → hex/pyramid/tetrahedron carrier |
| `plc_balanced_grid_hex` | affine PLC plus balanced-grid schedule → affine hexahedra |
| `plc_frame_grid_hex` | affine PLC plus frame/integer-grid schedule → affine hexahedra |
| `mapped_balanced_grid_hex` | exact mapped reference PLC plus balanced-grid schedule → preserved mapped hexahedra |
| `mapped_frame_grid_hex` | exact mapped reference PLC plus frame-grid schedule → preserved mapped hexahedra |

Each row still requires its route-specific source, control, certification, and
resource admission. In particular, affine, mapped, periodic, material, layer,
and high-order rows do not compose merely because each appears independently.

### Typing and structural contracts

Meshing conversion boundaries accept JAX/NumPy array-like values and convert once;
compiled kernels consume canonical JAX arrays. `MaskedSimplexMesh`,
`MeshMetricField`, and `MeshMetricSamples` opt into `phydrax.typing` structural
contracts. Their nominal dimensions enforce aligned vertex/cell slots, coordinate
components, simplex widths, and square metric tensors after transformations such as
`equinox.tree_at`, without adding device operations. Scientific admissibility remains
with the owning constructors and evidence: simplex orientation/capacity, metric SPD
properties, hard size/anisotropy bounds, gradation, and provider compliance are not
inferred from shapes.

`RegionControl` assigns a named material-neutral region to an explicit solid
scope. `PatchControl` assigns a named exterior or two-region interface to an
explicit face or edge scope. Derive those scopes from `BRepModel.solid_ids`,
`face_ids`, `edge_ids`, and `topology` through `GmshProvider.entity_scope`;
never reconstruct them from coordinates, import tags, or positions. This keeps
every patch adjacency tied to one exact B-Rep revision.

Uniform, curvature, and proximity controls compile into a resolved size field.
Proximity gaps are exact BVH nearest distances between samples of the source and
target scopes; opposite-normal controls require sample normals. Hard growth
limits enforce the edge-length-aware bound `h_i <= h_j + (rate - 1) |x_i - x_j|`
exactly. `size_field_metric` compiles a resolved field into isotropic metric
constraints bound to the same entity IDs.

`MeshMetricField` represents an SPD anisotropic metric whose eigenvalues are
inverse squared target lengths. Construction certifies every row against the
declared hard bounds, `1 / maximum_size**2 <= lambda <= 1 / minimum_size**2` and
`sqrt(lambda_max / lambda_min) <= maximum_anisotropy`, on the
`phydrax.linalg.verify_dense_properties` spectrum; only eigenvalue roundoff
(dtype epsilon times the row spectrum, hence condition-scaled at `lambda_min`) is
admitted, and a violating tensor raises `ValueError` instead of being repaired.
A field carries no gradation bound: requested gradation belongs to a
`MetricGradationPolicy` (certified by `MetricGradationEvidence`) or to provider
options such as `MmgOptions.gradation` and `OmegaHOptions.gradation_rate`.
`normalize_mesh_metric` applies an explicit `MetricNormalizationPolicy`: size and
anisotropy bounds, a target complexity `sum_i V_i sqrt(det M_i)` over vertex
volumes solved by the native bracketed root, and optional gradation. Untrusted
tensors enter as `MeshMetricSamples`, are symmetrized or projected only when the
policy requests it, and `MetricNormalizationEvidence` counts every repair and
clamp. `grade_mesh_metric` bounds growth with the physical law
`h_q <= h_p + (beta - 1) |pq|` or the metric-space law
`h_q <= h_p beta**l_p(pq)`: scalar gradation is an exact minimum-first
relaxation and anisotropic gradation uses Alauzet grow-and-intersect sweeps.
`MetricGradationStatus` distinguishes `CONVERGED`, `BOUNDS_CONFLICT`, and
`SWEEP_LIMIT`; convergence requires the independently measured
`maximum_violation` to meet `relative_tolerance`. `grade_mesh_metric` and
`normalize_mesh_metric` raise `MetricGradationError` carrying that evidence
instead of returning an executable field when hard bounds or the sweep budget
prevent the requested gradation.
`combine_mesh_metrics` intersects metrics by canonical simultaneous reduction
(order independent) and returns a `MetricCombinationResult`: the combined field
exists exactly when `successful`, while `MetricCombinationEvidence` reports an
empty intersection of the declared size intervals or every tensor row that
conflicts with the most restrictive declared bounds, so a contradictory metric
can never reach an adaptation route. Adaptation requests,
`BackgroundMetricControl`, and the Mmg and Omega_h plans accept only a certified
`MeshMetricField`. `interpolate_mesh_metric` is log-Euclidean,
`metric_edge_lengths` returns Riemannian edge lengths, and `lp_metric_from_hessian`
builds the Loseille-Alauzet `L^p` metric from a recovered Hessian
(`phydrax.discretization.fem.recover_hessian`). These operations prepare
requests; they do not certify that an external generator met them.

`LayerSchedule` declares physical layer thicknesses, ordered from the wall
outward. `BoundaryLayerControl` binds a schedule to a wall scope and one explicit
`BoundaryLayerRoute`:

- `EXACT_SWEEP` meshes a pre-partitioned straight slab cap to cap (wall, cap,
  and volume scopes);
- `CAD_EXTRUSION` is realized before meshing by
  `prepare_boundary_layer_extrusion`, which extrudes planar walls along their
  inward normals, splits the source by the exact prisms, publishes the
  partition, and returns the `EXACT_SWEEP` control that meshes it; it rejects
  prisms that leave their solid, overlapping slabs, and slab sides that would
  meet the core;
- `ADVANCING` grows native layers (`prepare_boundary_layers`) from an oriented
  triangle/quadrilateral wall mesh, or from closed BRep walls inside
  `GmshProvider`;
- `PROVIDER` lowers the schedule to Gmsh: a `BoundaryLayer` field on planar wall
  curves (geometric schedules), or boundary-layer extrusion of closed 3D walls.

For an independently authored native reference slab, `prepare_boundary_layers`
also consumes `EXACT_SWEEP` controls with an explicit
`sweep_control=SweepControl(SweepMapKind.EXTRUSION, schedule, map_id, axis=...)`.
The wall association and `source_domain` bind a real `PiecewiseLinearDomain`:
wall and cap scopes retain its complete exact endpoint triangulations, and the
volume scope covers its regions. The native sweep owns station coordinates
through the original schedule's cumulative thicknesses; it certifies the swept
carrier, verifies exact affine root charts, and assembles a `BoundaryLayerMesh`
with fresh wall-distance column measurements, affine-map validity, and lifted
periodic source orbits. It does not run advancing-layer optimization or copy an
advancing result's evidence. `certificate_limits` bounds the native sweep's
construction certificates. Exact sweeps retain their original cap; they do not
admit advancing obstacles, collision resolution, or transition pyramids.

When paired with a mapped `NativeLayerCoreSource`, these reference stations are
independent coordinates, not normalized physical data. The original advancing
mesh, its source bindings, and every actual interval endpoint remain unchanged;
the source-controlled column coordinate bank retains even one-ulp physical
variations. Reference and physical source/control/result identities may differ.
The owning prism chart uses axial coordinates `[0, 1]`, consistently with
`reference_cell_topology` and `swept_coordinate_element`.

Native advancing layers classify wall edges by dihedral angle, split columns at
convex ridges into fan columns and corner patches (`BoundaryLayerCornerPolicy.FAN`),
and point every column along the visibility-optimal direction (the minimum-norm
point of the convex hull of its face normals, a simplex QP solved by
`phydrax.optim`). Concave corners are accepted only while a visible direction
exists and the height stretch stays within `maximum_corner_stretch`; otherwise
they are rejected with the offending wall vertices. Heights are limited by the
smooth concave radius and by BVH-nearest medial-crossing probes; every layer is
certified by Bernstein validity and exact-predicate intersection against earlier
layers, the wall, adjacent surfaces, and obstacles. Collisions resolve only by
`BoundaryLayerCollisionPolicy`: `FAIL`, `TERMINATE_LOCALLY` (pyramid/tetrahedron
terminations, never below the first layer), `REDUCE_THICKNESS` (never below
`minimum_thickness_fraction`), or `MERGE` (mutually paired, face-to-face opposing
fronts share their midsurface). Failures carry the wall vertices and locations.
`BoundaryLayerMesh` holds the certified layer cells, the exact cap (quadrilateral
cap faces are closed by transition pyramids), and `BoundaryLayerEvidence` with
thicknesses measured as exact distances to the wall. Per-layer statistics cover
the surviving columns that carry each layer; `layer_active[k]` is `True` iff at
least one column carries requested layer `k`. A layer carried by no column (for
example after `TERMINATE_LOCALLY` stops every column early) has
`layer_active[k] == False`, NaN achieved, minimum, and maximum thickness, and NaN
for every growth rate touching it (`achieved_growth_rates[k]` is the ratio of
achieved thicknesses `k + 1` and `k`). Read layer support from `layer_active`,
not from NaNs; `terminated_vertex_count`, `reduced_vertex_count`, and
`merged_vertex_count` state why columns stopped short of or shrank the schedule.
`GmshProvider.fill_boundary_layer_core` tetrahedralizes between a closed cap and
the remaining boundary while keeping both fixed: fixed nodes are verified
bitwise and core faces must match the fixed triangles exactly; the merged mesh
carries `boundary-layer` and `core` zones and `wall`, `layer-core-interface`, and
`outer` patches. `PlanarBandControl` instead declares explicit side schedules over
a planar partition patch; `prepare_planar_bands` returns the per-layer partition
evidence.

`NativeLayerCoreSource` binds native core construction to the actual prepared
physical layers and their authored domain. Its `fidelity_source` remains a
dynamic, fixed-role, nontrainable source: array-bearing domain queries are not
hidden in static PyTree metadata. This intentionally changes the representation
and may change whole-container, binding, model, and archive IDs; historical
templates are not reused to conceal that change.

Durable source admission binds exact PLC geometry to the registered original
domain identity, revision, coordinate rows, scientific IDs, and deviation
bounds. Layer admission likewise binds the original fidelity source and
represented domain or parametric query domain. A self-certified foreign
authority is still foreign and is refused before theorem renewal. Core
association namespaces come from the bounded canonical PLC constraint owner,
including real crease edges within a shared facet group.

Source-bound chart fidelity uses the actual exterior boundary maps of volume
cells as well as surface cells. Complete original chart and scoped-face
coverage remain required; sampled proximity does not replace that proof.

Gmsh is optional (`pip install 'phydrax[meshing-gmsh]'`). Use
`GmshProvider` with a revision-checked reopenable `BRepModel` or solid
`BRepSource`. The B-Rep import or persistence call owns the required
`SpatialCoordinateContract`; `GmshOptions` does not own coordinates. Sessions
are host-side resources and must not enter JAX state. Use a context manager for
an explicit `GmshSession`; its cleanup does not suppress the original exception.
Open CAD faces use `BRepModel`; `BRepSource` retains its closed-solid invariant.
Required mixed families must actually occur in the generated result; permitting
mixed recombination does not guarantee that triangles will remain.

### Exact semantic B-Rep and planar routes

`BRepImportReport.source_digest` identifies the persisted CAD bytes;
`source_revision` additionally binds the coordinate contract and import policy.
Replacing source bytes invalidates an existing meshing plan even when the path is
unchanged. `BRepPartitionPlan` and `PlanarPartitionPlan` are geometry-owned
Boolean operations: they publish persisted B-Rep output, exact
`CADRevision`/`AssociationGraph` evidence, and named solid/face regions with
face/edge patches. Gmsh consumes that evidence; it does not repair, fragment, or
recover material meaning from names or proximity.

`RegionControl` and `PatchControl` are dimension-generic. A volume request binds
regions to solid scopes and patches to face scopes; an ambient-two surface request
binds regions to planar B-Rep faces and patches to exact B-Rep edges through one
`PlanarEmbedding`. The resulting planar `CellMeshingResult` has ordinary
top-cell zones and main-mesh edge patches, no synthetic `SurfaceModel` boundary.

`LayerSchedule` is explicit physical thickness data. `EXACT_SWEEP` boundary layers
admit only complete, straight prism sweeps that join unswept tetrahedra through
triangular caps. `PlanarBandControl` creates exact straight CAD strip partitions;
it rejects corners, junctions, collisions, insufficient clearance, and any
topology change. Pure-Q4 band requests require an entirely rectangular
transfinite closure; otherwise request an allowed triangular remainder.

Size bounds and growth become hard only when explicitly supplied. `target_size`
is a preference, not an unstated exact bound. Provider compliance records
per-control observed edges, hard-bound tolerances, and growth evidence.

For a selected volume-boundary scope,
`geometry.certify_source_fidelity(..., target_facet_ids=...)` compares the actual
global target facets with the selected authored patches of an original
`MeshingDomainBoundarySource`. The target facet list and occurrence-qualified
original source scope participate in certificate identity. A whole-surface
construction chart bank may supply that query: unselected records are ignored,
but missing or duplicate selected records and invalid trim chains still refuse.
Independent affine triangulations are compared by native exact halfplane
clipping, with nonoverlap, coplanarity and exhaustive exact projected measures;
the original continuous chart remainder remains in both directed bounds.
The proof shares the original remaining work and scratch allowances. Evidence
distinguishes actual coefficient work, retained canonical coefficient storage,
temporary CPython storage upper bounds, actual native clip work and observed
managed native peak bytes. Exact proofs can use zero distance samples without
claiming zero total work. Primitive telemetry unavailable from the exact-clip
ABI is explicitly marked unavailable rather than reported as a measured zero.
Work or byte exhaustion remains an unresolved, source-bound refusal.

`NativeMeshingProvider(NativeMeshingOptions(route))` meshes on one explicitly
selected native route and never switches routes after a failure. The options
carry only the route and its numerical schedule; sizes, fidelity, quality and
`MeshingLimits` (including work, cavity, geometry-query and scratch-byte
budgets) stay in the specification. Every route publishes only after the audit
and its `MeshCertificationSchedule` pass: the result carries the passed
`MeshCertificationReport` in `result.certification`, a `certification` trace
stage and a `MeshingEvidenceBinding`; a failed certification raises
`MeshingFailure` (`AUDIT_FAILED`, stage `certification`). Each source is a
revision-bound binding:

- `"planar_constrained_delaunay"` meshes a `NativePlanarSource` (a
  `PlanarMeshRegion` with hole loops plus optional embedded `SegmentMesh`
  constraints) for a planar `SurfaceMeshingSpec`. Uniform size controls may
  scope the region or individual source edges; they resolve into one graded
  size field (the smallest declared growth rate bounds its slope). Source
  edges are bisected to the local target, then exact meshcore constrained
  Delaunay refinement enforces the area and optional `MeshQualityTarget`
  minimum angle, and free edges are split to the local target within the work
  and scratch budgets. Certification proves global embedding and exact
  coverage and measure of the recovered loop chain, whose rounding from the
  source edges is reported. Constrained edges carry exact `PIECEWISE_LINEAR`
  associations with orientation, and loops and embedded constraints are
  published as patches and labels.
- `"implicit_surface"` meshes a `NativeImplicitSource`. The default
  `AdaptiveImplicitSurfacePolicy` discovers the surface by adaptive
  error-controlled octree refinement over the source lattice box (not
  differentiable); `ImplicitSurfacePolicy` selects fixed-lattice discovery,
  whose `NativeMeshingPlan.execute(state)` publishes the JAX realization in
  `result.geometry.coordinates` so observables differentiate with respect to
  design parameters along the fixed route. Certification bounds the two-sided
  deviation from the source through `ImplicitBoundarySource` against a
  `FeatureKind.SURFACE` protected feature's `maximum_deviation` (default: the
  target size). Bounds are certified only for established exact distances;
  otherwise they are recorded as sampled, and a hard feature refuses them.
- `"curve_arc_length"` meshes a `NativeCurveSource` (one-dimensional
  `BoundaryAtlas` charts or a `SegmentMesh`) for a `CurveMeshingSpec`.
  Intervals are placed at equal increments of the size density
  `|gamma'| / h`, solved with native bracketed scalar roots. A protected curve
  feature bisects intervals until their chord deviation meets its bound. Only
  declared `CurveJunction`s share vertices (a start/end junction closes a
  curve), and endpoints are never joined by proximity. Chord deviation is a
  certified two-sided bound `(b - a)^2 / 8 max |gamma''|` from outward-rounded
  interval evaluation of the chart's second-derivative program; charts without
  interval rules report sampled evidence.

Every route returns a `CellMeshingResult` only after the audit and
specification compliance pass; otherwise `MeshingFailure.evidence` reports the
stage, entities and requested-versus-achieved quantities.
Requested and achieved quantity banks remain independent named evidence;
diagnostic comparison joins only actual matching keys, never positional pairs
or fabricated zero values. Reports and exceptions retain the complete banks.
`MeshCertificateFinding.observations` is a separate immutable bank of unique
nonnegative integer scientific facts. Report keys preserve its
`observation:` prefix; resource requested/achieved banks retain actual
resource limits, requests and completed counts. Scientific SCI/action
observations are not reclassified resource counters or a global embedding
certificate. A surfaced original budget failure does not become a successful
route because its diagnostic is now complete.

Native source-bound surface refinement names the selected edge by its actual
endpoint rows. Its paired cavity replaces both adjacent triangles together,
even when the rounded chart midpoint is not exactly collinear with the old
internal edge. Every one of the four child charts must have exact positive
orientation, satisfy the existing source-normal guards, and conserve the
complete constrained outer boundary and reciprocal incidence. No snapping
tolerance or fitted source is introduced. A stale, constrained or illegal edge
request refuses unchanged; any later interior-point retry is a separate,
explicit generic request.

**Native host scratch ownership.** `NativeExecutionBudget` retains one original
CPU-native allowance across nested scopes; a nested scope cannot renew work,
geometry-query, cavity, scratch-byte, or deadline admission. Its scratch cap
bounds native managed allocations together with explicitly reserved,
conservative Python-owned storage bounds; it does not measure process resident
memory.

For an end-to-end route, preparation, generation and publication must retain
that same original root budget. Owning source-fidelity/coverage queries are
charged as source operations, separately from native primitive-query telemetry;
primitive counts cannot stand in for those charges. Execution phase records
describe only the phase actually measured, not Python construction, compilation
or a process-wide peak. An allocator-level contract alone is not evidence that
every public route has completed this lifecycle integration.

An ended source-preparation receipt is imported once through an actual child
scope; an already active root is borrowed, not restarted. Prior source duration
is an ancestor admission debit separate from the parent's raw elapsed clock.
Retained child `consumer_evidence` describes the original child and is not an
additional additive charge. Scoped source/child/parent checks exercise this
distinction and preserve unchanged scientific sources. They do not promote
unenforced public volume/envelope declarations or establish the original
full-volume acceptance campaign.

`NativeExecutionRecord.to_record()` validates and serializes the actual nested
preparation/consumer graph, retaining shared historical nodes instead of
reconstructing counters from a snapshot. Its strict JSON record leaves the
owning objects unchanged. Qualification result records carry these receipts
along with exact runtime enforced/unenforced limit sets; a missing receipt
remains absent, not a fabricated zero-work result. Receipt interchange alone
does not certify the scientific mesh or make an unenforced limit enforced.

An owning host consumer opts into the same allocator through
`execution.allocate_host_array(shape, dtype)`. This returns an uninitialized,
writable, C-contiguous NumPy array backed directly by native storage, with no
payload copy or second reservation ledger. Admission requires the currently
active scope on its creating thread. Supply a tuple of nonnegative Python
integer dimensions and an explicit non-object dtype; scalar and empty shapes
are supported. Boolean dimensions, negative dimensions, unaddressable
dimensions/storage/strides, and object-containing dtypes refuse before native
allocation. Top-level subarray dtypes are refused because NumPy would change
the requested shape. An explicit `np.bool_` element dtype remains valid.
Passing a NumPy scalar class or a parameterized `np.dtype` preserves that
scalar type in the returned `NDArray` annotation; other `DTypeLike` inputs
retain the generic host-array annotation and the same runtime validation.

The canonical native buffer-owner metadata request and the exact payload
request (`product(shape) * dtype.itemsize`, including zero) are both counted.
A refused request does not increase live bytes, peak bytes, or successful
allocation count. If owner metadata succeeded before a payload refusal, that
successful request remains legitimate peak/allocation evidence and is released
during rollback. Returned arrays and their shared-buffer NumPy/base/views keep
the exact native owner alive beyond lexical scope exit. Releasing the final
buffer reference releases its actual managed bytes, including collection on
another thread; it does not transfer admission to that thread.

`with execution.host_workspace() as workspace:` reserves genuinely owned host
storage without allocating a dummy native payload. Call
`workspace.set_bound(total_bytes_upper)` before allocating or growing that
storage, and keep the reservation alive for its full working lifetime.
Resizing is atomic: refusal preserves the previous bound. Scope exit releases
the reservation, including failure paths. Nested consumers share the original
pool cap with concurrently live native buffers.

`workspace.retain_owner(value)` admits actual immutable source/geometry owner
trees, deduplicates shared base owners and keeps their storage alive until the
workspace closes. Unsupported callbacks and cyclic owners refuse atomically.
`workspace.logical_unmanaged_bytes_upper` reports retained JAX leaf bytes as
logical, unmanaged storage; admitting that bound does not measure a device
buffer or compiler peak.

Exact coordinate preparation uses live-storage owners separately from
discardable temporary scopes. A live bound that grows inside a temporary scope
remains charged afterward, while discarded compositions and subdivision
workspaces are released. Neither shrinking storage nor leaving a temporary
scope restores consumed work.

Statistical tetrahedral insertion inspects its actual native source-authorized
cavity before admitting physical size/shape evidence and committing once.
The original `edge_split_point` source fraction can authorize its identical
bounded source-row carrier even when that carrier is not collinear with the
represented old edge. Its conflict cavity can therefore exceed two children
per incident tetrahedron. Inspection and commit bind the original fraction,
accepted-state generation and exact removed/proposed tuples; changed or stale
plans refuse. Ambient inspection buffers use the same native host allocator.
When source-authorized conflict preparation refuses, inspection can retain the
existing complete-star route instead. Both routes share the direct operation's
source, protection and surface admission; finite removed plus proposed cells
still fit the same original cavity allowance. Repreparation must reproduce the
exact inspected tuples before either route commits.
Sizing remains independently checked against the authored statistics and
tolerances after all actual edits.

The statistical proposal's nonsmooth trial capability is the public
`phydrax.optim.AbstractLeastSquaresTrialPolicy`, supplied with its paired
`trial_policy_id` and dynamic `trial_model_work_limit` to
`NonlinearLeastSquaresProblem`. The capability owns trial
scaling/model images and termination admission; it does not replace the linear
JVP used by the prepared Krylov solve. Actual invocations and refusals appear
in `LeastSquaresResult.method_evidence`. Unsupported least-squares methods and
smooth implicit/certificate routes refuse this capability rather than silently
discarding it.

The policy receives the dynamic prepared `JacobianLinearOperator` and remaining
model-work allowance and returns `phydrax.optim.LeastSquaresTrialResult`, not a
tuple. Its nine dynamic fields are `direction`, `scale`, `model_image`, `valid`,
`jvp_actions`, `vjp_actions`, `model_work_units`, `model_visits` and the Boolean
`resource_refused`. The returned direction drives the actual
proposal, step norm and prediction while the prepared linear Krylov J remains
unchanged. Actual executed receipts accumulate in method evidence as
`trial_policy_jvp_actions`, `trial_policy_vjp_actions`,
`trial_policy_model_work_units` and `trial_policy_model_visits`, with aggregate
`trial_policy_receipts_valid`. Invalid receipts stop before evaluating the
actual trial residual; their raw values are retained, not clamped or replaced
by inferred callback-count formulas.

Resource refusal is distinct from a non-descent direction:
`OptimizationStatus.RESOURCE_LIMIT` and `trial_policy_resource_refused` retain
that failure. `phydrax.optim.compose_active_gradient_bank` composes the complete
signed active bank, with `active_gradient_bank_work(D, A, K, B)` admitting each
whole bucket before gathers, Gram products and economy rank factorization.
`ActiveGradientBankResult` retains validity, resource refusal, executed work
and rank/projection evidence. A capacity/work refusal does not publish a
successful partial scientific frontier. Zero or deficient rank refuses this
direction; it does not establish stationarity, global feasibility or scientific
infeasibility. Actual source residual/publication tolerances remain independent.
Here `D` is parameter size, `A` the active-inequality bucket, `K` the equality
bucket and `B` the bank capacity.

The current native size policy identity is
`exact-linear-quantile/tie-barycenter/gap-scaled-component-rank-frontier/finite-source-trajectory-hard-shape-model/outgoing-events`.
Its finite-rank trial model, selected linear image, exact outgoing merit slope,
and actual residual evaluation have distinct roles. The floating
`phydrax.nonlinear.quadratic_event_bound` supplies event bounds, not exact
scientific source certificates. A selected generalized gradient of zero alone
does not establish successful Image termination.
Its component-rank frontier and finite physical source-trajectory candidates
retain each quantile residual separately; a satisfied literal-zero residual
cannot be traded for improvement of another statistic. Whole frontier capacity
overflow refuses rather than selecting a partial bank. Numerical rank thresholds
belong to the linear model, not a relaxation of scientific zero tolerance.
Earlier policy-contract receipts are not verification of a later source graph;
the current graph still requires its own behavior and original publication
acceptance evidence.
The finite trajectory model retains the original hard dihedral/radius channels;
it cannot admit a new violation in exchange for quantile improvement. This
does not turn the separately declared soft volume term into a hard scientific
constraint or relax the original native determinant threshold.

These mechanisms are not a qualification of the original Image, periodic,
CAD or two-million-work exact-coverage campaigns. Retained failures and resource
refusals remain failures until their original scientific acceptance gates pass
on the current source graph. A scoped contract test or finite-motion diagnostic
is not global convergence, source coverage or release evidence. Optional
external providers are explicit comparison/interoperability routes, never a
hidden native fallback.

For the original Image request, fresh-process source reopening at the unchanged
257-member/16-level/rank-eight archive defaults has passed, but sizing remains
unaccepted. Retained complete-frontier timeout and Neurofluid compliance/stalled
failures remain evidence; the newer hard-shape source run still resource-refuses
with unmet quantiles. Standalone typed metric-network reopening has also passed
with its actual original source/default archive controls. Neither source-only
nor metric-network cold identity is field/PDE continuation; those downstream
gates still depend on scientific sizing acceptance. Concurrent diagnostic
timings are not isolated performance evidence.

`phydrax.applications.neurofluid.NeurofluidTransportCheckpoint` is a typed
data owner for the original compartment source, certified result, unprepared
metric-network plan and transport parameters plus actual initial/history/final/
continued states, accepted flags and AD arrays. Its clock/cursor and original
physical controls bind the complete history; source, region, certification,
carrier and physics identities are validated. It does not retain a runtime,
cache or callback. This public data contract does not imply that the original
Image scientific producer, accepted full transport or fresh continued PDE
campaign has passed.

`host_storage_live_bytes_upper` and `host_storage_peak_bytes_upper` report these
conservative external bounds separately from the native allocator's six
measured counters. An upper bound is not a measured allocation or an RSS peak.

Consumers must allocate new owned host buffers through this API and explicitly
fill or write them; caller-owned source arrays are not donated or rebound.
Ordinary NumPy expression temporaries and Python metadata are not automatically
covered; their owner must supply and retain an explicit bound. Unknown
JAX/device/compiler temporary storage remains explicitly unenforced until its
owning operation provides a proven bound.
Neither entering a native scope nor converting an array to JAX measures or
bounds those allocations.

`ManifoldProvider` performs n-ary union, difference (first operand minus all
others), and intersection of closed, oriented triangular `SurfaceModel`
operands in the same physical frame. It preserves source-face ancestry and cell
tags through the backend's face relations. Named `vertex_properties` are carried
through the backend and published per face corner, each checked against
barycentric interpolation on its reported source face. Semantic
selections/interfaces require explicit transfer and are rejected rather than
silently dropped. Empty output is reported as a failure because
`CellMeshingResult` requires a nonempty carrier. Boolean topology changes are
nondifferentiable.

`OpenVDBProvider` ingests sparse voxel bricks in bulk and may re-distance the
isosurface with `OpenVDBLevelSetRebuild` before extraction. `PoissonProvider`
publishes native per-vertex sample density, optional density-quantile trimming,
an explicit thread count, and the declared (Neumann) boundary condition.

### Gmsh sizing, protection, generation, and remeshing

`ProximitySizeControl` lowers to Gmsh distance fields between its two exact
B-Rep scopes: the requested size is the local gap (sum of distances to both
walls) divided by `elements_per_gap`, clamped to the control's bounds and
combined with every other scalar request by minimum. With
`opposite_normals_only`, the closest chord must be normal to both scopes.
Compliance measures every edge against the exact CAD gap at its midpoint.

`BackgroundMetricControl` binds a `MeshMetricField` to every vertex of an
explicit affine simplex background mesh in the source coordinate contract and
lowers it to a Gmsh post-processing view. `ISOTROPIC` uses the smallest
directional size; `ANISOTROPIC` passes the full tensor and is the sole size
field for the BAMG surface algorithm (anisotropic volumes belong to
`MmgProvider`). Compliance reports every edge's metric length as Gmsh
interpolates the view.

Surface generation declares this field once on
`SurfaceMeshingSpec(background_metric=control)`. Gmsh validation and execution
consume that specification-owned field; a non-`None` provider-only surface
metric argument is rejected instead of overriding it. The external volume
comparison argument remains a separate admission/refusal contract; it does not
introduce a canonical native-volume metric field.

`ProtectedFeature` scopes on B-Rep corners and curves (and bounded faces in
volumes) are certified present in the canonical mesh with measured CAD
deviation. Free B-Rep curves and points are embedded into the unique host face
or solid containing them; unprotected free entities are discarded.

`GmshOptions` selects `GmshSurfaceAlgorithm`/`GmshVolumeAlgorithm` (including
`HXT`), `num_threads` for `General.NumThreads`, Netgen optimization, and
`GmshHighOrderOptimization` for geometry orders two through four. Sessions refuse
plans needing more threads than their `MeshingExecutionPolicy.parallelism`;
multi-threaded runs report nondeterministic runtimes, so deterministic
specifications require one thread. Order-three and order-four simplex maps are
resampled exactly at the canonical Lagrange nodes with shared entity nodes.
Compliance records Gmsh's native `minSICN`/`minSIGE` element quality.

`SurfaceRemeshingSpec` on a `SurfaceModel` (or a `CellMesh` with an explicit
coordinate contract) classifies sharp features and boundary curves per
`SurfaceReconstructionControl`, reparametrizes the discrete patches, and
remeshes them. Compliance bounds the two-sided vertex deviation and requires the
Euler characteristic, component count, and boundary loops to be preserved.

A `GmshSession` caches CAD imports by source revision and coordinate contract.
Every execution re-digests the persisted bytes; replaced bytes evict the entry
and fail as an invalid source. Gmsh imports a private verified snapshot.

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
refined = phx.meshing.execute_mesh_adaptation(
    phx.meshing.prepare_mesh_adaptation(
        result,
        phx.meshing.MarkedMeshAdaptation(np.asarray(result.mesh.blocks[0].global_ids)),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
        ),
    )
)
values = refined.transfer.apply(result.mesh.coordinates[:, 0])
coarsened = phx.meshing.execute_mesh_adaptation(
    phx.meshing.prepare_mesh_adaptation(
        refined.target,
        phx.meshing.MarkedMeshAdaptation(
            (), refined.target.mesh.blocks[0].global_ids, hierarchy=refined.hierarchy
        ),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
        ),
    )
)
```

`prepare_mesh_adaptation(source, request, policy=...)` binds one typed request
to one certified `CellMeshingResult` under one explicit `MeshAdaptationRoute`;
`execute_mesh_adaptation` runs exactly that route and never falls back to
another. Requests are `MarkedMeshAdaptation` (cells to refine and to coarsen by
global ID, plus the `hierarchy` returned by the previous adaptation),
`MetricMeshAdaptation`, and `RelocationMeshAdaptation` (a `MeshMetricField`
bound to the source vertices). Routes:

- `NATIVE_BISECTION`: Maubach/Stevenson tagged-simplex bisection of triangle
  (newest-vertex bisection) and tetrahedron meshes. The initial refinement edge
  is the longest edge (ties by global IDs); an incompatible initial labeling is
  rejected unless `BisectionCompatibility.UNIFORM_REFINEMENT` explicitly requests
  one barycentric compatibility refinement or, for host triangle meshes,
  `BisectionCompatibility.CONFORMING_CLOSURE` preserves the original labels and
  closes split edges through neighbor bisection. Device and periodic orbit routes
  refuse label-conforming closure. Every host closure batch is admitted against
  the cell, vertex, and work-unit limits before bisection, charging one work unit
  per bisection. Coarsening removes complete bisection patches (Chen–Zhang vertex
  removal) and restores the recorded cell, edge, and face IDs, so a refine/coarsen
  round trip reproduces the source topology exactly.
- `NATIVE_METRIC_2D`: planar triangle metric adaptation by edge split
  (metric length above sqrt(2)), collapse (below 1/sqrt(2)), metric-quality flip,
  and relocation, with deterministic independent-cavity selection. Orientation
  decisions use the policy's predicate mode; unresolved filtered signs reject the
  operation. The status is COMPLETE when the unit-mesh criterion holds, STALLED or
  PASS_LIMIT otherwise; `RelocationMeshAdaptation` moves vertices only.
- `DEVICE_BISECTION`: the marked bisection request executed as compiled
  fixed-capacity device passes (below); it commits the meshes, IDs, lineage,
  and hierarchy of `NATIVE_BISECTION` byte for byte.
- `DEVICE_METRIC_2D`: the planar metric and relocation requests executed as
  compiled device passes with FILTERED_DEVICE predicates (below).
- `MMG`, `OMEGA_H`: the configured provider's metric adaptation; the lineage
  records every entity as created/deleted and no vertex stencil is claimed. With
  an `association_transfer` the providers re-derive the B-Rep associations on the
  adapted target (see below); without one Mmg records the dropped associations as
  an adapter loss and Omega_h rejects the source.
- `HP`: tensor-product h-refinement or coarsening of a `FiniteElementHPEpoch`,
  with lineage projected by `project_hp_lineage`.

Protected scopes (including periodic boundaries) are never split, moved, or
removed; marks whose closure would split them are rejected and reported, with
status PARTIAL. Patches, zones, and labels are inherited through the lineage.
`MeshAdaptationResult` carries the certified target, `CellMeshTransition`,
complete `MeshLineage`, `VertexInterpolationStencil`, sparse P1
`FiniteElementTopologyTransfer`, route evidence (`BisectionEvidence`,
`LocalMetricEvidence`, or the provider result), compliance, and an optional
`MeshDistributionTransition`. `VertexInterpolationStencil.apply` resolves source
IDs once and applies one sparse gather; `as_transfer` certifies linear
reproduction and conservation claims. A `FiniteElementTopologyTransaction`
consumes the `MeshAdaptationResult`; rejection preserves the accepted state.
Unknown parentage remains unknown, never guessed from new positions; B-Rep
associations are re-derived by classification, not inherited.

`TargetMatrixOptimizationPlan` and `optimize_cell_mesh` optimize fixed-topology
triangle, tetrahedron, quadrilateral, hexahedron, prism, pyramid, and polygon corner
Jacobians, plus the face-centroid star simplices of polyhedra, against
target matrices with `phydrax.optim.minimize` (`ProjectedLBFGS` by default,
`NewtonTrustRegion` optionally). `MeshQualityObjective` selects the generalized
Knupp shape measure, shape plus size, metric alignment with a `MeshMetricField`,
or the Frobenius Gram-determinant energy; every objective rejects inverted
corners. Fixed vertices are eliminated from the parameters and stay
bit-identical; coordinate boxes are enforced by projection. Inverted input runs
the explicit Escobar untangling stages of `MeshUntanglingPolicy`; a stage is
accepted only when the inversion count reaches zero, the audit passes, and its
minimization converged. The optimized iterate is checked for inversions and
audited independently of the native termination status, and non-convergence is
never reported as optimized: a valid iterate is `OPTIMIZED` only when the
minimization converged, `VALID_NONCONVERGED` (accepted, certified) when it did
not and the plan sets `accept_valid_nonconverged=True` (which also admits
non-converged untangling stages), and `NONCONVERGED` (not accepted) otherwise.
`MeshOptimizationResult.result` is present exactly when `accepted`; every other
status returns the unmodified coordinates, and
`MeshOptimizationResult.minimization` carries the native termination evidence,
including the rejected iterate. `optimize_cell_geometry_coordinates` runs the
same route for user-defined high-order objectives; the objective must encode the
required curved-element validity, and `CellGeometryOptimizationResult` reports
`optimizer_status`/`converged` for the caller's explicit acceptance decision.
Derivatives of numerical objectives do not differentiate topology changes or
host-side acceptance.

`MeshMotionMonitor` assesses a moved fixed-topology coordinate proposal against its
reference mesh: the certified Bernstein lower bound of every cell Jacobian relative to
the reference certificate, per-cell mean-ratio degradation, displacement in units of
the smallest reference cell, and the caller's boundary residual.
`MeshMotionMonitorPolicy` thresholds yield a deterministic `MeshMotionDecision`
(`ACCEPT_MOTION`, `RELOCATE`, `REMESH`, or `REJECT`); uncertified cells always request
relocation first because an inverted mesh cannot seed a remesh.
`advance_mesh_motion` certifies accepted motion, relocates or untangles the free
vertices through `optimize_cell_mesh` toward the reference shapes on any crossing
(bounded by `MeshMotionMonitorPolicy.relocation_termination`; a relocation that
does not converge fails unless `accept_valid_nonconverged_relocation=True` admits
its valid iterate), and
escalates a still-failing proposal to a `MetricMeshAdaptation` (default metric: the
isotropic reference cell size) through the explicit `MeshAdaptationPolicy` route;
without a policy the advance returns the unexecuted request (candidate mesh and
`remesh_metric`, `accepted` false). The returned `MeshMotionAdvance` carries every
assessment and, for executed remeshes, the `CellMeshTransition` and adaptation
transfer consumed by the solver transaction.
Fixed-topology publication revalidates any original certification request on
the moved map, including its domain and source-fidelity obligations; a valid
Jacobian does not authorize movement outside that domain. Scientific association
scopes retain their exact IDs, row order, occurrence paths and source namespaces
after complete fresh support proof. Material and region-boundary evidence is
renewed by its owning transition, using the actual adaptation work limits and
the original certification limits rather than a fresh default allowance.
An admitted analytic implicit surface uses its original
`ImplicitAssociationTransfer`: motion calls both actual face propagation and
`remap_boundary`, retaining scalar-source revision, face-parent lineage, cell
tags, selections, interfaces and orientation provenance in a renewed
`SurfaceModel`. Canonical publication validates that model against the current
mesh/frame and recertifies the original source-fidelity request; a boundary
model is never silently dropped or replaced by an unbound scalar profile.

An adapted mesh reaches a running coupled problem only through one accepted
cross-owner rebind (`phx.lifecycle.CompositionRebind`). A certified conservative
`FiniteElementTopologyTransfer` (the adaptation transfer, or
`vertex_interpolation_transfer` with P1 coordinates and basis integrals) becomes a
topology-epoch transition with `transfer.epoch_transition(source_field,
target_field, source_epoch, target_epoch, source_measures, target_measures)`; its
claims come from the transfer's own certification. The transition's
`composition_transport(source_entry, target_entry)` reports the conserved content
and succeeds only if the staged field is the transfer image, so observations,
factorizations, and interface routes bound to the old topology must be
reprepared or the rebind refuses them (see the lifecycle section of the
numerical interoperability guide and `examples/adaptive_fe_fv_rebind.py`).

## Device adaptation epochs

```python
import jax.numpy as jnp


def near(mesh, center, radius):
    centroids = jnp.mean(mesh.coordinates[mesh.cells], axis=1)
    return jnp.linalg.norm(centroids - jnp.asarray(center), axis=1) < radius


policy = phx.meshing.MeshAdaptationPolicy(
    phx.meshing.MeshAdaptationRoute.DEVICE_BISECTION,
    device_policy=phx.discretization.AdaptiveSimplexPolicy(
        vertex_capacity=1 << 12, cell_capacity=1 << 13
    ),
)
prepared = phx.meshing.prepare_adaptive_simplex(result, policy=policy)
layout, state = prepared.layout, prepared.state
for center in ((0.2, 0.2), (0.25, 0.25), (0.3, 0.3), (0.35, 0.35)):
    marks = near(state.mesh, center, 0.3) & state.mesh.cell_active
    state = phx.discretization.refine_adaptive_simplex(layout, state, marks).state
    coarse = ~near(state.mesh, center, 0.45) & state.mesh.cell_active
    state = phx.discretization.coarsen_adaptive_simplex(layout, state, coarse).state
adapted = phx.meshing.commit_adaptive_simplex(prepared, state)
assert adapted.target.audit.passed
```

`prepare_adaptive_simplex(source, policy=..., hierarchy=None)` validates one
certified triangle or tetrahedron source on the host, computes the same
Maubach labels as `NATIVE_BISECTION` (or binds the supplied
`BisectionHierarchy`), resolves protection and organization classes, and pads
the epoch into an `AdaptiveSimplexState` of the policy's capacity bucket.
`AdaptiveSimplexLayout` is the static compile identity (dimension, ambient
dimension, vertex/cell/protected-edge capacities, closure and coarsening
bounds, precision); the source topology never enters it, so every epoch and
cycle in one bucket reuses one executable per entry point.

The state's `MaskedSimplexMesh` is the solver-visible layout: vertex and cell
slots with activity masks, positively oriented cell rows, and packed sibling
half-facets. Slot order equals global-ID order and slots are never reused inside
an epoch, so the compiled closure issues vertex and cell IDs by prefix sums in
exactly the host order: each closure round sorts the new refinement edges,
bisects the selected cells with the static Maubach templates, and selects every
cell holding a split edge until the fixed point (a `lax.while_loop` bounded by
`maximum_closure_iterations`). Marks whose own closure would split a protected
edge are found by one reverse-reachability fixed point over the union closure
and rejected, as on the host. Coarsening passes remove every unprotected
bisection vertex whose star is exactly the marked, class-uniform children of
its bisections, restore the parents under their original IDs, and retire the
children. Every call records its `AdaptiveSimplexStatus` flags in
`AdaptiveSimplexState.status_flags`, which accumulate over the prepared epoch.
Capacity overflow, the closure bound, a protected conflict, and a
certified-invalid child are terminal failures: the call rolls back every mesh,
topology, and numerical array but still records its flags, and every later
refine or coarsen call on that state is refused on device (state unchanged,
`AdaptiveSimplexReport.status` holding the terminal flags, zero operation
counts); recovery needs a new prepared epoch, for resources with larger
capacities. Children are checked with FILTERED_DEVICE orientation predicates;
unresolved signs set `NEEDS_HOST_RESOLUTION`, which stays applied.

`commit_adaptive_simplex(prepared, state)` performs one device-to-host transfer
and decides acceptance from the cumulative flags: a terminal flag raises
`MeshingFailure` (capacity or closure bound: RESOURCE_EXHAUSTED; protected
conflict: INVALID_SPECIFICATION; invalid geometry: QUALITY_REJECTED),
`NEEDS_HOST_RESOLUTION` requires exact host orientation predicates to certify
every committed cell positive (otherwise, or when meshcore is unavailable to
resolve a sign, QUALITY_REJECTED at stage `device-commit`), and a coarsening
stopped at its pass bound commits the valid partial topology with status
PASS_LIMIT (not converged). An accepted epoch runs the host edit assembly:
canonical `CellMesh`, organization inherited by
exact IDs, certification, `CellMeshTransition`, complete `MeshLineage`, sparse
P1 transfer, `BisectionEvidence`, and the next `BisectionHierarchy`, returned
as the `MeshAdaptationResult` the solver transaction consumes. Cell lineage
follows bisection-tree containment, so cells coarsened and re-created inside one
epoch still relate to their source cells. The next epoch is prepared from
`adapted.target` and `adapted.hierarchy`; with explicit capacities it reuses
the compiled entry points. `execute_mesh_adaptation` with the
`DEVICE_BISECTION` route runs one refine-then-coarsen epoch of a
`MarkedMeshAdaptation` and commits it.

`BisectionHierarchy` now retains its active forest, scientific block authority,
sibling records, retired-entity banks, and issued-ID cursors in the canonical
array-bearing PyTree. `hierarchy_id` fingerprints those exact values rather than
object addresses or a display serialization. The mixed-family hierarchy likewise
fingerprints its complete sibling, quotient-entity, layer-column, identity-cell,
and cursor content. This is an intentional representation cutover: affected
PyTree definitions, whole-container fingerprints, recipes, and archive content
IDs may change. Old layouts and IDs are not aliases, and restoration never
injects absent historical fields to preserve an obsolete fingerprint.

Multi-device epochs: `partition_adaptive_simplex(prepared)` places the policy's
`MeshDistribution` ownership into globally shaped, process-addressable states
on an `AdaptiveSimplexParts` device mesh. Refinement exchanges bounded topology
packets along prepared neighbor routes until conformity closes; bounded
distributed ordering assigns stable IDs without gathering global candidate
tables. A resource, protected-edge, or geometry failure rejects every part.

`commit_partitioned_adaptive_simplex` returns an ordered tuple of
`MeshAdaptationResult` values for this process's addressable partitions, rather
than one host-global mesh. Each target is a canonical `CellMesh` with owned
cells and an actually expanded neighbor closure. Its storage descriptor binds
global counts, stable per-degree IDs and ownership, canonical logical arrays,
and consumed collective subdivision evidence. Scientific identity excludes
capacity padding and placement; checkpoint leaves retain the actual masked
epoch and owner-local bisection forest. Exact uncertain orientation work stays
on the owning process, and any local resolution failure rejects publication
collectively. Dense export is an explicit bounded operation, not a prerequisite.

The admitted publication route requires a globally certified simplex source
without outstanding organization or periodic obligations, and exactly
representable binary64 dyadic geometry restrictions. Native exact predicate
batches reject rounded midpoints that cannot carry the recorded one-half
reference witnesses. Globally logical composed/restricted geometry for general
unrepresentable affine midpoints is not implemented; this is an explicit
missing capability, not an affine-complete certification claim. Refinement and
complete-family coarsening are admitted within the same prepared source epoch.
The route publishes
P1 constant/linear-preserving interpolation and concrete topology transitions
for `CompositionRebind`; it does not assert globally conservative field/history
remapping. Owner-local FE preparation consumes owned cell contributions and
certified vertex halos. FV preparation additionally requires the retained
physical-boundary and neighborhood-completeness witnesses and enough actual
closure for its stencil. Curved/organized
publication, cross-epoch forest preparation, and complete-family ownership
migration are separate unsupported gates, not implicit serial gather fallbacks.

Initial parametric surfaces also have a source-first, authored-patch ownership
path: shared source-curve parameter packets are agreed before each owner's
native chart CDT and physical refinement. Use
`prepare_distributed_surface_generation(source, specification, schedule, layout,
patch_owners, maximum_metadata_bytes=...)` with the actual `NativeSurfaceSource`,
`NativeSurfaceSchedule`, `DeviceGenerationLayout`, and an immutable prepared
process-group binding. Pass that artifact as `initial_partition` to
`NativeMeshingProvider.plan` on the same `"parametric_surface"` route; the normal
`plan.execute()` constructs and publishes the result. The provider binds the
artifact to the exact source, specification, schedule, ownership, and capacity
controls. The scheduling angle aim stays separate from a hard
`MeshQualityTarget`. Publication consumes an independent theorem of that actual
source domain and specification,
including chart/trim coverage, continuous source fidelity, physical controls,
unique ownership, reciprocal shared facets, and native cross-owner physical
intersection decisions. It publishes the canonical `CellMesh`/`CellMeshingResult`
through common-capacity, source-bound receipts, not a fictitious bisection
predecessor. The local audit remains local; artificial closure boundaries are
not relabeled as physical source boundaries. Rejected numerical findings remain
in `MeshingFailureEvidence.logical_findings` on their actual shards.
Before any owner-local result constructor runs, actual storage scientific-bank
bytes and global entity counts are admitted against the independent theorem.
A self-issued coordinate projection or unchanged cached geometry identifier
cannot authorize modified coefficient banks, even for a passive owner with
no resident cells. Independently restored buffers with identical scientific
content can be admitted; object identity alone is not the scientific check.
Every owner retains the complete authored patch and curve-label inventory.
Global stable-ID membership and scope identity remain the same on every owner;
locally absent records retain their globally nonempty scope and an empty local
view. Scope construction and validation consume prepared all-owner receipts,
so a local audit rejection cannot strand another owner in a global metadata
operation.

The exercised folded-plate case uses two real processes and reaches a P1
finite-element mass solve on the accepted surface. A source-point failure on
one process rejects both; physically overlapping authored patches reject both
and retain native triangle-pair findings. Authored material zones and their shared
interface retain the same global identities on both owners. An owner with no
authored patches still receives a genuine `CellMeshingResult`: its resident
coordinates, blocks, coefficient rows, graded entity sets, and local validity
samples are empty, while the globally positive inventory and independent
source theorem remain bound through actual receipts. No serial empty mesh,
dummy cell, proxy certificate, or `None` success is admitted. An actual stricter
local quality audit rejects publication on
both owners without a collective stall. This path does not provide arbitrary
within-patch distributed CDT ownership or distributed PLC cavity, segment, and
facet recovery. It makes no measured-performance or all-device construction
claim.

The `DEVICE_METRIC_2D` route runs split, collapse, flip, and relocation passes of
the planar metric adaptation as compiled fixed-capacity device passes: fixed-width
cavities, a deterministic independent set of non-overlapping cavities per round,
and FILTERED_DEVICE orientation predicates. An operation whose predicate is
unresolved is never applied and the pass reports `NEEDS_HOST_RESOLUTION`; there
is no host callback inside the passes. The commit reuses the host metric edit
assembly, so the result carries the same lineage kinds and vertex stencil as
`NATIVE_METRIC_2D`.

`prepare_device_metric_adaptation(source, request, policy=...)` binds a
`MetricMeshAdaptation` or `RelocationMeshAdaptation` to a `DeviceMetricState`
of the policy's capacity bucket; `adapt_device_metric(layout, state)` runs up to
`maximum_passes` passes in one compiled executable per `DeviceMetricLayout` and
returns a `DeviceMetricReport` (`AdaptiveSimplexStatus` flags, operation and
rejection counts, unit-mesh measurements); capacity overflow and invalid
geometry roll the arrays back, record the terminal flag in
`DeviceMetricState.status_flags`, and refuse every later call on that state.
`commit_device_metric_adaptation` applies the same acceptance as
`commit_adaptive_simplex` (terminal flags raise, `NEEDS_HOST_RESOLUTION` is
resolved by exact host orientation of the committed cells), performs one
transfer, and returns the `MeshAdaptationResult` with `DeviceMetricEvidence`; a
run that applied nothing is UNCHANGED, with convergence or stall in the
evidence (as for `NATIVE_METRIC_2D`).

## CAD association and high-order curving

```python
model = phx.geometry.brep_sphere(
    1.0, coordinate_contract=phx.SpatialCoordinateContract.si()
)
# An octahedron inscribed in the unit sphere, turned off its seam and poles.
c, s = np.cos(0.3), np.sin(0.3)
turn = np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
tilt = np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
mesh = phx.discretization.CellMesh.from_triangles(
    np.concatenate((np.eye(3), -np.eye(3))) @ (tilt @ turn).T,
    np.asarray(
        ((0, 1, 2), (1, 3, 2), (3, 4, 2), (4, 0, 2),
         (1, 0, 5), (3, 1, 5), (4, 3, 5), (0, 4, 5)),
        dtype=np.int32,
    ),
)
projection = phx.geometry.prepare_brep_projection(model)
policy = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8)
association = phx.meshing.associate_mesh_vertices(mesh, projection, policy=policy)
curved = phx.meshing.curve_cell_mesh(
    mesh,
    association,
    projection,
    policy=phx.meshing.HighOrderCurvingPolicy(degree=2),
)
assert curved.status is phx.meshing.HighOrderCurvingStatus.CURVED
```

A B-Rep `GeometryAssociation` classifies mesh entities on B-Rep entities of one
exact revision: every row names the entity dimension and index
(`source_dimensions`, `source_indices`, matching the `<revision>:<kind>:<index>`
identity), closest-point `parameters`, the projection residual, ambiguity, the
orientation relation of the entity's canonical vertex order to the B-Rep tangent
or oriented normal, and parent provenance (`parent_dimensions`, `parent_ids`,
`parent_association_id`, `GeometryAssociationProvenance`).
`associate_mesh_vertices` classifies each vertex geometrically: mesh-boundary
vertices on the lowest-dimensional B-Rep entity within the classification
tolerance, interior vertices of volume meshes in their solid.
`associate_mesh_entities` derives edge, face, and cell classes from the vertex
classes: a cell lies in the unique region containing its vertex classes, an
entity interior to one region inherits it, and boundary or interface entities lie
on the lowest B-Rep entity containing their vertex classes within the closure of
every adjacent class (ties broken by the centroid distance, otherwise ambiguous).

`propagate_association` carries a vertex association through an exact
`MeshLineage`: preserved vertices (B-Rep corners included) keep their class,
relocated vertices are re-projected, children inherit the class of the source
entity they were created on (edge children inherit the edge, face-interior
children the face) and are projected, swaps change no vertex class, and a collapse
is legal only when the kept vertex's class lies in the closure of the removed
vertex's class; an illegal collapse raises `AssociationPropagationError`.
Coordinates are not snapped: residuals (e.g. the sagitta of a split chord) are the
evidence. After provider remeshing with unknown lineage, `rederive_association`
classifies target boundary vertices only on the B-Rep classes (with closures) of
source boundary facets carrying the same facet organization (the patches, zones,
and labels Mmg references and Omega_h class IDs preserve), then projects them.
`BRepAssociationTransfer` binds a projection and policy for
`MeshAdaptationPolicy.association_transfer` and the Mmg/Omega_h providers: native
routes fix B-Rep corners, protect ambiguously classified edges (ambiguity rejects
the operation), separate cell, facet, and planar feature classes by B-Rep entity,
and certify the target with the propagated associations.

`curve_cell_mesh` creates Pk simplex (P2, P3) or Qk tensor geometry nodes in the
reference ordering of the discretization's Lagrange elements, shared between cells
by owning entity and position (independent of local orientation), classifies each
node through its owning mesh entity, and projects nodes on B-Rep entities of lower
dimension than the ambient space. Interior nodes relax through
`optimize_cell_geometry_coordinates`; the objective combines the Escobar-regularized
inverse mean-ratio distortion of the Jacobian relative to the straight-sided
element, sampled at the Bernstein control-point lattice of the determinant, a
displacement term, and the distance of constrained nodes from their tangent
spaces. Constrained nodes are re-projected after every round. A candidate is
accepted only when `certify_cell_geometry_validity` certifies every cell and every
constrained residual is within `residual_tolerance`. The initial CAD projection
is accepted on its own when valid; a relaxation whose minimization did not
converge cannot replace the accepted geometry unless
`HighOrderCurvingPolicy(accept_valid_nonconverged_relaxation=True)` permits it.
`HighOrderCurvingResult.relaxation_statuses` records every relaxation's native
termination status and `accepted_round` names the accepted candidate (0 for the
projection). Without an accepted candidate the result rolls back to the straight
geometry with the rejected candidate's evidence (`HighOrderCurvingStatus`,
including `ROLLED_BACK_NONCONVERGED`). `PeriodicCoupling` vertex pairs make target-side
high-order nodes the declared isometry of their source-side nodes
(`PeriodicCoupling.match_points` pairs the nodes). `verify_curved_geometry`
certifies an existing high-order geometry, such as a Gmsh high-order output,
without moving it.

## Polyhedral storage

Polyhedral topology uses packed incidence offsets/values. Exact-width
`PolyhedralBlock` groups coexist with standard cell blocks. Padding belongs only
in explicitly bounded execution worksets, not authoritative connectivity.
`prepare_polyhedral_worksets` rejects allocations beyond its entry budget.

Native PLC-restricted polyhedra retain authoritative material, facet and feature
associations. `PolyhedralMeshAdaptation(REGENERATE, complex_=..., sites=...,
weights=...)` uses `MeshAdaptationRoute.NATIVE_POLYHEDRAL`: source-stratum
correspondence and positive physical face/edge overlap carry organization, while
complete geometric common refinement carries cell inventories. Generator site
ancestry is never a nodal interpolation stencil. Volume adaptation must retain
the rejecting watertight audit policy. `examples/native_polyhedral_meshing.py`
executes nonconvex two-material generation, canonical regeneration, VEM/FV
conditioning and solves, and conservative physical/history `CompositionRebind`
before continued diffusion. This exercised planar-coordinate route does not
establish exact skew-site radical-plane geometry, all native work/byte bounds,
or periodic/layer combinations; those require their owning construction and
scientific-control evidence rather than projected or triangulated substitutes.

For scalar H1 VEM in two dimensions, periodic quotient layouts identify
scientific vertex/edge entities and retain edge parity, physical-boundary
traces and winding. Prepared local factorized actions scatter through the
quotient map; their transpose is the actual reversed gather/scatter action,
including nonsymmetric local factors, not an assumed symmetric application.
Scoped degree-one through degree-four preparation/layout/trace checks and the
transpose checks have passed. Scalar H1 evidence does not extend to a de Rham
complex.

In three dimensions,
`PreparedPolyhedralH1VirtualElement3D.bind_scalar_diffusion(cell_coefficients)`
binds positive scalar or per-physical-cell material coefficients to the native
factorized quotient operator, retaining its local source maps and scaling both
consistency and stabilization. `pinned_diffusion_operator(pin,
cell_coefficients=...)` declares a connected zero-value gauge;
`compatible_pinned_load(load, pin, compatibility_tolerance=...)` rejects a
nonzero-total physical load instead of hiding it at the pinned row. Zero
material coefficients refuse.

`assemble_cellwise_constant_load(cell_forcing)` accepts a real scalar or one
constant force per original physical cell. Each admitted source volume is
uniformly lumped over that cell's vertices once and gathered to quotient DOFs;
the load sum is the physical `sum(f_cell * volume_cell)`. This is explicit
vertex lumping, not exact higher moments, continuum quadrature or automatic
mean removal. Prepared sorted-arity buckets retain authenticated original
`cell_indices`, so no physical cell is omitted or counted twice.

`vertex_field_space(name)` exposes the actual scalar H1 quotient vertex layout.
`vertex_field_transfer(target, physical_transfer, geometry, field_name=...)`
composes scientific-ID lifting, the owning physical finite-element stencil
and representative restriction with its true transpose/Euclidean adjoint.
The scoped current test exercises an identity physical stencil only. This is
interpolation, not conservative transfer, arbitrary face/interior VEM transfer
or proof of nontrivial adaptation's independent continuum PDE acceptance.

The original standalone native periodic cube source closure has exercised
heterogeneous diffusion, gauge/refusal checks, exact source/RNE regeneration
and fresh-process continued diffusion from persisted material/history.
That scope retains original PLC, sites, weights, constraints and schedule
controls. It is not a certified-result lifecycle or three-dimensional de Rham
qualification, and does not close every periodic meshing campaign.

The scoped native proper-quarter-turn polyhedral FV workflow has passed actual
scientific quotient/source moments, reciprocal once-only flux and Euler state
frame/ledger/AD checks. Sparse trace preparation retains the current state and
three history banks with its raw transpose; frame exchange remains distinct
from physical source/boundary flux. Its public-certified default
257-member/16-level/rank-eight first-write and fresh cold recertification retain
both authentic reports, field/dynamics owners, all state/history and epoch/
density before continued Euler residual evaluation. Stale count/ID/binding and
one-ULP source changes refuse. This is the exercised proper-rotation contract,
not translation, screw periodicity or complete W11 qualification.

## Assemblies, distribution, and coupling

`MeshPart` retains a certified cell carrier, prepared tensor grid, point cloud,
or spline carrier. `MeshAssembly` organizes named parts without tessellating
compact carriers. Part identities include numerical geometry: moving a mesh
invalidates old distribution and coupling evidence even when topology is fixed.

`CellPartition` is the solver-neutral cell ownership contract. `MeshDistribution`
adds exact global IDs, owned rows, ghost residence, dependency routes, and
`MeshPartitionEvidence` (part weights, imbalance, edge cut, ghost replicas).
Its default ghosts are the `halo_width`-layer face-adjacency reach of each part,
found by breadth-first search over CSR cell adjacency and ordered by global ID.
`lower_finite_element` produces native FE phase/workset plans whose worksets
are padded to the largest part, never a parts-by-cells table;
`lower_finite_volume` produces a grid-revision-bound Cartesian FV decomposition.
The old FEM-specific partition class is removed, without an alias.

`prepare_mesh_distribution(part, policy=MeshPartitionPolicy(kind, parts))`
selects the ownership route with `MeshPartitionKind`:

- `MORTON` and `HILBERT` quantize cell centroids isotropically, order cells by
  curve code then global ID, and split the order into weighted contiguous
  ranges; every part is non-empty and, when attainable, part weights stay
  within one cell weight of the mean, so
  `imbalance <= 1 + parts * max_weight / total_weight`.
- `GRAPH` runs the native deterministic multilevel k-way partitioner of
  `phydrax.graph.partition_graph` on the face-adjacency cell graph (heavy-edge
  matching, recursive greedy-growing bisection, capacity-constrained boundary
  FM refinement) through the meshcore library. It needs no external engine;
  integral cell weights are used exactly and real weights are quantized, and
  the provenance records the balance status, weighting and backend identity.
- `METIS` is the explicit external comparison route: METIS k-way partitioning
  through its C ABI, loaded from `PHYDRAX_METIS_LIBRARY` or the platform loader
  path; absence raises `MetisUnavailableError`, and `metis_seed` seeds it.
- `PROVIDER` takes ownership computed by a provider such as Omega_h or ParMmg.

`MeshDistribution.prepare_finite_element_closures(...)` returns a
`FiniteElementClosurePreparation` containing `authority`, `programs` and
`execution_evidence`. It prepares actual whole-certified-source and partition
DOF authority before owner-local extraction; paired transfers include their
actual supporting cells under the original caps. Exact-source subset ownership
retains original quotient scientific IDs, group actions and winding. An ended
owned preparation supplies its genuine receipt; an external live root does not
invent an ended receipt. These API contracts are not proof of complete
refine/coarsen/distribution/cold lifecycle acceptance.
`CollectiveMeshEvidence.raw_source_blocks` retains the original compiled source
declaration axis; `publication_source_blocks` separately retains the actual
published epoch's source axis, including restored coarse roots. Both are
dynamic tuples of actual `CellBlock` owners. Geometry presentation regrouping
does not replace either namespace with current target blocks, and cold replay
checks the two original declaration identities separately.

`prepare_distribution_transition(source, target_part, lineage, policy=...)`
moves a distribution onto an adapted revision of the same named part. Preserved
and refined cells keep their parent's owner, merged cells the majority owner,
and created cells the majority owner of owned face neighbors. When that
ownership exceeds `maximum_imbalance`, the target is repartitioned by the policy
route and parts are renamed to the source ranks they overlap most, minimizing
migration. `MeshDistributionTransition` holds the target distribution with
rebuilt ghosts, `send`/`receive` sparse relations over rank-ordered message
slots, migration counts and volume, and `transfer`, which moves cell data
exactly as the migration would. Global IDs are never renumbered.

Conformal and periodic overlays validate node bijections. Periodic vector
traces include the isometry's rotation. `ContactCoupling` is frozen node-to-node
normal contact; `ContactCoupling.search` pairs each target node with its exact
BVH nearest source node inside a capture radius. `OversetCoupling` provides
donor interpolation and its transpose with explicit holes and partition-of-unity
weights; `OversetCoupling.search` locates receptors in a simplex donor mesh and
uses clipped barycentric weights. Overset coupling is interpolation, never a
conservative overlap remap (`conservative` is false). Searches that leave a
receptor without an admissible donor raise `CouplingSearchError`, whose
`CouplingSearchEvidence` reports every receptor's `CouplingSearchStatus`.

`OversetCoupling.from_field_query` retains the donor's actual prepared field
query and full coefficient layout, including high-order, mixed-block and
H(curl)/H(div) reconstruction. A fixed image route binds both `rotation=R`
and `translation=t` so every donor query point maps to its target vertex.
`value_action="polar-vector"` requires an exact Euclidean Gram identity of
the original encoded source and target image matrices, not merely approximate
orthogonality or cancellation to a relative identity. Native preparation
retains these original matrices in the coupling. Explicit
`value_action="contravariant-vector"` instead declares numeric affine
component push/pull; it makes no rigid-isometry, Piola or conservation claim.
Both vector actions transpose their actual linear operator before the owning
query transpose. Use `value_action="invariant"` for scalar or explicitly
invariant components. Value shapes are validated only after the scientific
action is declared; derivative-component queries require a separate tensor
action and are refused. This binds an already located field
query and never makes interpolation conservative.

Native `OversetPartSpec(..., image_rotation=R, image_translation=t)` registers
one authored support image of its existing certified source carrier:
`x_world = R x_source + t`. Both arrays are required together. The finite
encoded matrix must meet the near-rigid admission premise, but is treated as
an authored affine map unless its exact Fraction Gram identity proves an
encoded Euclidean isometry. No matrix is normalized or projected. Donor and
receptor geometry, explicit walls, mapped solid authorities and native B-Rep
membership use this same map. Original source certificates, entity IDs and
coordinate-axis identity are retained; no virtual image mesh replaces them,
and this does not assert periodic quotient/source equivalence.
Bounded donor search maps receptor-source sites into each donor's source
chart before invoking its actual locator. Pullbacks use the owning native
linear solve of the encoded map, retaining its conditioning/status rather
than assuming a numerically near-orthogonal matrix has inverse `R.T`.
Wall classification uses the same source-chart pullback. Wall/cell broad
phases outward-enclose exact requested coefficient solves of complete source
intervals; possible curved intersections remain explicitly ambiguous.
Affine-image donor measures include the original encoded determinant.
Native B-Rep membership uses the original query owner's bounded work route,
including its affine source theorem rather than a Newton/quadrature proposal.
Its actual owner operations debit the registration's one remaining
`maximum_wall_candidate_pairs` allowance alongside PLC/cell-wall work, across
all parts. Refusal maps to structured `RESOURCE_EXHAUSTED` evidence before
publishing a partial classification. Historical source-certificate maxima
are not renewed as additional per-part allowances.
If an original native execution scope is active, its actual remaining work
and source-evaluation counts also constrain admission; nested scopes cannot
renew their parent's allowance. Actual host owner work and source points are
charged once before publication, separately from numerical primitive telemetry.
Host query scratch uses the owning query's allocation budget; native
managed-pool free bytes are not misreported as host scratch or a process peak.
An exhausted native membership row is `RESOURCE_EXHAUSTED`, never hidden by
a boundary-contact or generic unresolved-region decision.

For image registrations, `prepare_overset_field_transfer` requires
an explicit invariant, polar-vector or contravariant-vector `value_action`,
never a guess from component count. Vector values are returned in the
target's stored source chart: the effective action solves
`R_target @ O = R_donor`, and multiplying target values by `R_target` gives
registered-world components. Exact matching authored poses permit algebraic
cancellation only; this does not equate scientific sources. The fixed
route retains donor query sites, target-source sites and world receptor sites
with exact part revisions, and its transpose reverses the declared action
before the owning FE/FV query transpose. Motion and restart retain these
authored poses and refresh coverage without changing wall/orphan priority.
Canonical Newton location, fixed donor-side selection, and FE basis/Piola/side
preparation execute as bounded compiled numerical actions. Host preparation still
checks the current source map, packet sites and supports; compilation does not
erase binding metadata or reuse donor evidence after motion. First preparation
and compilation remain part of the qualification's whole-lifecycle wall gate.

For real owner-local field traffic, call
`transfer.prepare_owner_local_exchange(coefficient_ids, execution_group,
axis_name="parts", message_capacity=...)` on every process. The prepared
exchange binds each native donor/fringe packet's owner to one actual
execution-group device per process. Stable nonnegative `int64` scalar
coefficient IDs follow the donor field's exact flattened FE/FV layout,
scoped by its field and revision. Supply ID metadata only for locally owned
donor fields and fields referenced by local receptor queries. Repeated
`exchange.apply(local_coefficients)` accepts only owned donor arrays and
returns only that process's receptor outputs; `exchange.transpose(local_duals)`
returns contributions to their actual donor owners.

Preparation exchanges unique requested support IDs through colored active-peer
routes and refuses absent owner IDs or a collectively exceeded
`message_capacity`. Runtime reuses fixed bounded `ppermute` value packets and
reverses those same routes for the exact transpose: no complete donor-field
gather or query-by-coefficient matrix is formed. The canonical overlay's
explicit invariant, polar-vector, or contravariant-vector image action is folded
into the bounded local scalar stencil, including its transpose; vector meaning
is not inferred from shape. The exercised two-process profile requests six P2
scalar coefficients, twelve two-component P2 coefficients, or one FV cell
average, while retaining unrequested donor coefficients locally. It covers
live coefficient refresh, an exactly encoded quarter-turn polar image,
identically shaped invariant components, and a numeric `.6/.8` affine image
under an explicitly declared contravariant-vector action. That affine action
claims no rigidity, Piola transport, or conservation. The same scenario proves
transpose duality and collective capacity/missing-owner refusals, and rejects
the nonisometric affine image when a polar-vector action is requested.

This sparse profile admits complete FE fixed routes and FV owners exposing a
bounded coefficient-linear stencil. Nonlinear or global-read FV reconstruction
owners are refused rather than silently gathering their full state. Geometry
and query metadata may still be prepared serially: this communication proof
does not claim distributed hole cutting or distributed donor search, and its
interpolation transpose is not a conservation proof.

`prepare_overset_conservative_state_transport(source_entry, target_entry,
states, content_tolerance=..., policy=...)` is a separate physical-overlap
route for real cell-average fields, material densities/fractions, and spatial
history registers on a moving mesh. Each state explicitly binds its source
cell topology and exact part revision. The route returns the prepared common
refinement and one `CompositionTransport` per scientific state meaning, with
independent per-component source/target inventory ledgers. Pass those transports
to `prepare_overset_motion_rebind` and publish only at an accepted boundary;
refused publication retains the original composition and caller arrays.
Categorical labels are not averaged, and incomplete physical overlap requires
an explicit exterior-content owner rather than silently filling newly exposed
volume or discarding outflow.

The native overset qualification's separate conservative profile moves an
interior vertex of the actual native mesh, preserves the authoritative physical
boundary, recertifies source associations/coverage/embedding and the original
hard request, then commits density, two-material fractions and spatial history
through the motion rebind. Its changed cell volumes and independently integrated
inventories establish moving **cell-average** conservation, not conservation of
the separate moving FE donor interpolation or of a moving physical boundary.
Interior and rigid native publication compute exact current validity, embedding and coverage
once under the original request's lowered certificate limits; both association
propagation and final acceptance reuse these owning prepared premises. Exact
planar support reuses only its per-proof authored segment bank and still charges
every original predicate visit against the unchanged support-query allowance.
Exact ordinate comparisons may reject a segment that cannot contact the point
or cross its winding ray before determinant arithmetic; all original segment
visits remain charged, and boundary endpoints retain their exact predicates.

The unchanged original native overset routine (resolution 4, capacity 20000,
three solver repetitions, two motion steps, wall/donor/overlap pair limits
500000) passed its 120-second whole-lifecycle gate in 105.2898 seconds.
Both motions exercised exact rollback, accepted state transfer and continued
Poisson solves; the native source/request/physical-field checkpoint was restored
without serialized prepared caches, and restart continuation matched the
uninterrupted solve exactly. Its measured residual/boundary error was
`9.3925e-14` against `0.05`; this is not an analytic L2 error, a conservation
claim for moving FE interpolation, or completion of the full adversarial corpus.

### Interface attachments

`MeshInterfaceAttachment(part, scope, association, geometry_entity_ids,
tolerance=..., region=...)` attaches exact entities of one named `MeshPart` to
declared authoritative geometry entities, for example the `BRepEntityId`
strings of one B-Rep edge or face. The scope may be a part-bound `MeshingScope`
or a `MeshPatch`, `MeshZone`, or `MeshLabel` certified on the part's carrier;
organization evidence is recorded in `organization_ids`. The witness is a
`GeometryAssociation` certified on the same carrier: every attached entity must
be resolved, unambiguous, classified on a declared entity within `tolerance`,
and every declared entity must be attached. An association of another carrier,
another entity set, or another geometry revision is refused; equal shapes or
names never substitute for it. The attachment does not classify anything
itself.

A sided attachment records which side of the authoritative oriented normal it
lies on. With `region=` (the cells of the attached side of a volume carrier),
every attached facet must bound exactly one region cell, and
`orientation = +1` means the region's outward normal equals the B-Rep face
normal (three dimensions) or the B-Rep edge tangent rotated clockwise,
`(t_y, -t_x)` (two dimensions); `-1` means it is opposed. A codimension-one
carrier (a surface mesh) declares instead with `carrier_side=InterfaceSide`
which side of its own cell orientation it represents. Without either, the
attachment is unsided. `require_current(part)` and
`MeshAssembly.require_attachment(attachment)` refuse a part whose revision
changed; moving or remeshing a part therefore requires a new attachment, while
redistributing its ownership does not. Attachments currently require a
certified cell carrier, since only cell results carry geometry associations.
See [Numerical interoperability](guides_numerical_interoperability.md#interface-bindings)
for the solver-level bindings that consume them.

## Learned proposals remain untrusted

Marking, size, metric, and coordinate proposals bind an exact certified source.
`project_mesh_proposal` applies deterministic bounds, protected-entity,
gradation, SPD, and displacement constraints. `prepare_mesh_proposal` invokes
trusted native refinement or optimization and returns a
`MeshProposalTransaction`. Safety audit and compliance are separate evidence;
rejection or rollback retains the accepted source.

Simplex marking proposals use native bisection, including tetrahedra and exact
restriction of supported curved coordinate maps. Other cell families require an
explicit qualified `native_policy`, such as `NATIVE_MIXED` for the admitted 3D
template families; a missing family route is not replaced by simplex refinement.
Native size/metric proposals use the admitted planar or tetrahedral metric route;
an explicit `native_policy`
can supply association transfer and route controls, but cannot bypass projection,
protected scopes, resource limits or route certification. Unsupported curved
metric/family combinations are refused, not flattened or sent to a provider.
Mesh promotion does not implicitly transfer PDE fields: consume the exposed
transition through the solver topology transaction with every required field,
material, history and prepared-artifact disposition.

`AbstractMeshProposer` is the neutral `DECISION` component slot
(`slot_semantic_id="phydrax.meshing.mesh-proposer"`). A proposer decides where
and how a mesh should adapt; it never produces a mesh. `LearnedMeshProposer`
maps one feature row per scope entity (sorted global-ID order) to a marking
score per cell, a size, metric tensor or source-contract coordinate vector per
vertex, and
`propose(source, features)` requires `MeshProposalFeatures`: the matrix is bound
to the exact source revision, entity scope, producing owner and scientific
column IDs. Raw arrays remain valid for differentiable `evaluate`, not for
execution. Stale features are rejected before model evaluation. The typed
proposal rejects non-finite values and stale scopes, and identical values from
any proposer project identically: protected entities,
size bounds, gradation, capacity limits, native refinement, safety audit, and
compliance all apply unchanged. The model is a dynamic child whose arrays stay
PARAMETER; `evaluate(features)` is the differentiable per-entity map for
supervised training. A model declaring ports binds only through the proposer's
declared `ports` (one feature-row port of shape `(in_size,)` and the proposal
value port) and an explicit `port_mapping`.

`propose(source, features, scope=..., key=...)` accepts an explicit JAX key
for scientifically addressed stochastic evaluation. Omitting the key retains
deterministic evaluation. Addressed evaluation folds both halves of each
`int64` entity ID and the complete address digest, so row order and partition
do not select different random streams for the same scientific entity.
The proposer's `model_id` binds its current dynamic model content and declared
semantics; its authored `proposer_id` remains a label. The returned proposal's
`proposer_id` instead identifies the actual evaluation, including model,
features, source/address and key data. None of these identities certifies the
proposal's scientific acceptance.

Supervised marking targets come from native estimators, reordered to the
proposal scope's sorted global IDs: `FiniteElementDWRIndicators.absolute` from
`phydrax.discretization.fem.local_dual_weighted_residual` for goal-oriented
finite-element refinement, and the `refine_mask` (or the per-channel
`indicators`) of the high-enthalpy AMR indicator evidence for
aerothermodynamic refinement. Size or metric targets are likewise native
projected fields; a learned proposer imitates them but is always re-certified.

## Solver-aware decisions and fixed epochs

`SolverAwareDecision` admits a finite set of prepared `SolverAwareCandidate`
routes to an explicit `DecisionBudget` physical-error or certified QoI target.
Estimator owners supply `PhysicalErrorEvidence` with separate field, geometry,
algebraic and transfer contributions; field p-order does not imply geometry
fidelity. `RouteFeasibility` names actual family/geometry/field layouts, compiler
admission, geometry/topology certificates and complete composition dispositions.
`MeasuredAdaptationCost` records observed preparation, compilation, solve,
transfer, independent reanalysis and decision time plus measured peak memory.
It is not a predicted price or proof that repeated local edits attain tolerance.
Failed alternatives count toward the campaign wall budget.
`SolverAwareDecision(..., observed_campaign_seconds=...)` accepts the actual
elapsed campaign cost, including failed or coincident attempts. It must be
finite, nonnegative and at least the sum of uniquely priced candidate costs;
omitting it uses that priced sum. Exceeding `maximum_wall_seconds` refuses all
candidates. No selected route means target failure, even if mesh quality
improved.

The existing `examples/hp_metric_order_adaptation.py` exposes two bounded
physical workflows (run with `JAX_ENABLE_X64=1`):

```sh
python examples/hp_metric_order_adaptation.py --p-degree 3 --with-history
python examples/hp_metric_order_adaptation.py --metric
```

The first transports named physical history fields through h/p trials; the
second performs native triangle metric adaptation, independent physical
reanalysis, rejected-trial rollback and fixed-transfer JVP/VJP checks. Scoped
current smokes observed L2 target improvement, history/content checks and
independent transfer derivative checks. They do not establish cold-performance
superiority, embedded/rational geometry coverage or complete W14 qualification.
Reported memory is shared process-lifetime peak RSS, not a candidate allocation
delta; ordered repeat timings are samples, not uncertainty or superiority.

Changing geometry order toward an authoritative source is an explicit physical
domain change, not an exact restriction of the old straight mesh.
`prepare_source_geometry_realization` binds the same material-reference complex
to separately certified source and target embeddings, encloses the actual map
displacement and cell measures, and does not claim common-world coverage.
`GeometryRealizationMeshAdaptation` executes that binding through
`NATIVE_GEOMETRY_REALIZATION`; its `GEOMETRY_REALIZATION` transition preserves
topological lineage while renewing geometry and source certificates.

`prepare_source_realization_field_transfer` requires declared semantics:
`"intensive"` carries material-reference values and history;
`"conservative-density"` preserves physical inventory using the changed cell
measures, not unchanged constant density; and `"material-compatible"` retains
compatible cochains with the target map's Piola action. The finite-element
transaction selects these through `surface_chart_semantics`. A `GEOMETRY_ORDER`
trial must undergo independent physical reanalysis and remains unpublished until
the solver-aware decision admits it. Changing the source realization invalidates
fixed-epoch derivative evidence even when corner topology is unchanged.

Solver publication still requires independent physical reanalysis and one
`CompositionRebind`. The finite-element transaction's `composition_rebind`
callback attaches running solver artifacts, histories and RNG state to the core
mesh/field/material rebind without replacing its transports. Decision admission
checks the exact staged composition. Rejected transfer or reanalysis retains
the accepted boundary and all state.
For native mesh transitions, `field_transfer(transferred_target_fields,
adaptation, args)` may refresh the canonical transferred initial guess with the
actual independently solved target state. Every target field must be returned
with its prepared shape and finite values. The rebind certifies the intermediate
remap image separately, then records the solved-state update; it never borrows
remap-image evidence for different solved coefficients. Declare each field's
role through `surface_chart_semantics`: `"intensive"` and
`"material-compatible"` refreshes do not inherit the remap's content badge;
`"conservative-density"` requires an actual prepared content ledger and retains
its final content check. A nonconservative high-order or Piola field action
cannot acquire conservation merely through that declaration. Undeclared fields
retain only the owning remap's actual content obligation. A generic P2 solved-
state refresh may pass independent physical reanalysis without any conservation
claim. The intermediate remap must pass in every case, and independent physical
reanalysis still gates the one publication boundary. Declared semantics are part
of the transaction identity.

`FixedEpochDerivativeEvidence` composes existing owner-qualified geometry,
PDE and transfer derivatives. It gates the numerical JVP/VJP operators rather
than inventing them. The qualifications' `numeric_revision_id` must equal the
exact `epoch_id`; the three routes are exactly geometry, PDE and transfer and
must match `execution_plan_id`. Parameter identities and positive event margins
remain required; retopology, acceptance, feature classification, Boolean
branches and donor changes invalidate the record. These events do not
carry a smooth topology gradient. Comparative cost-to-target and derivative
qualification must be demonstrated for each claimed workflow.

Primal convergence alone does not establish JVP/VJP accuracy.
`LinearSolvePolicy.derivative_solve` supplies independent
`LinearDerivativeSolvePolicy` tolerances for the native implicit derivative
solves; tightening those tolerances leaves the primal solve policy unchanged.
Retain native convergence, true-residual and iteration/matvec evidence together
with finite-difference and adjoint-duality checks, rather than relaxing the
physical derivative acceptance gates.

The native design qualifier prepares one `phydrax.linalg.PreparedLinearization`
per fixed function and parameter point, retaining its primal, JVP and transpose
actions. Its two finite-difference perturbations independently realize the
authoritative geometry and require successful native PDE solves. Geometry, PDE
and transfer comparisons reuse those same admitted numerical states, not a
corner approximation or a second solver. Reuse does not relax derivative
tolerances or make topology events differentiable.


## Additional optional backends

| Provider | Dependency | Supported execution boundary |
| --- | --- | --- |
| Mmg | Mmg 5.8 libraries plus the packaged persistent worker | Simplex metric, level-set, and Lagrangian adaptation through Mmg2D/MMGS/MMG3D with region/boundary references and field interpolation |
| fTetWild | `phydrax[meshing-ftetwild]` | Robust surface-to-tetrahedron generation; sampled boundary-envelope evidence |
| Poisson | `phydrax[meshing-poisson]` | Open3D screened Poisson reconstruction from oriented points |
| OpenVDB | Native `openvdb` Python binding; conda-forge provides it | Existing sparse voxel field to isosurface, with explicit background semantics |
| Omega_h | Omega_h, built either serially or with MPI, plus the packaged persistent worker | Simplex metric adaptation with class preservation, field transfer, ownership, and ghost residence |
| VoroCrust | Source-built mesher plus the packaged persistent extraction worker | Surface sampling and explicit face-defined Voronoi cells |
| TIOGA | MPI-enabled TIOGA plus the packaged persistent collective worker | Overset hole cutting, moving-part updates, and donor/receptor interpolation between affine cell parts |

fTetWild accepts one soft `UniformSizeControl` whose scope exactly matches the
complete requested source boundary. It rejects patch controls, region controls,
seeds, layers, periodic constraints, and local sizing rather than weakening or
inventing their semantics.

External topology-changing providers are nondifferentiable. Mmg and Omega_h
output IDs identify the new revision; unknown ancestry is not replaced with a
nearest-neighbor transfer. Only explicitly declared fields are transferred, each
with method evidence: Mmg locates every output vertex in the source simplices
and interpolates P1 barycentrically (projection outside the source is counted
and bounded), and Omega_h applies its own linear or conservative transfer with
integrals reported before and after. Omega_h output metrics are checked against
the input field's hard size and anisotropy bounds after provider gradation; a
violation rejects the provider result instead of widening those declarations.
fTetWild's boundary check samples vertices and centroids and is not a continuous
Hausdorff certificate. Poisson reconstruction does not invent CAD associations.
OpenVDB must know how inactive and out-of-domain voxels are extended.

Native provider worker sources ship with every distribution: under
`native/providers` in the source tree and source distribution, and inside the
installed package in a wheel. `native_provider_source_path(provider)` (from
`phydrax.meshing.providers`, for `"mmg"`, `"omega_h"`, `"tioga"`, or
`"vorocrust"`) returns the provider's standalone CMake project in either layout;
its sibling `common` directory holds the shared header-only protocol. Each
project builds one persistent worker executable against a separately installed
upstream library; importing `phydrax` neither compiles nor launches them. Build
and link each worker with the same C++ compiler and, where applicable, the same
MPI implementation as its upstream library. Install the executables into one
prefix and either put its `bin` directory on `PATH`, set the provider's worker
variable (`PHYDRAX_MMG_WORKER`, `PHYDRAX_OMEGA_H_WORKER`,
`PHYDRAX_TIOGA_WORKER`, `PHYDRAX_VOROCRUST_WORKER`), or pass the executable
explicitly. The build commands below resolve the source directory with:

```console
source_of() {
  python -c "import sys; from phydrax.meshing.providers import native_provider_source_path; print(native_provider_source_path(sys.argv[1]))" "$1"
}
```

A provider launches its worker lazily and reuses that process across calls
until `close()` (providers are context managers), a failure, or the worker
policy's call bound. The worker reports its exact runtime identity (library
release, commit, build options, MPI ranks) once at startup; no version process
is spawned per call. Arrays travel through a binary exchange directory: one
little-endian NPY file per array plus a canonical `manifest.json` listing name,
dtype, shape, and payload SHA-256, verified and memory-mapped on receipt. Every
call is bounded by `MeshingLimits.maximum_wall_seconds` and
`maximum_data_bytes`; worker memory is limited by an address-space limit where
the host enforces one (Linux) and otherwise audited against the peak resident
size the worker reports, and runtime evidence names which applies. Timeouts,
exits, protocol violations, and native rejections surface as `MeshingFailure`
with the worker log tail; the raw worker evidence is kept on `__cause__`.

For an installed Mmg 5.8 CMake package (shared libraries):

```console
cmake -S "$(source_of mmg)" -B build/phydrax-mmg \
  -DCMAKE_PREFIX_PATH=/path/to/mmg-install
cmake --build build/phydrax-mmg
cmake --install build/phydrax-mmg --prefix /path/to/phydrax-native
```

`MmgProvider` adapts a certified `CellMeshingResult`. Blocks and cell zones
become Mmg region references, facet patches and zones become boundary
references, and all of them are rebuilt by name on the adapted mesh; required
vertices/edges/faces and ridges are `MeshingScope` values on the source. An
exactly isotropic `MeshMetricField` is sent as scalar sizes, any other as a
tensor, and the evidence names the representation. `MmgLevelSet` discretizes a
vertex level set into interior/exterior labels per region and a contour patch.
`MmgLagrangianMotion` requires Mmg built with `-DUSE_ELAS=ON` against the ISCD
LinearElasticity library; without it the worker refuses with
`UNSUPPORTED_CAPABILITY`. Mmg's install records no runtime path to
LinearElasticity, so configure such an Mmg with
`-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON`. Mmg 5.8.0's `MMG2D_Set_vectorSols`
stores 2D vector solutions one vertex slot away from where its Lagrangian code
reads them; the worker writes the layout the motion code reads. Configure with
`-DPHYDRAX_MMG_WITH_PARMMG=ON` and ParMmg (plus its own Mmg install) on
`CMAKE_PREFIX_PATH` to build the collective `phydrax-parmmg-worker` for
tetrahedral remeshing under an MPI launcher. The worker imports ParMmg's CMake
package: the ParMmg 1.5.0 release installs none, so build ParMmg at upstream
commit `d2eddc5` (1.5.0 plus its package export) or later.

For an installed Omega_h CMake package:

```console
cmake -S "$(source_of omega_h)" -B build/phydrax-omega-h \
  -DCMAKE_CXX_COMPILER=/path/to/mpicxx \
  -DOmega_h_DIR=/path/to/omega-h/lib/cmake/Omega_h
cmake --build build/phydrax-omega-h
cmake --install build/phydrax-omega-h --prefix /path/to/phydrax-native
```

Use the MPI compiler wrapper as `CMAKE_CXX_COMPILER` when the installed Omega_h
was built with MPI. The Omega_h worker stays alive across calls, one MPI world
per rank count. Blocks, zones, patches, and labels on cell and facet dimensions
become Omega_h class IDs and are rebuilt by name after adaptation; other
boundaries are classified by `OmegaHOptions.feature_angle`. `OmegaHOptions`
passes AdaptOpts quality and length targets verbatim, and evidence reports the
achieved quality and metric edge-length extremes. `OmegaHField` declares LINEAR
vertex or CONSERVE cell-density transfer. Distributed runs return per-rank
`OmegaHPartition` ownership, ghost, and global-ID arrays validated without a
per-entity merge; `gather=True` also certifies the global carrier and a
`MeshDistribution`.

TIOGA must first be installed with its CMake package metadata and global node-ID
support. Its package publishes neither a release version nor its unique-ID
option, so the worker requires both explicitly:

```console
cmake -S /path/to/tioga -B /path/to/tioga-build \
  -DCMAKE_CXX_COMPILER=/path/to/mpicxx \
  -DCMAKE_INSTALL_PREFIX=/path/to/tioga-install \
  -DTIOGA_HAS_NODEGID=ON -DBUILD_SHARED_LIBS=ON
cmake --build /path/to/tioga-build
cmake --install /path/to/tioga-build
cmake -S "$(source_of tioga)" -B build/phydrax-tioga \
  -DCMAKE_PREFIX_PATH=/path/to/tioga-install \
  -DPHYDRAX_TIOGA_REVISION=<exact-release-or-git-commit> \
  -DPHYDRAX_TIOGA_ENABLE_UNIQUEID=<ON|OFF, matching TIOGA_ENABLE_UNIQUEID>
cmake --build build/phydrax-tioga
cmake --install build/phydrax-tioga --prefix /path/to/phydrax-native
```

The worker uses the MPI compiler wrapper recorded by the TIOGA package and
refuses a different one. TIOGA is BSD-3-Clause (relicensed upstream from
LGPL-2.1; see its COPYRIGHT). The worker links the caller-installed library and
reports its caller-supplied exact revision; Phydrax does not redistribute TIOGA.

VoroCrust does not install CMake package metadata. Build a serial CPU checkout,
then point the worker at that exact source and build tree:

```console
cmake -S /path/to/vorocrust -B /path/to/vorocrust-build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DVOROCRUST_ENABLE_OPENMP=OFF \
  -DVOROCRUST_ENABLE_MPI=OFF -DVOROCRUST_ENABLE_EXODUS=OFF \
  -DVOROCRUST_TPL_ENABLE_KOKKOS=OFF \
  -DVOROCRUST_TPL_ENABLE_KOKKOSKERNELS=OFF \
  -DVOROCRUST_TPL_USE_LAPACK=OFF \
  -DVOROCRUST_TPL_BUILD_OPENBLAS=OFF
cmake --build /path/to/vorocrust-build --target vc_mesh libVCMesh
cmake -S "$(source_of vorocrust)" -B build/phydrax-vorocrust \
  -DVOROCRUST_SOURCE_DIR=/path/to/vorocrust \
  -DVOROCRUST_BUILD_DIR=/path/to/vorocrust-build \
  -DPHYDRAX_VOROCRUST_REVISION=<exact-release-or-git-commit>
cmake --build build/phydrax-vorocrust
cmake --install build/phydrax-vorocrust --prefix /path/to/phydrax-native
```

The worker refuses VoroCrust builds with optional MPI, Kokkos, LAPACK, or
Exodus linkage whose transitive build-tree dependencies cannot be reconstructed
safely. It detects and links OpenMP when the upstream build enabled it. Pass the
`vc_mesh` executable and optionally the `phydrax-vorocrust-worker` path to
`VoroCrustProvider`; `vc_mesh` runs once per call as a bounded subprocess and
extraction reuses the persistent worker. Its radius control is the backend
sphere-sizing bound, not a guaranteed edge length. No material identities are
inferred from seed colors. Before extraction, output is bounded per emitted
cell: each of the m interior cells is a convex polytope with at most n - 1
facets for n seeds, so Euler's formula bounds it by 2(n - 1) - 4 vertices and
6(n - 1) - 12 face-loop entries; exceeding the limits refuses the run.
Backend vertices that differ only within `relative_merge_tolerance` (default
1e-9) times the output bounding-box diagonal are normalized to the lowest
source vertex ID; coordinates are not averaged. Real VoroCrust output leaves
some shared corners unwelded up to a few 1e-11 relative. Collapsed alias faces
and merged vertex counts are reported in compliance. Set the tolerance to zero
for exact aliases only. Upstream VoroCrust exposes no bounded native output
allocator or callback. Runtime evidence therefore records native output
entity/connectivity preallocation as unenforced. Set
`VoroCrustOptions(require_native_output_preallocation=True)` only as a
fail-closed capability requirement; execution returns `UNSUPPORTED_CAPABILITY`
instead of pretending that preallocation was enforced.

TIOGA distributes complete named parts among MPI ranks and preserves part-local
ID namespaces. It does not partition one part or support curved/polyhedral donor
cells. `TiogaProvider.move` updates the coordinates of named moving parts in the
resident registration and reruns connectivity without restarting the worker; a
lost session or a superseded registration fails explicitly instead of
registering again. Runtime-specific MPI launcher flags belong in deployment
configuration, not mesh semantics.

Run native benchmarks with
`python -m tools.meshing_benchmarks --resolution 8 16`. Add
`--gmsh-semantic` to run real semantic Gmsh scaling at 2, 8, and 32 regions;
each result contains raw partition, plan, and execution samples plus ranges.
Use `--case <name>` for one scaling case: `cad-partition`, `planar-semantic`,
`layout-decode`, `planar-band`, `hybrid-slab`, `gmsh-semantic`,
`implicit-discovery`, `lbvh`, `overlap-pairs`, `incidence`, `supermesh`,
`remap`, `device-bisection`, `host-bisection`, `local-metric`, `optimization`,
`predicates`, `boundary-layer`, `high-order-certification`, `repartition`, or
`provider-worker`. The `device-bisection`, `host-bisection`, `local-metric`,
`optimization`, `predicates`, `boundary-layer`, `high-order-certification`, and
`repartition` cases report host preparation, lowering, compilation, cold and
warm execution, compiler temporary and output bytes, and retained bytes; a
host-only route is timed as host stages while its compiled consumer (transfer
application, quality or geometry evaluation) carries the compiler measurements.
Layout decode is explicit because its parser dependency is optional, and
`provider-worker` (cold launch versus persistent session) needs a real provider
worker; without one it returns a `missing-dependency` record.

`--case native-family-lifecycle --family-case <name>` selects a canonical
native family requirement; the parser derives its choices from that requirement
inventory and defaults to `dual-hex`. The
`curved-quarter-extrusion-quad` case retains the original rational quarter
extrusion source and constructs pure quadrilaterals in ambient dimension three.
Its physical oracle is `u=x[:, 0]**2` with intrinsic forcing `-2`, not the
ambient three-dimensional Laplacian `-6`. Original source controls, weights,
knots, degrees and scientific IDs remain authoritative.

Family phase evidence keeps lowering, compilation, first execution and warm
execution separate. Compiler temporary, output and code sizes are distinct
from logical retained storage; unavailable measurements remain
`None`/unmeasured, not zero. These interfaces describe the benchmark contract,
not a positive receipt or qualification of every listed family.

The original structured transfinite-quad workflow has passed its 16-to-64-to-16
lifecycle with original source/history, Q1 native PDE and FV consumers, default
257-member/16-level/rank-eight fresh cold reopening and four resource refusals.
That is one exercised family workflow, not completion of all W10 families.
The observed parent/child code identities differed during ongoing integration;
this receipt is not stable-release identity or isolated-performance evidence.

`python -m tools.meshing_qualification --scenario gmsh` proves native-BRep
replay identity, stale-source rejection, exact multi-region zones/patches, and
solid-scoped sizing evidence. The qualification command also names
cad-partition, composite-heat-3d, planar-semantic, composite-heat-2d,
layout-stack, layout-observable, planar-bands, hybrid-layer, and
hybrid-diffusion scenarios, as well as Manifold, Mmg, fTetWild, Poisson, and
VoroCrust. VoroCrust accepts explicit `--executable` and `--worker` paths.
The substrate scenarios `bisection`, `device-adaptation`, `metric-adaptation`,
`gmsh-metric`, `supermesh`, `remap-p1`, `cad-curving`, `boundary-layer`,
`gmsh-boundary-layer`, `predicates`, `triangulation`, `distribution`,
`forest-amr`, `omega-h`, and `omega-h-distributed` each record source and target
identities, the runtime identity (build, environment, backend, topology,
precision, and meshcore library), quality before and after, conservation and
transfer properties, resource use, and status; a scenario whose external
dependency is absent returns an explicit `missing-dependency` record instead of
passing. `examples/meshing_omega_h.py` exercises serial or MPI adaptation.

For `--scenario native-corpus`, `--corpus-profile` takes one or more exact
canonical profile names after one flag (it is not a repeatable append flag).
It executes only those selected routes; the report's
`qualification_dependencies` still retains every mandatory profile. Unrun
profiles remain not-run/unassessed/incomplete, not implicitly passed.
Selected contracts lacking their actual original archive/content/core inputs
are not executed with synthetic replacements: the ledger records
`missing_original_source_inputs` and `unqualified_selected_profiles`.
Unqualified selected contracts make the CLI exit with failure even if a
component sample reports passed. Source-only wall or sweep checks cannot
qualify a positive hybrid lifecycle.
Selection does not replace any original campaign's source, capacity, timeout,
target, degree, seed, adaptation or archive controls with an easier request.
Source status, scientific admission, full workflow, phase evidence and release
authorization remain separate gates. This selection/reporting interface is not
evidence that the current complete corpus pipeline has passed.

The local Homebrew MPI launcher required the explicit deployment setting
`HWLOC_SYNTHETIC='pack:1 core:10 pu:1'` and launcher arguments
`--bind-to none --map-by slot` to avoid an upstream hardware-topology startup
crash. This is not injected by any provider.

The QA extra installs SciPy and CAD typing packages. Native interfaces missing
from upstream declarations use narrow, runtime-checked protocols: the CAD edge
downcast and OpenVDB module, grid, accessor, and NumPy polygon outputs are
explicitly typed. OpenVDB remains lazily loaded. These boundaries require no
type ignores or unchecked casts and are exercised against the real libraries.

## Provider prerequisites and feasibility

VoroCrust source is available at
[sandialabs/vorocrust-meshing](https://github.com/sandialabs/vorocrust-meshing).
Its upstream build defaults to OpenMP, while the one-thread command above
disables it. Leave OpenMP enabled only when the upstream library and worker use
the same OpenMP-capable compiler. On Apple Silicon, the shown route avoids
assuming that Apple Clang supplies an OpenMP runtime.

Prime is an extraction feasibility decision, not an implemented provider.
The documented [Part API](https://prime.docs.pyansys.com/version/stable/api/_autosummary/ansys.meshing.prime.Part.html)
exposes zone/topology queries, not a direct cell-connectivity array route.
The [FileIO API](https://prime.docs.pyansys.com/version/stable/api/_autosummary/ansys.meshing.prime.FileIO.html)
can export Fluent meshes, LS-DYNA, and MAPDL CDB, so file-mediated extraction is
feasible in principle but requires format-specific semantic auditing.
[Prime Server](https://prime.docs.pyansys.com/version/stable/getting_started/index.html)
requires a licensed matching Ansys installation on supported Windows/Linux
systems; installing the Python client alone does not supply the server.
