# Single-device unstructured finite volume

Phydrax provides fixed-topology, cell-centered finite volume for:

- triangular, quadrilateral, and mixed triangle/quadrilateral meshes in two dimensions;
- affine tetrahedral meshes in three dimensions;
- scalar conservation laws and compressible Euler systems;
- piecewise-constant, general cell-polynomial, and unstructured WENO-Z reconstruction.

`CellComplexTopology` remains the incidence authority. Prepared finite-volume geometry
derives owner/neighbor routes and homogeneous face blocks without imposing explicit
connectivity on structured discretizations.

## Geometry

```python
import jax.numpy as jnp
import phydrax as phx

vertices = jnp.asarray([
    [0.0, 0.0],
    [0.5, 0.0],
    [1.0, 0.0],
    [0.0, 0.5],
    [0.5, 0.5],
    [1.0, 0.5],
    [0.0, 1.0],
    [0.5, 1.0],
    [1.0, 1.0],
])
quadrilaterals = jnp.asarray([
    [0, 1, 4, 3],
    [1, 2, 5, 4],
    [3, 4, 7, 6],
    [4, 5, 8, 7],
])

system = phx.equations.EulerSystem(2)
mesh_plan = phx.discretization.UnstructuredFiniteVolumePlan(
    vertices,
    quadrilaterals=quadrilaterals,
    component_names=system.component_names,
)
fv = mesh_plan.prepare()
```

Preparation normalizes cell orientation and validates:

- exact oriented incidence and boundary-of-boundary zero;
- edge or face manifoldness and opposing interior incidences;
- positive triangle area, bilinear quadrilateral Jacobians, or tetrahedron volume;
- owner-outward face vectors and per-cell vector closure;
- complete, non-overlapping physical boundary patches;
- stable vertex and cell global IDs.

Two-dimensional triangle and quadrilateral cells share one global line-face block.
Tetrahedra use triangular faces and three-point face quadrature. Cell-volume quadrature
is retained for mapped moments, polynomial reconstruction, and positivity points.

`topology_id` excludes coordinates. `geometry_id` binds coordinates to that topology.
`plan_id` binds both. This distinction allows fixed-connectivity geometry epochs without
pretending that a new connectivity is the same mesh.

For a mesh with an authoritative `CellGeometrySpec`, pass
`UnstructuredFiniteVolumePlan.from_cell_mesh(mesh, coordinate_spec=geometry)`.
An optional current certified `embedding=` reuses the owning geometry proof.
The prepared cell measures, physical centroids, face metrics and bounded-workset
quadrature then come from that map, not from its rounded vertex carriers.
Geometry identities distinguish these meanings even when the carrier coordinates
coincide. Integration errors and coordinate validity remain available on the
prepared discretization; quadrature exactness is advertised only when proved.
`measure_relative_tolerance` defaults to `1e-12` and drives rational cell-volume
integration without an absolute cutoff. Published `cell_measure_errors` include
rounding into the prepared array dtype; unrepresentable requested integration
accuracy is refused.

Face fluxes retain their existing face-averaged scalar convention. An integrated
face area vector is not a pointwise normal on a curved face. Carrier-only field
queries and vertex-only motion are refused for this geometry mode rather than
silently changing its meaning. Static stage evidence measures actual oriented
face closure; it does not assume a zero defect. Ordinary polyhedral stages retain
their producer's admitted closure bound in addition to dtype roundoff. This does
not relax mapped-geometry closure admission.

## Conservation dynamics

```python
boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
    fv.boundary_patch_names,
    {
        name: phx.discretization.ExtrapolationBoundary()
        for name in fv.boundary_patch_names
    },
)
method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
    phx.discretization.PiecewiseConstantReconstruction(),
    phx.discretization.HLLCFluxPlan(),
)
problem = phx.equations.ConservationProblemIR(
    "quadrilateral-euler",
    "state",
    system,
    boundaries,
)
compiled = phx.equations.compile_conservation_problem(problem, fv, method)
```

Every internal face flux is computed once, subtracted from its owner, and added to its
neighbor. Stationary meshes accept any `AbstractArbitraryNormalNumericalFluxPlan`,
including `EntropyStableEulerFluxPlan`; fluxes are evaluated on physical unit normals.
Moving and overset couplings additionally require
`AbstractArbitraryNormalALENumericalFluxPlan` (Rusanov, HLL, HLLC, Einfeldt-HLL,
`AllSpeedHLLFluxPlan`, and `ShockAwareAllSpeedFluxPlan`); preparation raises
`ValueError` for a stationary-only flux. Positivity fallback stays explicit through
`FluxPositivityPlan`, whose fallback flux obeys the same rule. Prepared dynamics
expose face fluxes, CFL limits, conservation diagnostics, JVP/VJP linearization,
shared SSPRK, and conservative positivity/retry.

`UnstructuredFiniteVolumeMethodPlan(..., closure=...)` accepts an
`AbstractFaceClosurePlan`. The closure corrects the baseline normal flux at every
centroid or quadrature site with a `FaceFluxContext` holding the physical unit
normal, face measure, geometry identity, and, on moving and overset stages, the exact
quadrature grid-normal velocity; inactive stage routes receive no correction. The
positivity fallback flux stays uncorrected.

## Cell-polynomial and WENO-Z reconstruction

`CellPolynomialReconstructionPlan(degree)` prepares a conservative zero-mean
total-degree basis. Cell moments come from physical cell quadrature. Characteristic
lengths normalize coordinates, deterministic breadth-first stencils provide cell
averages, and weighted column-scaled SVD produces fixed-capacity factors.

```python
polynomial = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(fv)
method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
    polynomial,
    phx.discretization.RusanovFluxPlan(),
)
```

Degree one is the general WLSQ path and is affine-exact on qualified polygonal and
tetrahedral meshes. Degree two is k-exact on qualified triangle and quadrilateral
geometries when the stencil rank policy passes. Reconstruction evaluates at face
quadrature rather than only at face centers.

`UnstructuredWENOZReconstructionPlan` combines an optimal polynomial with directional
sector candidates through a CWENO decomposition and Jiang--Shu derivative Gram
matrices. WENO-Z weights retain the exact constant and affine smooth limit. The optional
cell-extrema rescaling preserves cell averages while bounding all live face traces.

## Face traces and field views

A finite-volume field is a set of cell averages, so its value on a face depends on
the side and on the reconstruction. Face traces publish that choice explicitly.
`integration_domain("exterior_facet")` and `integration_domain("interior_facet")`
list faces in the face table's order; the owner is the cell the stored area vector
points out of. `prepare_side_trace` returns a `PreparedTraceAction` whose sites
follow the owner's facet parametrization (edges, or triangles for tetrahedra), whose
weights are the physical face measure, and whose normals point out of the side cell:

```python
rule = phx.discretization.FacetTraceRule(points=2)
interior = fv.integration_domain("interior_facet")
averages = fv.prepare_side_trace("state", interior, rule=rule)
polynomial = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(fv)
owner = fv.prepare_side_trace("state", interior, rule=rule, reconstruction=polynomial)
neighbor = fv.prepare_side_trace(
    "state", interior, rule=rule, side="neighbor", reconstruction=polynomial
)
cell_averages = jnp.ones(fv.state_shape)
jump = owner.apply(cell_averages) - neighbor.apply(cell_averages)
```

Without a reconstruction the trace repeats the side cell average at every site
(`representation="cell-average"`, `trace_degree=0`). A cell polynomial, triangle
k-exact, or unlimited triangle MUSCL reconstruction publishes the reconstructed face
state (`representation="face-state"`) through a route that gathers only the side
cell and its stencil; `dual_pullback` and `inject_load` are its exact transpose.
Owner and neighbor traces of an interior face share sites and have opposite normals.

WENO-Z and limited MUSCL face states are nonlinear in the cell averages.
`prepare_nonlinear_face_trace` returns a `PreparedNonlinearFaceTrace` with `apply`
and `linearize(state)`; its `jvp`/`vjp` are the exact derivative at that state, and
no global transpose exists. `prepare_side_trace` refuses these reconstructions, and
`side="average"`, neighbor sides of boundary faces, foreign domains, normal or flux
quantities, and polyhedral faces are refused as well. The WENO-Z cell-extrema limiter
scales each face by its own site set, exactly as the conservation dynamics do at face
quadrature.

`prepare_finite_volume_field_reconstruction` evaluates the same reconstructions at
arbitrary points on structured, unstructured, and triangular meshes. Triangle k-exact
and WLSQ MUSCL planes publish exact coordinate derivatives; limited MUSCL queries are
nonlinear and expose `linearize` instead of `transpose`.

## Implicit compressible stepping

`FiniteVolumeBackwardEulerPlan` wraps any prepared structured, triangle, or
unstructured finite-volume dynamics in the canonical matrix-free nonlinear stack. Its
residual is the physical backward-Euler residual. `prepare()` binds a reusable symbolic
Newton/Krylov plan; `refresh()` updates time-step numerics without changing that
symbolic identity. Failed nonlinear solves are fail-closed: the accepted state and time
do not advance.

The finite-volume precision policy remains authoritative for state, flux, reduction,
and output dtypes. Nonlinear precision is derived from it unless supplied explicitly.

Global conservation diagnostics and conservative transfer helpers use a twofold
compensated reduction after the finite-volume precision policy has cast contributions
to `reduction_dtype`. Remap, overset/sliding, accepted-ledger, multiblock, and
small-cell defects accumulate signed final contributions together so separately
rounded totals cannot create a false exact zero.

## Collocated incompressible pressure correction

`PreparedUnstructuredCollocatedOperators` supplies:

- volume-weighted pressure gauge projection;
- cell WLSQ and two-point face-normal pressure gradients;
- owner-oriented divergence;
- momentum-diagonal-aware Rhie--Chow face interpolation;
- a positive gauged weighted pressure operator.

`UnstructuredPressureProjectionPlan` refreshes the matrix-free Schur operator for the
current, potentially nonuniform inverse momentum diagonal. The same face coefficient
drives the Rhie--Chow prediction, pressure solve, face correction, and cell-velocity
correction. `UnstructuredPressureCorrectionPlan` performs a fixed JAX correction loop
and retains every linear status and divergence norm.

## Mesh archive, case, checkpoint, and output

`write_unstructured_fv_archive()` stores connectivity, coordinates, patches, and stable
global IDs in the canonical checksum-validated array archive.
`read_unstructured_fv_archive()` reconstructs and revalidates every identity.

Portable unstructured case documents use schema version 2. They reference a native mesh
archive by path and SHA-256, select an allowlisted equation/method/boundary set, and
provide an explicit constant primitive or conservative initial state.

Finite-volume checkpoints use archive schema version 5, runtime-state schema version 4,
and content schema version 2. Schema 5 binds the case, current precision evidence,
topology/geometry/family identities, conservative content, effective volumes and active
mask, geometry/evidence versions, complete topology epoch journal, controller/
integrator/forcing/random/output state, coordinates, and stable IDs. A sliding state
additionally stores its normalized shift, accepted event ID, coupling/evidence IDs, and
every overlap route and measure array. Writing a sliding state requires the originating
prepared runtime; reading rebuilds the coupling from its prepared sliding plan and
rejects any non-identical route, measure, or identity.

The reader dispatches on the manifest version before applying that version's inventory
rules. Restart support is exact:

| Checkpoint schema | Accepted archive | In-memory migration | Executable restart requirements |
| --- | --- | --- | --- |
| 2 | Legacy structured ZIP/NumPy writer with its exact checksums, manifest identity, case, and precision evidence | Cell averages become conservative content using the supplied prepared runtime's effective volumes; current precision, geometry/evidence version zero, canonical initial epoch, and an empty initial journal are constructed | A compatible fixed-geometry structured prepared runtime must be supplied to `FiniteVolumeCheckpointPlan`; no ALE, sliding, or accepted topology event can be recovered |
| 3 | Canonical array archive with exact structured or unstructured mesh identity and stable-ID payload | The same content/current-precision/initial-epoch/empty-journal migration as schema 2 | A compatible fixed-geometry prepared runtime must be supplied; the schema contains no ALE version, sliding artifact, or accepted topology journal |
| 4 | Transitional unreleased archive | Rejected cleanly; its incomplete family/sliding state cannot be reconstructed safely | Rewrite from the originating worktree before upgrade |
| 5 | Current canonical content-authoritative archive | No lossy migration; every stored runtime, content-family, sliding, and journal leaf is reconstructed and identity-checked | Static restarts use the compatible prepared runtime. Sliding read/write also requires the originating prepared runtime so the accepted coupling can be rebuilt exactly; a committed topology successor is executable only with its matching prepared successor runtime |

Unknown versions, incompatible cases or precision, changed mesh/global IDs, incomplete
inventories, checksums, payload identities, or subsystem identities fail closed.

`FiniteVolumeOutputPlan` writes structured or unstructured HDF5/XDMF time series.
Unstructured XDMF contains one spatial child grid per live cell block and one scalar
cell attribute per conservative component. Every unstructured step references points
stored under its HDF5 geometry epoch and geometry version. Geometry version zero may use
the prepared points implicitly; a moved version requires an explicit accepted
`UnstructuredFiniteVolumeGeometryState` or accepted vertex array. A missing state, or a
state whose time, version, topology, or geometry-layout identity is stale, is rejected.
`write_vtk_snapshot()` uses the same accepted-points contract and emits a meshio VTK
sidecar with global IDs; VTK is not a restart format.

## Fixed-connectivity motion and topology events

`FixedConnectivityMotionPlan` evaluates immutable geometry epochs from one topology.
It reports vertex velocity, face-normal grid velocity, swept face volume, cell-volume
change, and the discrete geometric-conservation-law defect. Geometry evaluation and
topology events are intentionally host-side and occur between accepted steps.

Without a `motion_policy` (or with the `PRESCRIBED` route) the motion callable maps
every vertex. Any other `FiniteElementMeshMotionRoute` makes the callable prescribe only
the boundary vertices; the interior follows the shared finite-element motion route
(`monitor(time, points, args)` drives `MMPDE`). Trial epochs are accepted through the
shared `MotionValidityPlan` corner-Jacobian check.

The discrete GCL holds for any vertex trajectory, including rotations and routed
positions from an iterative solve. Each SSPRK(3,3) stage takes its face volume rates
from exact swept volumes between the realized positions at `t`, `t + dt` and
`t + dt / 2` (straight vertex paths; midpoint quadrature in 2-D, two-point Gauss in
3-D). The rates are combined so that the stage volume recurrence reproduces the
coordinate volumes of those positions at rounding: `v1 = V(t + dt)`,
`v2 = V(t + dt / 2)` and `vnew = V(t + dt)` when `vn = V(t)`. The flux grid-normal
velocity is the time-JVP vertex velocity, shifted uniformly on each face so that its
quadrature equals that swept rate. The JVP velocity itself stays in the stage face
block (`quadrature_kinematic_grid_normal_velocity`) as the kinematic witness that a
`MovingSlipWallBoundary` follows its wall-velocity provider. The wall reflects about
the flux (swept) velocity, so no volume or mass crosses the discrete wall.

`VariablePatchGeometryPlan` takes the same `motion_policy`: its coordinate map then
prescribes the level-zero grid boundary and every patch vertex interpolates the routed
level-zero Q1 grid.

`UnstructuredConservativeRemapPlan` is an explicit CSR common-refinement artifact. It
validates complete source/target coverage and applies conservative cell-integral
transfer. A remesh is therefore a new topology and geometry plus an auditable remap,
never an in-place mutation.

`PreparedFiniteVolumeRuntime.commit_remesh(accepted_result, transition,
successor_dynamics, content_semantics=...)` commits an accepted remesh or adaptation
transition at the step boundary, and accepts two transfer semantics:

- `"eulerian_physical_overlap"` transfers content by the exact physical overlap of
  the endpoint geometries.
- `"material_following_content"` moves extensive content along the transition's
  accepted volume material chain; it is refused when the transition has no chain.

The topology transaction prepares the transfer, checks coverage and conservation,
and publishes the successor runtime only on commit.

## Owner-local conservative transfers

`prepare_owner_local_conservative_remap(source_view, target_view, execution_group)`
builds an owner-routed physical overlap map from accepted
`OwnerLocalCellGeometry` views. Only candidate source-cell maps, requested source
values, and target-owned overlap rows are communicated; fields and complete meshes
are not gathered. The returned plan refreshes target ghosts from their actual
owners.

`UnstructuredFiniteVolumePlan.from_owner_local_geometry(view, ...)` prepares only
the compact owned-plus-required-ghost cells already present in an
`OwnerLocalCellGeometry`; it does not gather a root mesh. Physical-boundary faces
and artificial owner-local halo faces have separate boundary patches. During
SSPRK execution the runtime refreshes ghost state from each cell's owner, while
`FiniteVolumeConservativeContentState.owned_cell_mask` keeps ghost copies out of
integrals, source totals, and CFL decisions. A rank with no owned cells uses a
genuine zero-cell plan and state, with no placeholder physical cell, and still
participates in the same stage collectives.

A periodic view's resident patch is a lifted piece of the quotient mesh. A cell's
periodic neighbour is often resident at another lattice image, so their shared
face appears twice and is not welded. The lowering joins these seam faces by
quotient identity (vertex roots and their windings relative to an anchor, never
coordinates) into interior faces. A periodic part therefore has no physical
boundary faces: `__physical_boundary__` is empty, and only artificial halo cuts
stay open. `periodic_face_translations` records, per FV face, the translation
from the owner's frame to the neighbour's lifted frame (zero off the seams).
`face_topology_ids` maps FV faces back to the lifted connectivity, and
`neighbor_face_topology_ids` names the unwelded twin face the neighbour holds.
Every operator works in the frame of the cell it evaluates:

- Cell-polynomial and WENO stencils are built from cell images. A stencil
  member is a `(cell, lattice image)` pair, and `stencil_images` translates
  each member beside its centre cell. A periodic cell reached at two images is
  two members, exactly as in the periodic tiling. Within each ring, members
  are ordered by image (the identity first), then by cell index.
  Neighbour-side traces, both static (`reconstruct_at`) and stage-evaluated
  (moving reconstruction), are taken at the face points translated into the
  neighbour's frame (`neighbor_side_points`).
- Unstructured viscous gradients and the viscous stability bound take a seam
  neighbour's centre at its lattice image beside the owner.
- Side traces publish shared facet sites in the owner's frame. A seam
  neighbour's trace reads its twin facet, oriented by the twin's vertex
  parity, and evaluates at those sites translated into its own frame.
- Continuous ALE (`FixedConnectivityMotionPlan`) evaluates the geometry on
  every lifted face and reduces it to the joined faces. The prescribed motion
  must be translation equivariant: each seam twin vertex stays its partner's
  lattice translate and moves with the same velocity. Otherwise the stage
  geometry is invalid, and a host epoch
  (`UnstructuredFiniteVolumePlan.with_periodic_seam_vertices`) refuses it.
  Routed interior motion (`motion_policy` with a non-`PRESCRIBED` route) is
  solved on the periodic quotient: the FE extension
  (`FiniteElementMotionExtension(periodic_vertex_pairs=...)`) gives each seam
  twin class one displacement unknown, keeps the lifted cell geometry in its
  element tensors, and counts lattice copies of a seam edge once. Moved twins
  stay exact lattice copies, so a moving or rotating obstacle inside a periodic
  box drives the whole quotient mesh, and the swept-volume stage rates keep the
  ALE GCL at rounding.

Each operator is regression-tested against the same serial operator on a
5×5 tiling of the single-owner periodic patch, comparing the central rows.

```python
plan = phx.discretization.finite_volume.prepare_owner_local_conservative_remap(
    source_view,
    target_view,
    execution_group,
)
target_averages = plan.apply_averages(
    source_averages, source_view, target_view, execution_group
)
target_contents = plan.apply_content(
    source_contents, source_view, target_view, execution_group
)
```

`apply_averages` uses physical overlap divided by target-cell measure and returns
cell averages. `apply_content` uses physical overlap divided by source-cell measure
and returns extensive cell contents. `transpose` is the raw algebraic transpose of
the average map; `weighted_adjoint` is the distinct adjoint under source/target
cell-volume pairings.

The plan retains cell measures, overlap moments, coverage defects and tolerances,
owner routes, accepted global source/part composition IDs, and rank-local carrier
IDs. An optional `CellGeometryTransition` binds its endpoints to the canonical
rank-local meshes and coordinate maps; `advance_id` is common to the owner group,
while transition identities remain rank-local. Straight-sided triangles,
quadrilaterals, polygons, tetrahedra, and authored polyhedra use the native
common-refinement path. Nonperiodic curved 3D tetrahedra require complete exact
transition reference/refinement/coarsening maps to a shared root; overlap measures
and moments then use the certified curved root Jacobian, retaining integration-error
and reference-coverage evidence. Curved maps are never replaced by corner geometry.

Incomplete or double coverage remains diagnostic evidence and is not transportable.
`PreparedOwnerLocalConservativeRemap.from_stored_fields` rebuilds the current
accepted views, directory routes, and physical overlap evidence, then rejects any
replay whose recomputed remap identity differs from the stored artifact.

`OwnerLocalCompositeCellGeometry((donor_view, receptor_view), remote_ghosts=...)`
compiles several owner-local part views into one disjoint local layout. Each part
keeps its own rows and ownership. `OwnerLocalRemoteGhostCells` are donor cells
fetched from their owners and placed in this rank's lattice image; their state is
refreshed by the same per-stage owner exchange. The owner-local overset coupling
works as follows:

- `prepare_owner_local_finite_volume_overset(connectivity, views, donor, receptor,
  execution_group, component_names=...)` builds the composite, its FV geometry, and
  a conservative `UnstructuredOversetPlan` over composite rows.
- Each rank's owned fringe receptor cells must be covered by live (non-hole) donor
  cells, summed over every periodic lattice image.
- A coverage gap expands the donor halo collectively. It first requests the native
  winners of the fringe points, then the face neighbours of donor cells touching
  uncovered receptors.
- The coupling refuses only when no owner can add a cell. Every refusal names each
  failing owner's reason.
- A route lives on its fringe receptor's owner. When its donor row there is a
  ghost, every SSPRK stage returns that donor debit to the donor's owner, so the
  global ledger balance equals the owned content change.
- Every certified receptor face artifact includes `receptor_face_orientations`:
  `+1` identifies the physical face's owner incidence and `-1` its neighbor
  incidence. This is required even when both incidences have the same periodic
  cell ID. Quadrature points and normals use the selected receptor image and
  outward orientation; orientation is part of the artifact identity.
- Owner-local preparation selects a face with positive outward first-coordinate
  area using exact source-vertex coordinates and a stable quotient face key.
  Opposite incidences cannot both select that face, so selection is independent
  of the rank-local routed-receptor subset. This requires closed,
  full-dimensional receptor cells with exact periodic source vertices.

`commit_owner_local_repartition(..., owner_local_part_name=...)` migrates one named
part of a composite runtime through the native graph migration. The owned cells of
the other parts stay on their owners and keep their exact content.
Explicitly declared cell-state banks may include typed JAX PRNG keys with the
`rng` role. Their implementation and key words follow each stable cell ID; the
repartition does not itself draw random values or advance the key.

Owner-local archive replay rebuilds its source and target contexts through
`phydrax.solver._finite_volume_archive_context`. A recipe naming a source must be
replayed with an explicit `sources` mapping from that exact `source_id` to the
current callable; unknown identifiers are rejected. The callable is bound to both
source and target dynamics, and its state-dependent inputs remain the caller's
responsibility. The archive does not serialize Python callables.
Receptor boundary and wall scopes follow lifted global facet-vertex identities
across repartition; excluded cells follow global cell IDs. The rebinding keeps
the declared priority, owner rank, solid index and image transform, and refuses
scoped source entities missing from the target. Cold archives reject non-None
solid queries. Live rebinding retains a query on its original owner, but rejects
the result collectively if that owner has no resident wall: the current part
specification cannot represent a query without a wall or an empty wall scope.

`DrivenEulerSource` is a cellwise acceleration and decaying heat source with
explicit `DrivenEulerSourceArgs`: material coefficients, driving history, typed
cell keys, committed clocks and an active-row mask. Construct keys with
`driven_euler_cell_keys` from both 32-bit words of each persistent global ID.
Keep the banks in full dynamics-cell-row order and pass
`banks.select_rows(runtime.dynamics.source_cell_indices)` as the source
arguments. This gathers every field, including typed keys, in the order used
by the native unstructured evaluator. Staged evaluations are pure; accepted
updates receive and retain the full bank layout.
`PreparedFiniteVolumeRuntime.advance_with_auxiliary` composes the ordinary FV
advance with an accepted-only update, such as `advance_driven_euler_banks`.
On rejection it returns the original banks with the native failure evidence.
Partitioned callers must migrate the banks by stable cell ID and rebuild the
row-aligned arguments for the successor runtime.

`commit_moving_overset_update(..., owner_local_part_name=...)` likewise moves one
named part of a composite runtime. The rebuilt successor coupling binds the moved
part's successor view and the unchanged views of the other parts. The moved part's
rows take the owner-routed conservative remap, and rows of the other parts keep
their content by stable cell ID. Remote donor ghost rows follow the moved overlap and
are refreshed from their owners before the successor state is admitted. An Eulerian
commit constrains only the moved part's own physical boundary; a native periodic
part closes its periodic cell and has none. Accepted Composition entries other than
the overset registration and the FV state need the caller's explicit dispositions
(`retained_state_entry_ids`, `reprepare_entries`, `invalidate_entry_ids`), as in
`commit_owner_local_repartition`.


## Embedded boundaries and VOF

`EmbeddedBoundaryPlan` performs exact linear-edge clipping of a level set on
two-dimensional polygonal cells. It produces fluid volumes, face apertures, cut-face
centers/normals/measures, body tags, safe masked inverses, and fluid closure evidence.

`UnstructuredVOFPlan` provides bounded conservative upwind phase transport and
host-prepared two-dimensional PLIC segments. PLIC offsets are solved against the exact
clipped polygon area. The reported stable step is the monotone phase-flux restriction;
updates are rejected rather than clipped if they would violate bounds.

## Single-device AMR, overset, and sliding maps

`UnstructuredAMRHierarchyPlan` binds a fixed two-level nested hierarchy through
conservative overlap maps. It provides deterministic fixed-capacity tagging,
prolongation, volume restriction, temporal coarse/fine ghost filling, composite
subcycling, and explicit accepted-step interface reflux. Reflux requires certified
per-level interface route IDs; whole-cell ledger correction is rejected.
After an accepted composite step, a changed selection conservatively restricts departing
fine ownership and prolongs newly activated fine cells, certifies the old/new composite
integral, commits geometry-matching coarse and fine successor epochs atomically, and
returns a reprepared successor AMR runtime.

`UnstructuredOversetPlan` owns immutable donor/receptor overlap maps with active, hole,
fringe, donor-eligibility, global-ID, coverage, union-volume, and epoch evidence.
Conservative runtime correction requires a separate certified receptor-face artifact:
face quadrature points, unit normals, positive face measures, and receptor ownership.
Volume-overlap CSR entries are never treated as synthetic faces. The static
`receptor_route_faces` map assigns every overlap route to one certified face of its
receptor cell, and several routes may share a face. That face's exterior trace is
the overlap-weighted mean of its routes' donor traces. The route fractions of each
face sum to one exactly; this is checked at preparation. Each route carries its
fraction of the face flux, from its donor to its receptor. Sliding couplings use
the same correction, with the interval overlap routes as the route-to-face map.

This is a physical cell-overlap remap: each CSR measure is a donor/receptor
intersection measure, and the map requires complete coverage evidence before its
conservative content balance is used. `remap_to_receptors` returns overlap-weighted
cell averages; this API has no interpolation or bounded-clipping mode. Native assembly
overset connectivity separately classifies holes/fringes and prepares point-field
queries through `PreparedFieldQuery`. Those queries preserve the selected FE/FV
reconstruction, its action and exact transpose, but make no conservation claim. Use
the common-refinement remap route when conservative transfer is required.

At an accepted finite-volume step, `PreparedFiniteVolumeRuntime.commit_moving_overset_update`
can rebind a serial native overset mesh. The supplied `MeshMotionAdvance` values are
checked against `OversetConnectivity.moved_from_advances`; the successor FV map must
bind that exact connectivity epoch. An Eulerian step permits only a fixed-time rezone
with unchanged physical boundary coordinates, and transfers the accepted post-step
content through native physical overlap. A continuously moving boundary instead requires
accepted ALE geometry whose endpoint exactly matches the native coordinate transition,
plus successor dynamics with an ALE plan that starts at that endpoint. The ALE transfer
is formed from the realized endpoint geometry, so it does not replace or reset the
accepted SSPRK/GCL update. In either mode, the topology event and
`prepare_overset_motion_rebind` publish together; a failed candidate leaves the accepted
predecessor state and its runtime executable. The result then carries no
`successor_runtime` and names the refusal in `topology_event_refusal`
(`FiniteVolumeTopologyEventRefusal`: failed event status, reason, request and evidence
IDs). `commit_remesh` reports its refusals the same way.

`PeriodicSlidingInterfacePlan` rebuilds a periodic one-dimensional overlap at accepted
step boundaries and transfers interface flux with equal-and-opposite integrated budget.
Refreshes are explicit topology transactions; a failed map leaves the predecessor epoch
as the only valid runtime state.

## Embedded boundaries and capillarity

`EmbeddedBoundaryPlan` clips a 2-D polygonal level set into fluid volumes, open-face
apertures, cut-face routes, body tags, closure evidence, and small-cell policy data.
`UnstructuredEmbeddedBoundarySet` requires explicit stationary wall ownership. The
runtime compacts inactive routes before EOS/Riemann evaluation, excludes solid cells
from positivity and CFL, and redistributes small-cell rate excess only to stable
non-small recipients.

`BalancedCapillaryOperator` and `SurfaceTensionPolicy` provide curvature evidence, a
balanced capillary force block, and an explicit capillary step restriction. The
curvature is `kappa = t . dn/ds` of the 2-D PLIC normals: `+1/R` for a drop of the
alpha phase and `-1/R` for a bubble. Each interior face carries the force
`phi_f (alpha_Q - alpha_P) A_f n_f` with `phi_f = sigma * kappa_f`, and each of its two
cells receives half. The cell force approximates `phi grad(alpha)` per unit volume.
It uses the same face operator as the arithmetic-mean Riemann pressure, so for uniform
curvature it cancels the Laplace pressure `p0 + sigma * kappa * alpha` to rounding.
It is a body force, not a flux: the net force is the discrete `oint sigma kappa n dS`
(zero for uniform curvature). The capillary work on each cell is its force times the
cell velocity. `face_rate_block` refuses interface cells without valid curvature and
alpha jumps without adjacent valid curvature, unless `sigma = 0` (exact disabling path).
The force uses the base geometry: embedded cut apertures and moving-mesh stage metrics
are not balanced. Curvature from the differentiated first-order PLIC normals of a real
reconstruction is noisy and does not converge under refinement; relax
`curvature_tolerance` only with that limit in mind. Contact angles require an
`EmbeddedBoundaryContactAngleSet`; no angle is inferred from a wall type.

## Two-material VOF status

`TwoMaterialEOSClosure` supports ideal-gas and stiffened-gas affine caloric materials.
`TwoMaterialVOFSystem` exposes the explicit d+4 layout
`[m0, m1, momentum[d], total_energy, alpha0]`, conservative extensive fluxes,
admissibility, pressure, and signal bounds. Ordinary Riemann fluxes deliberately set
the alpha component to zero: alpha transport must use `UnstructuredVOFPlan` PLIC
apertures and its dedicated `phase_transport_flux`/`phase_swept_flux` contracts.

The compiler requires `TwoMaterialVOFSystem`, piecewise-constant cell states, and a
prepared `UnstructuredVOFPlan`. Every SSPRK stage rebuilds fixed-capacity JAX PLIC
geometry from that stage alpha, derives phase-consistent mass/alpha fluxes from one
donor decision, and adds the Kapila dilatation source. Optional capillarity and explicit
embedded-wall contact angles use the same stage PLIC evidence; stale or uncertain
geometry fails closed.

### Cavitation and interfacial phase transfer

`BarotropicEulerSystem` with
`HomogeneousEquilibriumCavitationMaterial` represents instantaneous
homogeneous equilibrium without an independently transported interface. The
linear and Wallis mixture branches expose pressure, sound speed, vapour
fraction, barotropic energy, branch margins, and hyperbolicity evidence.

Kinetic and thermal VOF transfer instead use
`TwoMaterialVOFPhaseChangePlan`. Merkle, Kunz, and Schnerr--Sauer laws provide
pressure-driven transfer; interface heat resistance, temperature relaxation,
and Stefan heat-flux laws provide thermal transfer. Positive rate always means
phase 0 to phase 1. Both phase mass rates, the volume rate, alpha rate, latent
power, donor restriction, and defects derive from that one signed rate.

`VOFPhaseChangePlan` binds a transfer law to the exact stage PLIC geometry.
Interface-resistance and Stefan rates use reconstructed interface measure per
cell. Stefan transfer additionally reconstructs both phase-side normal
temperature gradients from fixed neighbor routes and fails when an active
interface lacks evidence on either side.
`UnstructuredTwoMaterialThermalDiffusionPlan` adds conservative
heat conduction to total energy, static adiabatic/prescribed-temperature/
outward-heat-flux patch closures, and an explicit thermal restriction; latent
heat is not added twice because material reference energies remain in the
conserved total-energy closure.
For stiff local transfer, `FiniteVolumePhaseChangeStrangMethod` applies two
bounded half-source updates around one exact transport attempt. It accepts only
an unretried transport step of the requested size and otherwise restores every
runtime leaf atomically.


Moved PLIC reconstructs polygons from `FiniteVolumeStageMetrics.vertices` and
recomputes its least-squares interface normal from stage cell centers. Static
and certified moved/sliding overset corrections evaluate the donor PLIC
half-plane at receptor quadrature
points and use that same aperture for both partial-mass and alpha fluxes.
Remap continues to transfer the complete extensive VOF state. Phase
appearance, donor exhaustion, PLIC branch changes, and overlap-graph changes
remain declared discrete derivative events.

## Automatic common refinement and second-order remap

By default,
`prepare_unstructured_conservative_remap(source, target, provenance=..., policy=...)`
intersects the two declared carrier meshes with
`phydrax.geometry.prepare_common_refinement` and native exact predicates/clipping.
`CommonRefinementPolicy` sets coverage mode, relative coverage tolerance, and
candidate/accepted/memory limits. The returned `PreparedUnstructuredConservativeRemap`
carries the `PreparedCommonRefinement`, its `CommonRefinementStatus` and evidence, a
`reason`, and `plan` (an `UnstructuredConservativeRemapPlan`) only on `SUCCESS`.
Geometric failures, coverage gaps, double coverage, and resource refusal are returned
as statuses without a plan; a certified refinement that does not cover the
finite-volume cell volumes within tolerance reports `COVERAGE_GAP` or
`DOUBLE_COVERAGE` rather than raising from the plan constructor.

```python
remap = phx.discretization.prepare_unstructured_conservative_remap(
    source, target, provenance="remesh"
)
if not remap.succeeded:
    raise RuntimeError(remap.reason)
first_order = remap.plan.apply(averages)
second = phx.discretization.UnstructuredSecondOrderRemapPlan(
    remap.plan, remap.refinement, source
)
result = second.apply(averages)  # values, limiter and conservation evidence
```

### Accepted nested coordinate geometry

An accepted exact `CellGeometryTransition` supplies a different physical overlap
route. Both FV endpoints must retain their actual coordinate specs:

```python
source = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
    transition.source_mesh,
    coordinate_spec=transition.source_geometry,
    embedding=source_embedding,
).prepare()
target = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
    transition.target_mesh,
    coordinate_spec=transition.target_geometry,
    embedding=target_embedding,
).prepare()
remap = phx.discretization.prepare_unstructured_conservative_remap(
    source,
    target,
    provenance="accepted-material-transition",
    geometry_transition=transition,
    source_embedding=source_embedding,
    target_embedding=target_embedding,
)
```

Here `source_embedding` and `target_embedding` are optional current certificates,
not identifiers substituted for proofs. The owner checks exact map restrictions,
reference-cell inclusion and disjoint coverage, endpoint currentness, and physical
embedding before integrating the declared geometry. An ancestry table alone is
not a conservation proof. Forward refinement and inverse/coarsening use the same
physical contract, including the requested partial-coverage semantics.

The resulting `refinement.geometry_evidence` retains the endpoint certificates,
integration errors, partition findings and geometry/transition identities.
Material labels are not used to discard overlaps; positive carrier crossings are
not rounded to zero. FV coverage still has to satisfy the original tolerance.
The nested route budgets rational measure error at
`min(1e-12, policy.coverage_tolerance / 16)`. Prepare both FV endpoints with
`measure_relative_tolerance` at least this accurate when requesting tighter
coverage. Remap admission checks their actual published error arrays, not merely
the requested preparation setting; insufficient precision returns
`PREDICATE_UNCERTAIN` without a plan. Neither volumes nor weights are normalized.
Requests for physical overlap simplices are refused when this mapped route cannot
represent them faithfully; no weighted or collapsed carrier is presented as
geometric output.


`UnstructuredSecondOrderRemapPlan` reconstructs `u_s + phi_s g_s . (x - c_s)` about
each source cell's declared-geometry centroid `c_s`. The weighted least-squares
gradient uses all cells sharing a vertex with `s`, so boundary and corner cells keep a
full-rank stencil; its normal equations are prepared once with batched
`phydrax.linalg` small solves (rank and condition evidence is retained; rank-deficient
or ill-conditioned stencils raise) and applied as a `phydrax.sparse.SparseLinearMap`.
The reconstruction is integrated exactly with the overlap first moments,
`I_st = V_st u_s + phi_s g_s . (m_st - c_s V_st)`, so linear fields are remapped
exactly and every source cell keeps its content for any `phi_s`.
`UnstructuredRemapLimiter.BARTH_JESPERSEN` picks the largest `phi_s` keeping every
overlap average within the extrema of the source stencil, hence every target average
within its neighborhood bounds. The global residual left by coverage tolerance and
rounding is redistributed into target cells in proportion to their bounded slack
(target volume without a limiter); the result reports residual before and after,
redistributed content, whether the slack sufficed, and limiter activation count,
fraction, and minimum factor. `apply` is pure JAX and differentiates with respect to
the field values and the stored overlap/centroid geometry for fixed combinatorics.

For a frozen route graph, the first-order `apply_fixed_combinatorics` accepts dynamic
intersection measures and source/target volumes. JVP/VJP therefore cover smooth
coordinate changes that preserve every intersection route. A changed intersection
graph, remesh choice, or failed coverage certificate is a discrete event and has no
fabricated ordinary gradient.

`FiniteVolumeStageEpochTransition` is the FV-owned physical payload for DCD segmented
execution after SSPRK stages 1 or 2. `PreparedUnstructuredSSPRK3Runtime` transfers
every live Shu--Osher step-start/stage register and accepted-ledger accumulator,
switches to the prepared successor dynamics, and rolls back the original content
atomically when a stage or transfer fails. Scheduling, bounded event tapes, replay,
and saltation remain owned by the solver hybrid-event APIs.

## Polyhedral geometry and moving reconstruction

`prepare_polyhedral_finite_volume_geometry(CellMesh.from_polyhedra(...))` consumes the
canonical root `PolyhedralConnectivity`. Planar outward face loops are certified with
Newell area vectors, divergence-theorem volume/centroid identities, positive
star-tetrahedron quadrature, manifold owner/neighbor routes, and per-cell closure
residuals. A face farther than `planarity_tolerance` (relative to its diameter) from
its Newell plane is warped, for example the bilinear prism sides of boundary-layer
columns that diverge at a sharp edge. It is split into triangles about its vertex
centroid, each of which must have positive area along the face normal. The face area
vector is the sum of the triangle area vectors, and both adjacent cells integrate
volume, centroid and quadrature over the same triangles. The shared triangulated
surface therefore keeps per-cell closure and global conservation exact. Planar faces
keep their first-vertex fan unchanged. Padded face/cell quadrature capacities stay
fixed per mesh: n - 2 fan slots for a planar n-gon and n slots for a warped n-gon.
`UnstructuredFiniteVolumePlan.from_cell_mesh(mesh)` wires this certified geometry into
the ordinary owner/neighbor face block, spaces, quality report, reconstruction, and
dynamics path without a private polyhedral topology.

This default declares the polyhedral carriers themselves as the integration
geometry. It is distinct from passing an authoritative `coordinate_spec`; an
accepted source-supported refinement must not substitute the former for the latter.



Moving fixed-connectivity execution admits stage-refreshed degree-one WLSQ.
Degree-two polynomial and CWENO-Z reconstruction are additionally certified for rigid
translations, where normalized moments and smoothness grams remain invariant; other
degree-two motion fails explicitly. The stencil graph stays fixed and rank/condition
failure rejects the stage.

`UnstructuredLowMachLESPlan` is the conservative constitutive route: fixed
conforming 3-D tetrahedra, Favre transport, optional static KSGS,
piecewise-constant upwind flux, Rhie–Chow mass flux, deferred nonorthogonal
gradients, and closed boundaries. `UnstructuredLowMachLESFixedStepMethod` adds
the gauged matrix-free pressure projection, exact fixed-step predictor/correction,
complete pressure/face-flux/restart continuation, advection/diffusion/source/
positivity bounds, conservation/energy/pressure evidence, and atomic rollback.
It still refuses 2-D/polyhedral, periodic/open, moving/coupled,
dynamic/low-Re KSGS, and nonzero molecular bulk viscosity. See the
[LES guide](guides_large_eddy_simulation.md#unstructured-low-mach-favre-les).

Accepted continuation stores the corrected pressure, pressure increment,
pressure-corrected face-normal velocity, and the authoritative mass flux recomputed
from that corrected rate; predictor flux is never committed.

Static-KSGS raw production uses negative shared SGS face work, equally split
between adjacent cells and volume normalized; negative raw production fails.
Limiter-retained KSGS gain plus rejected production equals raw transfer, with
the rejected amount thermalized into modeled enthalpy density. The step gates
that split and total enthalpy-inclusive energy balance.

Phase-change transfer is provided only through a geometry-bound
`VOFPhaseChangePlan`; there is no detached Stefan source kernel.
`VariableSurfaceTensionPolicy` evaluates nonnegative sigma and the wall/interface
tangential Marangoni gradient without duck-typed field extraction.


`FiniteVolumeTopologyEventScheduler` coalesces remesh, AMR, and overset requests at
accepted-step boundaries. Transactions require typed candidate epoch, remap coverage,
metrics/evidence status, and conservation evidence; absent evidence never defaults to
success. Given `source_geometry` and `target_geometry` without a remap, a transaction
prepares the automatic remap under `remap_policy`; the result's `automatic_remap`
exposes its status and evidence, and a failed refinement fails the event with
`FAILED_RESOURCE_LIMIT` or `FAILED_COVERAGE`. A committed event advances the epoch
journal atomically.

## Scope

The general unstructured runtime remains single-device: graph partitioning and
distributed FV halos are not claimed. The LES specialization is narrower still:
fixed conforming tetrahedra and fixed closed boundaries only. Bounded planar-faced
polyhedra mean finite orientable manifold convex or star-shaped cells within
declared face/vertex/quadrature capacities, not arbitrary nonmanifold or curved
cells. Moving degree one supports general fixed-connectivity metrics; moving degree
two/WENO is restricted to certified rigid translation. Remap derivatives are valid
only for fixed intersection combinatorics. Topology selection is differentiated
only through the DCD event/replay contract; the upwind low-Mach LES route is
branchwise on a fixed mesh.
