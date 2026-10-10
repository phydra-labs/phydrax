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
(`monitor(time, points, args)` drives `MMPDE`). Vertex velocity is the exact time JVP of
the realized vertices either way, so the discrete GCL holds by construction, and trial
epochs are accepted through the shared `MotionValidityPlan` corner-Jacobian check.
`VariablePatchGeometryPlan` takes the same `motion_policy`: its coordinate map then
prescribes the level-zero grid boundary and every patch vertex interpolates the routed
level-zero Q1 grid.

`UnstructuredConservativeRemapPlan` is an explicit CSR common-refinement artifact. It
validates complete source/target coverage and applies conservative cell-integral
transfer. A remesh is therefore a new topology and geometry plus an auditable remap,
never an in-place mutation.

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
Volume-overlap CSR entries are never treated as synthetic faces.

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

`prepare_unstructured_conservative_remap(source, target, provenance=..., policy=...)`
intersects the two discretizations' cell meshes with the canonical
`phydrax.geometry.prepare_common_refinement` (exact meshcore predicates and clipping;
`CommonRefinementPolicy` sets coverage mode, relative coverage tolerance, and the
candidate/accepted/memory limits). The returned `PreparedUnstructuredConservativeRemap`
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

`UnstructuredSecondOrderRemapPlan` reconstructs `u_s + phi_s g_s . (x - c_s)` about
each source cell's certified decomposition centroid `c_s`. The weighted least-squares
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
residuals. Padded face/cell quadrature capacities remain fixed.
`UnstructuredFiniteVolumePlan.from_cell_mesh(mesh)` wires this certified geometry into
the ordinary owner/neighbor face block, spaces, quality report, reconstruction, and
dynamics path without a private polyhedral topology.


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
prepares the automatic remap under `remap_policy` (a supplied
`PreparedUnstructuredConservativeRemap` is admitted the same way); the result's
`automatic_remap` exposes its status and evidence, and a failed refinement fails the
event with `FAILED_RESOURCE_LIMIT` or `FAILED_COVERAGE`.

Only after coverage, conservation, positivity/admissibility, and resource checks pass
does a transaction stage the candidate `TopologyEpoch` (role `topology`), its
`FiniteVolumeTopologyArtifacts` (role `prepared-graph`), and the transferred content
(role `physical-state`, structure identity the epoch ID) as one
`phydrax.lifecycle.CompositionRebind`. A complete conservative cell remap between
distinct epochs transports content through its own `TopologyEpochTransition`, which
accepts only the remap image of the accepted cell averages; any other transfer
carries the remap-reported content change against the transaction tolerance. The
journal commit is published only with a published `CompositionRebindReceipt`,
exposed as the result's `receipt`. A refused receipt fails the event with
`FAILED_COVERAGE` and returns the accepted epoch, artifacts, and content objects;
`receipt` is `None` when the event fails before staging. The sliding-map refresh of
`PreparedFiniteVolumeRuntime` refreshes its prepared runtime and accepted state only
from a published receipt.

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
