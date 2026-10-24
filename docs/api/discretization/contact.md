# Contact discretization

## Geometry and participants

::: phydrax.discretization.CollisionSurfacePlan

---

::: phydrax.discretization.PreparedCollisionSurface

---

::: phydrax.discretization.ContactParticipantScene

---

::: phydrax.discretization.LinearContactParticipant

---

::: phydrax.discretization.FunctionContactParticipant

---

::: phydrax.discretization.ContactProxyPlan

---

::: phydrax.discretization.FunctionImplicitContactGeometry

---

::: phydrax.discretization.SphereContactGeometry

---

::: phydrax.discretization.PlaneContactGeometry

## Typed feature identities and route keys

A participant scene owns one `ContactFeatureCatalogue` of typed
`(participant ID, feature kind, original feature ID)` triples over its surfaces
and static planes. Triples are unique and ranked lexicographically; the int32
rank is a device execution address and the original IDs remain scientific
metadata. Reordering source arrays with unchanged scientific IDs leaves the
catalogue unchanged. `ranks` and `identities` translate between scene features
and catalogue ranks; they clip indices rather than validating them, so callers
pass only catalogue-owned addresses.

`ContactRouteKeySchema(catalogue)` encodes one route as

```text
key = ((kind * F + left_rank) * F + right_rank) * W + witness_slot
```

with `F` catalogue features, `K = len(ContactStencilKind)` route kinds and
`W = len(ContactWitnessSlot)` witness slots. Preparation computes the exact
maximum with host integers and refuses a catalogue whose keys would reach the
canonical device integer maximum (int64, or int32 when x64 is disabled); that
maximum is reserved as the empty-slot `sentinel`. A key is meaningful only
together with its `catalogue_id`: equal integers from different catalogues are
not the same route. `encode`/`decode` are pure arithmetic and do not validate
digit ranges; `prepare_contact_stencil_batch` (host) and
`pack_contact_stencil_batch` (device) own orientation, slot-range, endpoint and
uniqueness checks.

This replaces the former unchecked `canonical_contact_route_keys` pairing of raw
feature IDs and the row-only `CompiledCandidateBatch`. Prepared topology rows
are `PreparedRouteCandidates`; runtime routes are `ContactStencilBatch` rows of
a `ContactCandidateEpoch`.

### Primitive profiles and witness slots

`ContactPrimitiveProfile.MESH` emits edge–vertex routes for 2-D or face-free
scenes, face–vertex and edge–edge routes for 3-D faced scenes, and vertex–plane
routes. `ContactPrimitiveProfile.SPHERE_CAPSULE` requires a face-free scene and
interprets isolated vertices as spheres and edges as capsule axes; capsule
endpoints do not emit duplicate sphere routes:

| Pair | Route kind | Witness slots |
| --- | --- | --- |
| sphere–sphere | `VERTEX_VERTEX` | one |
| sphere–capsule | `EDGE_VERTEX` (vertex left) | one |
| capsule–capsule | `EDGE_EDGE` | two |
| sphere–plane | `VERTEX_PLANE` | one |
| capsule–plane | `EDGE_PLANE` | two |

Capsule–plane slot `k` is the body-attached endpoint `k`; the two endpoint
gaps `n·a - d - r` and `n·b - d - r` jointly certify the whole capsule, so a
level capsule can carry a centered load without an invented torque. For
parallel capsule axes with an overlapping axial projection, `FIRST`/`SECOND`
are the two extremal closest-point pairs ordered along the left capsule's
first-to-second endpoint axis; a collapsed overlap, disjoint parallel interval
or nonparallel pair uses the unique branch in `FIRST` only. Degenerate
segments, coincident cores and zero normals are reported as degenerate rather
than given a fallback normal. A support tie has zero feature margin: the primal
support is valid, while derivatives through it are not admitted.

Planes are streamed analytically against finite primitives; no finite AABB is
invented for an infinite plane. Physical radii and solver clearances are
separate quantities.

::: phydrax.discretization.ContactFeatureCatalogue

---

::: phydrax.discretization.ContactRouteKeySchema

---

::: phydrax.discretization.ContactPrimitiveProfile

---

::: phydrax.discretization.ContactWitnessSlot

---

::: phydrax.discretization.ContactStencilLayout

---

::: phydrax.discretization.ContactRouteLayouts

---

::: phydrax.discretization.ContactStencilBatch

---

::: phydrax.discretization.prepare_contact_stencil_batch

---

::: phydrax.discretization.empty_contact_stencil_batch

---

::: phydrax.discretization.pack_contact_stencil_batch

---

::: phydrax.discretization.ContactStencilPacking

---

::: phydrax.discretization.PreparedRouteCandidates

`ContactStencilLayout` admits exactly two witness slots for `EDGE_PLANE`, one or
two for `EDGE_EDGE`, and one otherwise. Host preparation validates a
caller-authored batch: catalogue identity, capacity shapes, feature and slot
ranges, typed orientation, canonical order of symmetric kinds, complete and
distinct endpoints, finite weights, nonnegative separations, counts, and
duplicate active keys. Its `overflow`/`overflow_count` record a declared
overflow; consumers gate on that flag and on epoch completeness. Device packing
sorts accepted rows by key, preserves the discovered count when an upstream
buffer saturates, invalidates every packed row on overflow, and reports
duplicate identities in `ContactStencilPacking.duplicate_count` instead of
deduplicating them.

## Search, canonical epochs, kinematics, and safety

Every search backend returns the same `ContactCandidateEpoch` schema: six
fixed-capacity route batches, the searched coverage boxes, per-object envelope
radius, path interval and numeric revision, bound geometry (extents and planes),
a `ContactSearchStatus`, and `complete`. `covers` compares candidate boxes with
the searched coverage; `binds` refuses a foreign catalogue, topology or routing
identity and returns false for numerically changed geometry. The envelope is per
object, so two moving primitives consume twice that envelope. Output overflow,
traversal or stack exhaustion, duplicate routes and invalid inputs are
incomplete epochs; a truncated epoch is never a smaller successful contact set.
Discovered counts after exhaustion are work actually discovered, not the total
required capacity.

- `DenseContactSearchPlan` and `SweepAndPruneContactSearchPlan` are host
  correctness authorities. `search` returns `HostContactSearchResult`, whose
  `HostContactSearchEvidence` (elapsed seconds, estimated memory) stays outside
  the scientific epoch; `build` returns the epoch.
- `CompiledContactSearchPlan` prepares every legal oriented route row once and
  activates, counts and orders rows on device; it is the deterministic compiled
  authority for LBVH.
- `LBVHContactSearchPlan` is the implemented device broad phase over
  Morton-ordered AABBs with fixed traversal-visit, depth, and output budgets.

All four accept `end_positions` or a `ContactPathEnclosure`, so candidate
coverage can be built from the same trajectory enclosure that path
certification later uses.

::: phydrax.discretization.ContactSearchStatus

---

::: phydrax.discretization.ContactSearchLimits

---

::: phydrax.discretization.ContactCandidateEpoch

---

::: phydrax.discretization.DenseContactSearchPlan

---

::: phydrax.discretization.SweepAndPruneContactSearchPlan

---

::: phydrax.discretization.HostContactSearchResult

---

::: phydrax.discretization.HostContactSearchEvidence

---

::: phydrax.discretization.CompiledContactSearchPlan

---

::: phydrax.discretization.CompiledContactSearchResult

---

::: phydrax.discretization.CompiledContactSearchEvidence

---

::: phydrax.discretization.LBVHContactSearchPlan

---

::: phydrax.discretization.LBVHContactSearchResult

---

::: phydrax.discretization.LBVHContactSearchEvidence

---

::: phydrax.discretization.CachedContactSearchPlan

---

::: phydrax.discretization.ContactKinematicsEpoch

Kinematics evaluates every active batch of the epoch. Its success requires a
successful search, bound geometry, finite evaluation and zero failed contacts;
a failed active narrow-phase row is counted, never dropped from an otherwise
successful epoch.

---

::: phydrax.discretization.InclusionCCDPlan

---

::: phydrax.discretization.CertifiedAABBCCDPlan

---

::: phydrax.discretization.collision_free_step_limit

---

::: phydrax.discretization.ContactSafetyEvidence

---

::: phydrax.discretization.ContactGuaranteeLevel

---

::: phydrax.discretization.ContactGuaranteeEvidence

---

::: phydrax.discretization.SimplexInversionStepPlan

## Articulated trajectories and hard-contact admissibility

`collision_free_step_limit` remains the strict-separation CCD of barrier
consumers and rejects initially touching states. Hard-contact consumers use the
separate `certify_admissible_path(plan, scene, epoch, trajectory)`, which proves
`g(s) >= -gap_floor` for every candidate pair of the epoch and every `s` in
`[0, 1]` of the declared trajectory, including legal initial resting or sliding
contact. Pairs outside the epoch count as covered only when the epoch binds this
scene and its searched enclosure belongs to this trajectory, revision and
interval and covers the trajectory's full-path enclosure; otherwise the result
is `COVERAGE_MISMATCH`, not a weaker success. A trajectory whose declared
guarantee is weaker than `required_guarantee` is refused before tracing.

| `AdmissiblePathStatus` | Meaning |
| --- | --- |
| `ADMISSIBLE` | Gap floor proved over the whole interval |
| `INITIALLY_INADMISSIBLE` | The initial state already violates the floor |
| `CERTIFIED_PREFIX` | A violation was witnessed; only a prefix is certified |
| `UNRESOLVED_INTERVAL` | An interval could not be decided at the time tolerance |
| `INVALID_GEOMETRY` | Invalid interval bounds or primitive geometry |
| `WORK_LIMIT` | The interval budget was exhausted |
| `SEARCH_INCOMPLETE` | The epoch was incomplete |
| `COVERAGE_MISMATCH` | The epoch does not cover this trajectory/revision/interval |
| `NONFINITE_INPUT` | Nonfinite inputs |

`AdmissiblePathEvidence.successful` is true only for `ADMISSIBLE` with a
successful guarantee; `certified_prefix` is diagnostic and never a successful
full step. Pairs whose AABBs overlap are decided from midpoint primitive
distance minus Hausdorff displacement bounds, so oblique separated features are
not refused merely because their boxes overlap.

`ArticulatedCollisionGeometry` attaches positive-radius spheres (equal local
endpoints) and capsules (distinct local endpoints, whose order defines the
capsule axis) to bodies of a prepared fixed-base scalar-joint articulation, plus
static planes `n·x = offset`. Root-attached primitives are static; same-body and
excluded body pairs never form routes. Point velocities and force pullbacks are
native FK actions and their exact linear transpose.

`ArticulatedLiftedPath` is the binding segment path `q(s) = q0 + s h v` in lifted
scalar coordinates; it never wraps principal angles, so a full turn with equal
endpoints is a nonzero path. `ArticulatedContactTrajectory` evaluates native FK
exactly on that path. `frames`, `interval_bounds`, `enclosure` and
`midpoint_bound` enclose every ancestor hinge/prismatic transform over an
interval, including interior sine/cosine extrema and full turns, and contract
with the interval. Its declared guarantee is `ENCLOSURE_CONSERVATIVE`: the
enclosures carry a modeled floating-point allowance, not directed rounding, and a
caller cannot upgrade that level.

Motion bounds use one topological body pass followed by vertex gathers:
storage and work are linear in edges, body capacity and vertices, not the
vertex-by-edge Cartesian product. The hinge-arm triangle bound can be looser
for folded chains; it does not change the declared guarantee.

When every ancestor hinge has exactly zero increment, world motion is affine.
The trajectory intersects its generic box with a native-FK endpoint hull plus
modeled rounding allowance. This retains cancelling prismatic motions at
persistent contact. Opposite moving hinges do not qualify merely because their
net orientation change is zero.


::: phydrax.discretization.ArticulatedPrimitiveKind

---

::: phydrax.discretization.ArticulatedCollisionGeometry

---

::: phydrax.discretization.ArticulatedLiftedPath

---

::: phydrax.discretization.ArticulatedContactTrajectory

---

::: phydrax.discretization.ArticulatedPathFrames

---

::: phydrax.discretization.ContactPathEnclosure

---

::: phydrax.discretization.ContactPathMidpointBound

---

::: phydrax.discretization.AdmissiblePathPlan

---

::: phydrax.discretization.AdmissiblePathStatus

---

::: phydrax.discretization.AdmissiblePathEvidence

---

::: phydrax.discretization.certify_admissible_path

## Interface, trajectories, and distribution

::: phydrax.discretization.ContactInterfacePlan

---

::: phydrax.discretization.CubicHermiteContactTrajectory

---

::: phydrax.discretization.RigidSweepContactTrajectory

---

::: phydrax.discretization.DistributedContactPartitionPlan

---

::: phydrax.discretization.prepare_cell_mesh_collision_surface

---

::: phydrax.discretization.GeometricContactFilterPlan

---

::: phydrax.discretization.build_triangle_mortar_interface

---

::: phydrax.discretization.HydroelasticPressureFieldPlan

---

::: phydrax.discretization.HydroelasticPressureFieldState

---

::: phydrax.discretization.HydroelasticPatchExtractionPlan

---

::: phydrax.discretization.ContactHaloExchangePlan
