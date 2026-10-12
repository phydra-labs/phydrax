# Level-set DEM, packing, and MPM–grain contact

Level-set grains use the same `CompliantDEMMethodPlan` and
`PreparedCompliantDEMDynamics` as spheres, clumps, superquadrics, and convex
polyhedra. There is no second level-set stepper. See
`examples/level_set_dem.py`
for two grains sharing a single voxel template.

## Preparation and units

Prepare a complete, nonperiodic 3-D `PreparedSparseVoxelGrid`, then call
`VoxelGeometrySamplingPlan.sample_level_set(geometry)`. Its certificate has
`DistanceSemantics.LEVEL_SET`, approximate zero-set accuracy and piecewise-smooth
regularity, **not exact signed-distance semantics**. Exact-SDF requirements of
implicit barriers remain unchanged. Sparse differential sampling requires every
corner, even a zero-weight corner; missing support returns unsupported evidence
and NaN values/gradients, never interpolated background data.

`LevelSetShapeTemplatePlan(samples, node_ids, local_nodes, physical_areas,
material_id)` owns stable nonnegative local node IDs, local coordinates and
positive physical quadrature areas in m². Identical bodies reference one slot in
`LevelSetRigidShapeSetPlan.templates`; fields are not duplicated per owner.
Templates share topology, dtype and node capacity. Owner positions are COMs,
not the original voxel-field origin. Local field queries add the template COM.

Particle masses remain caller-owned. Supply `exact_volume`,
`exact_center_of_mass`, and `exact_inertia_per_mass` together when exact mass
properties are known; otherwise voxel-center sign quadrature computes volume,
COM and inertia per mass. Evidence reports voxel spacing, resolved interior
count, uncertain boundary-cell count and a conservative relative volume error
bound, plus COM and inertia-per-mass error bounds (infinite when the resolved
volume lower bound is not positive). `volume_tolerance` admits the volume bound,
not an exactness claim. Inertia and COM are quadrature estimates, not CAD
properties; refine and compare them.
No empty interior, incomplete volume domain, unenclosed zero set, or non-SPD
inertia is repaired. Exact properties do not turn the field into an exact SDF.

## Discrete energy and forces

Both A surface nodes against B's field and B nodes against A's field are
emitted. **Each direction receives half its physical area**, so the symmetric
contact energy is counted once:

- Gap: g = φ_h(X), overlap δ = max(−g, 0).
- Surface energy density: Ψ(g) = k_A δ²/2.
- Route energy: U = Ψ(g) w_A/2.
- Conservative force on the source node: k_A δ ∇φ_h w_A/2.

`LevelSetPenaltyNormalPlan.stiffness_per_area` is N/m³; damping per area is
N·s/m³. `LevelSetTangentialContactPlan` explicitly uses the same per-area units
for Cundall–Strack history stiffness/damping. **Raw interpolated gradients**
produce conservative force; unit normals define normal velocity, damping and
friction. DEM damping uses an evidenced no-tension projection; the MPM adapter
returns the direct penalty/dashpot traction. Existing point-force
normal/tangential laws, cohesion and rolling laws cannot be silently reused on
surface-area routes. Hertz curvature is not claimed for trilinear fields.

The relevant analytic sphere reference is the integral of this surface penalty
law, not Hertz's point-contact law. For equal spheres with radius R and center
distance d ≥ R, the axial force is
π k_A R/d² times the integral from d−R to R of
(R−r)(d²−R²+r²) dr. Grid/node refinement must converge to this reference;
resolution-dependent point-force stiffness is not a convergence oracle.

The focused float64 sphere test (R = 0.5 m, d = 0.8 m, k_A = 100 N/m³)
measured the following grid/node sequence against the exact 2.5852689545 N
surface-integral force. These are observed errors, not a universal order theorem.

| Grid | Surface nodes | Spacing (m) | Absolute force error (N) | Observed rate |
|---|---:|---:|---:|---:|
| 16³ | 256 | 0.125 | 0.2122487452 | — |
| 32³ | 512 | 0.0625 | 0.0594881441 | 1.8351 |
| 64³ | 1024 | 0.03125 | 0.0171265102 | 1.7964 |

The finest field has about 0.6625% error; the 32³ field alone has 2.3% error.
The test's 2% criterion therefore requires the finer field rather than relaxing
the tolerance or renormalizing node weights.

Every route uses a common world application point. Owner reductions preserve
force and total moment exactly up to floating-point summation. Stable history
keys contain the physical unordered owner pair, ordered source side and local
node ID; they are remapped even when a reused Verlet pair gains/loses active
nodes. Verlet rebuild bounds include orientation displacement times bounding
radius. Candidate scratch requires 2 P N_s slots. Active-route overflow refuses
the entire candidate; no top-k contact truncation or partial state commit occurs.

## Shared queries and MPM grains

`evaluate_level_set_point_query` is the one posed point-query kernel used by
DEM and `MPMLevelSetGrainContactParticipant`. It reports raw gradient, unit
normal, complete stencil, scale-relative gradient margin and per-route validity.
No medial-axis direction is invented. Derivatives are branchwise inside a voxel
cell; gradients can jump across cell faces. General second derivatives and
Hertz regularity are not guaranteed.

The MPM adapter composes an existing `AbstractContactParticipant`; no new MPM
state representation or transfer contract is introduced. Inputs are physical
per-vertex area, sample radius and surface mass. It returns generalized MPM
effort, grain force/torque, participant duality, force/moment defects, counts,
overflow and status. Zero-mass vertices are inactive. Failed support, gradient,
input, duality or capacity evidence zeros all accepted loads. Status codes are
0 admitted, 1 overflow, 2 unsupported query, 3 invalid input, 4 failed duality,
and 5 nonfinite effort. The caller must reject an unsuccessful candidate.
The dense reference search streams at most 256 candidate addresses at once,
retaining only the declared route buffer; it never allocates a vertex×grain
coordinate array. Surface and grain supports must be nonempty and candidate
addresses representable. Native deterministic relation reductions accumulate
MPM efforts and grain force/moment payloads.

## Signed-field bounds and packing

Prepared shapes expose `evaluate_signed_field_query` with value, Lipschitz bound,
evaluation error, support and success. Spheres and convex polyhedra return exact
Euclidean signed distance; clumps return the exact min-of-sphere field (not the
Euclidean distance inside overlapping components). These fields are 1-Lipschitz
with zero field-evaluation error. Superquadrics explicitly refuse certification.
Level sets bound their trilinear gradient and carry source enclosure/interpolation
error; without exact-source enclosure evidence the error is infinite and the
certified query refuses, although DEM field queries remain available. This keeps
later cut-volume consumers from treating arbitrary fields as exact distances.

`RandomSequentialPackingPlan(lower, upper, maximum_tries_per_body,
maximum_candidate_evaluations=...)` prepares a bounded sequential initialization.
Stable physical IDs address typed PRNG keys by `fold_in`, independently of owner
array ordering. Containment uses conservative body bounds and overlap checks use
the shape narrow-phase oracle. Attempt counts and partial candidates are
diagnostics only: incomplete packing is unsuccessful and has no accepted bodies.
The evaluation budget is checked before execution. There is no retry-until-full,
overlap repair, or partial accepted packing.
