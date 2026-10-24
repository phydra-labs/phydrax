# Multiphysics and patch contact

Mechanical contact changes transport, surface evolution, and unresolved
microscale response. Phydrax models those effects as explicit interface closure;
it does not hide them inside a normal-force law.

## Hydroelastic patches

`HydroelasticMaterialPlan` defines pressure modulus, compliant slab thickness,
normal dissipation, friction, and velocity regularization. A compliant pressure
trace may be generated from local compression or supplied by a volumetric
pressure field. `evaluate_hydroelastic_contact` integrates normal and frictional
traction over `ContactInterfacePlan` quadrature and reports patch area,
resultant force/moment, pressure positivity, dissipation, and action-reaction.

`HydroelasticPressureFieldPlan` binds a canonical affine tetrahedral
`CellMesh`; pressure is supplied separately by
`HydroelasticPressureFieldState`. `HydroelasticPatchExtractionPlan.prepare`
uses certified host tetrahedron intersections to fix nonmatching overlap
polytopes. Pure-JAX evaluation cuts the affine pressure difference, returns
plus/minus cell IDs and interpolation weights, and reports predicate, tie, area,
pressure-balance, capacity, and derivative evidence.

Rigid-rigid hydroelastic contact is unsupported rather than replaced by point
contact. At least one side must provide a compliant pressure field.

## Canonical contact routes and device broad phase

Every scene owns a `ContactFeatureCatalogue` of typed
`(participant, CollisionFeatureKind, original feature ID)` identities. Canonical
ranks are the lexicographic order of those triples, so surface, participant,
vertex and primitive source permutations with unchanged scientific IDs give the
same catalogue and the same routes; original IDs are carried unchanged (never
hashed or narrowed). A route key is
`((kind * F + left_rank) * F + right_rank) * W + witness_slot`, with `F` the
catalogue size, kinds from `ContactStencilKind` and `W` from
`ContactWitnessSlot`. Preparation proves with exact host integers that every
value fits strictly below the device integer maximum (the reserved empty
sentinel) and refuses unrepresentable catalogues. Keys are meaningful only with
their `catalogue_id`; they are never transferred across catalogues.

Rows are oriented: vertex before edge/face/plane, edge before plane, and the
lower canonical rank first for vertex-vertex and edge-edge routes. Normals point
from the right feature to the left one, matching the coefficient signs. A
witness slot distinguishes stable witnesses of one primitive pair: capsule-plane
routes carry one row per capsule endpoint and parallel capsule-capsule routes
two extremal support rows; an inactive second support row is not a failure.
`ContactPrimitiveProfile.SPHERE_CAPSULE` interprets isolated vertices as spheres
and edges as capsules (radius = physical radius); capsule endpoints never
become sphere rows. Static analytic planes (`ContactParticipantScene(...,
planes=...)`) use an explicit bounded plane route with the exact box support
bound, never an invented finite box.

Dense and sweep-and-prune host authorities, `CompiledContactSearchPlan` and
`LBVHContactSearchPlan` all return one `ContactCandidateEpoch` schema
(device plans return it as `result.epoch` with backend evidence). Device
packing encodes, orders and counts rows on device with no host constructor or
fingerprint; static identities are content-free, so numeric updates never
change PyTree structure or trigger recompilation, while `revision` is dynamic.
Host timing and memory estimates are host evidence (`plan.search(...)`), not
epoch state. Overflow invalidates every row of the batch, traversal or stack
exhaustion reports `TRAVERSAL_EXHAUSTED`, duplicate route identities report
`DUPLICATE_ROUTE`, and a traversal-exhausted count is discovered work, not the
required capacity. Kinematics propagate nonfinite, degenerate or contained
active witnesses as `failed` rows and an unsuccessful epoch.

Search accepts a full-path `ContactPathEnclosure`; primitive boxes are hulls of
the per-vertex path boxes. `envelope_radius` is per object: finite pairs are
activated within `activation + 2 * envelope + extents` because both primitives
may move, plane pairs within one envelope. `epoch.covers(lower, upper)` checks
that later path boxes stay within the searched boxes plus the envelope ball;
`CachedContactSearchPlan` reuses an epoch only under that predicate (a bounded
`lax.cond` for device searches) and rebinds the enclosure revision.

Completeness also depends on numeric geometry that does not change vertex
paths. The epoch records the contact extents and plane normals/offsets it
searched with and the scene `routing_id` (pair policies, exclusions, static
masks). `epoch.binds(scene)` raises for a foreign catalogue, topology or
routing policy and is false after any radius, clearance or plane change;
kinematics, cache reuse and path certification require it. Search refuses
nonfinite or negative extents and non-unit or nonfinite plane normals before
any pruning (`INVALID_INPUT` on device, an error on the host).

`LBVHContactSearchPlan` orders full-path primitive boxes by Morton code, culls
node pairs conservatively and applies the exact pair activation and legality at
leaves. Static planes are streamed one at a time against the prepared
primitive set into the fixed output capacity, so no primitive x plane table is
stored; the primitive x plane work is reported as `plane_pair_work`. Node,
depth, visit, duplicate-code, stack, visit, and output budgets are explicit
evidence. The compiled legal-pair plan (which stores every legal pair) and
dense search remain candidate-equivalence authorities. “LBVH” is a bounded
algorithm/resource contract, not a universal device-optimality claim.
`contact_route_capacity_bounds(scene, profile)` gives never-overflowing row
bounds per kind in O(features) without pair tables. Zero capacities are legal
for scenes with no possible routes; a discovered candidate under zero capacity
overflows and fails closed.

## Rough contact

`HomogenizedRoughContactPlan` provides a differentiable pressure-separation and
real-contact-area closure for macroscale interfaces. `PeriodicRoughContactPlan`
solves a periodic nonnegative-pressure half-space problem using FFT compliance
and projected iterations. The result reports pressure, elastic displacement,
gap, complementarity, load, and contact area.

`hertz_sphere_half_space` supplies the analytical Hertz reference used to
qualify smooth patch pressure and force.

## Thermal, electrical, and mass transfer

`CoupledContactTransportLaw` consumes three oriented interface jumps:

```text
temperature
+ electric potential
+ chemical or hydraulic potential
```

Pressure and gap control the effective conductance. Frictional dissipation is
returned separately as generated heat. `assemble_contact_fluxes` partitions
that heat between participants while enforcing equal/opposite conductive heat,
electrical current, and mass flux.

### Rotational measures and fuel-gap exchange

`FiniteElementMortarMetricData` accepts `rotational_measure=CoordinateMeasurePlan`
and an explicit `coordinate_frame`. Supply unweighted meridional quadrature
weights; the native metric applies the rotational factor once. An
`axisymmetric-rz` plan therefore integrates a cylindrical interface with `2πr`,
not a planar interface or a second application-side circumference factor.
The measure frame must agree with the supplied reference or current geometry.

The nuclear-fuel application's `GapConductanceLaw` combines gas conduction,
source-bound accommodation and contact conductance, and gray-surface radiation.
Its assembled interface exchange is equal and opposite between the fuel and
cladding; connected gas volume and closed HBS pores are tracked separately.
Contact pressure and frictional state remain mechanical unknowns rather than
being inferred from a heat-transfer coefficient. See
[the fuel workflow](guides_nuclear_fuel.md) for its admission, source-support,
and scientific-release boundaries.

## Wear and cohesive evolution

`FrictionWearEvolutionLaw` advances accumulated slip, Archard-style wear depth,
cohesive damage, rate-state history, and film thickness. Damage and wear are
irreversible. Remeshing uses `ContactStateTransferPlan`, which applies affine
parent interpolation while preserving the maximum inherited damage and wear.

## Lubricated contact

`LubricationContactPlan` remains the local squeeze/shear/asperity closure.
`ReynoldsFilmPlan` adds a connected P1 surface-film owner with declared pressure
references and a fixed-iteration projected Reynolds cavitation solve. It
assembles variable `h³/(12 μ)` diffusion, squeeze and tangential transport,
interpolates pressure to `ContactInterfacePlan`, and uses the canonical
equal/opposite traction assembly.

`ReynoldsFilmEvidence` reports PDE residual, complementarity, flux balance,
load, dissipation, minimum film, active-set margin, and solver status. A
nonpositive film, unreferenced component, active-set tie, or failed solve is
never replaced by pointwise squeeze pressure. This is a Reynolds variational
inequality, not an Elrod–Adams mass-fraction model.
## Distributed ownership

`DistributedContactPartitionPlan` assigns stable vertex and route ownership,
counts cross-rank halo routes, and fails closed on halo-capacity overflow.
`ContactGraphPlan` forms route components for local block and additive coarse
preconditioning. Distribution never changes physical route keys or accepted
interface history.
