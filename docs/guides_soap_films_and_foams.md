# Soap films and foams: equilibrium, dynamics and topology events

`phydrax.applications.foams` computes equilibria, constrained pressure-driven
dynamics, Plateau-border drainage and deterministic thickness-triggered
rupture of explicit soap films and dry foams on a validated
`PreparedMultiRegionSurface` (see the
[multiregion surfaces guide](guides_multiregion_surfaces.md)).

`foam_candidate_profiles()` contains only `foams.*` capabilities and their KKT
release gate. Geometry, remeshing, hard-label extraction and topology-event
profiles are provided by `phydrax.geometry.multiregion_surface`.

## Material

`FoamMaterialPlan` carries an `InterfaceTensionMatrix` of effective pair
tensions `gamma_ij` matched to the surface regions by stable identifier. A soap
film has two liquid-gas surfaces, so `FoamMaterialPlan.soap_film` stores
`gamma = 2 sigma` and a spherical soap bubble obeys `p = 2 gamma / R = 4 sigma / R`.
Generic interfaces carry their single tension. The tension admissibility
(triangle inequality) is reported in the evidence. `FoamWireConstraints`
prescribes selected coordinates of wire vertices (pinned points or axis-aligned
sliding planes); wire positions are differentiable parameters.

## Equilibrium problem

`PreparedFoamEquilibrium` minimizes `E = sum_f gamma_f A_f` subject to
`V_r = V_r^0` for independent finite regions over nondimensional free vertex
coordinates. The native optimizers use `L = E + lambda^T c`, so region pressures
are `p_r = -lambda_r` with `dE = sum_r p_r dV_r`.

- **Pressure gauge.** Boundary labels are the zero-pressure reference. When the
  free coordinates cannot change the total volume of a set of finite regions (a
  rigid container), preparation keeps a maximal independent set of volume rows
  in table order by native rank decisions; each dropped region is the zero
  reference of its set (`pressure_reference_region_ids`) and its target is
  certified after the solve (`DEPENDENT_TARGETS_INCONSISTENT` otherwise).
- **Rigid-motion gauge.** Without wires, six linear rows fix the mean
  displacement and linearized rotation; their multipliers vanish at
  equilibrium (net force and torque) and are reported.
- **Routes.** `method="augmented_lagrangian"` (default) uses the native
  Powell–Hestenes method with a matrix-free Newton trust region, robust to the
  indefinite tangential curvature of irregular meshes; `method="sqp"` uses dense
  native SQP with the exact Lagrangian Hessian and converges quadratically near
  a regular equilibrium but its convex QP subproblems fail on indefinite
  iterates. Open films without volume or geometric gauge constraints use the
  native unconstrained Newton trust region. Dense routes are refused above
  `maximum_dense_dimension`.

## Evidence

`FoamEquilibriumEvidence` reports optimizer status and iterations, the
nondimensional stationarity and volume residuals, the dependent-target residual,
the discrete virial residual `(3 sum_r p_r V_r - 2 E) / (2 E)` (exactly zero at
any equilibrium of a free foam), and the dense KKT matrix `[[H, J^T], [J, 0]]`
spectrum: rank, inertia, condition number and the status `REGULAR`,
`RANK_DEFICIENT`, `INDEFINITE` (a constrained saddle), `ILL_CONDITIONED` or
`NOT_EVALUATED` (resource bound). Face-based junction wedge angles at triple
edges are first-order accurate indicators of Plateau's 120 degree law.

## Fixed-topology derivatives

`implicit_equilibrium` re-solves from a result and differentiates positions,
pressures and energy with respect to tensions, volume targets and wire
positions through `implicit_constrained_minimize`. It is refused with
`FoamDerivativeUnavailableError` unless the solve converged with `REGULAR` KKT
evidence.

Interior manifold vertices of an open boundary-to-boundary film have two
tangential reparameterization modes. A finite-cell separating film has the same
energy- and volume-neutral modes at the flat limit when its sheet-wide
point-to-plane defect is below
`FoamEquilibriumPlan.tangential_gauge_tolerance`. Preparation fixes both cases
with a deterministic orthonormal quotient gauge;
`tangential_gauge_dimension` reports its rows, and KKT evidence and derivatives
are evaluated on that physical quotient.

## Overdamped relaxation

`PreparedFoamRelaxation` integrates the inertia-free constrained gradient flow
`zeta A_v dx_v/dt = -dE/dx_v - sum_r lambda_r dV_r/dx_v` with film friction
`zeta` (`FoamRelaxationPlan.friction`), the lumped barycentric vertex area
`A_v` and one multiplier per independent finite-region volume. Eliminating the
multipliers with the mobility `M = diag(1 / (zeta A_v))` (zero on wire-fixed
components) gives `lambda = -(J M J^T)^{-1} J M grad E` through a native dense
solve, so the velocity conserves every constrained volume to first order and
`p = -lambda` are the region pressures (the equilibrium convention). Each
explicit step is projected back onto the volume targets by mobility-weighted
Newton iterations. The step is `time_step` limited so that no vertex moves
farther than `maximum_displacement_fraction` of the shortest active edge. The
evidence reports elapsed time, initial and final energy, whether the energy
decreased monotonically, the final volume residual, the success of every Gram
solve and the smallest face area and edge length.

This first-order route drives evolution between topology event passes, for
example through the loss of an equilibrium. It is not a dynamics model (no
film or air inertia, no drainage) and it is not collision-certified between
passes; event passes certify their own motions.

## Topology events in foam workflows

Every topology change goes through `apply_surface_events` of the geometry
package (see the [multiregion surfaces guide](guides_multiregion_surfaces.md)):
quality remeshing (`propose_remesh`), T1 pops of vanishing films
(`propose_t1_pops`), neck pinches (`propose_pinches`) and film merges
(`propose_merges`). Carry conserved quantities as extensive fields: film liquid
volume and surfactant amount as sheet fields, gas amount (or an incompressible
volume target) as a region field; event passes conserve them exactly and split
them by component volume when a pinched bubble becomes two regions. After a
pass that splits a region, rebuild the `FoamMaterialPlan` for the new region
table (for soap films `FoamMaterialPlan.soap_film(topology.region_ids, sigma)`)
and recompute volume targets from the region fields. Wire vertices listed in
`SurfaceEventPolicy.fixed_vertex_ids` are never moved or removed, so
`FoamWireConstraints` stay valid across epochs.

A typical epoch prepares the surface, relaxes (or solves the equilibrium),
proposes events on the relaxed state and applies one pass. Every event candidate
must undergo full self-intersection validation; `SurfaceEventPolicy` refuses a
validation policy with `check_self_intersection=False`. The host uses optional
meshcore adaptive predicates when available and an exact dyadic-rational
fallback otherwise, so midpoint splits with collinear welded vertices remain
certified without an unchecked split path. Any unresolved decision refuses the
whole transaction unchanged.

## Thickness-driven rupture

`FoamRupturePlan` consumes film thickness and the matching
`SurfaceFilmEvidence` on the multiregion `(vertex, region-pair)` sheet-slot
layout. A `BurstProposal` exists only when the film step status is `ACCEPTED`,
its nonlinear/positivity/conductance evidence passed, its geometry revision
matches, its rupture mask is true, and at least `minimum_trigger_slots` have
thickness at or below the declared threshold. Proposals are ordered by
thickness, stable region IDs and stable face IDs. There is no random rupture
route.

`apply_foam_rupture` deletes the complete separating sheet and merges the two
region labels through the typed E4 transaction. The supplied face IDs must
equal every active face of that pair; stale or partial support is refused.
Surviving sheet fields transfer conservatively, region extensive fields merge,
and the candidate undergoes the full self-intersection validation. Removed
film liquid is never silently redistributed. Without supported physical
borders, `apply_foam_rupture` adds it to the explicit scalar
`unresolved_rim_content`. E8's `apply_foam_rupture_with_borders` uses the same
public E5 transaction, resolves that new amount onto the bounding physical
border edges when complete support exists, and leaves the pre-existing
unresolved ledger unchanged. Unsupported cases retain the entire unresolved
amount and report `BORDER_SUPPORT_UNAVAILABLE`. The evidence closes the
sheet-liquid plus resolved/unresolved rim ledger, reports removed circulation
and gas amount/internal-energy residuals, and carries stable region lineage. A
failed candidate returns the source surface and every ledger unchanged.
Rupture is a topology epoch boundary and has no derivative.

The threshold is a user-supplied physical criterion backed by an accepted
drainage solve. This route does not predict a stochastic nucleation rate or
claim a universal rupture-similarity law.

## Constrained region-pressure dynamics

`RegionPressureAirPlan.incompressible` constrains target volumes on the
topology-first independent row basis. Each closed connected finite partition
drops exactly one row for its pressure gauge; open partitions connected to an
ambient volume action retain every finite-region row. Dependent rows remain
explicit in `FoamDynamicsEvidence`. `RegionPressureAirPlan.compressible`
instead evaluates an `AbstractBubbleCompartmentGasLaw` for every finite cell.
Gas amount and internal energy are extensive, and each update reports pressure
work plus amount and caloric rate residuals.

`PreparedFoamDynamics` offers two fixed-topology routes:

- `route="overdamped"` applies area-lumped mobility to the gradient of surface
  energy. One prepared volume linearization supplies volume values, JVPs, and
  pressure-weighted VJPs without materializing a region-by-vertex Jacobian.
  The independent volume Gram uses native operator actions and an SVD solve,
  then the same actions project incompressible cells back to their targets;
- `route="film-inertia"` uses the solver-owned
  `PreparedSHAKERATTLEPlan`. Its E6 volume-and-wire constraint Jacobian is a
  native operator with JVP and transpose actions and is reused for pressure
  recovery. Native minimum-norm projection reports numerical rank, condition,
  position/velocity residuals, projection work and rollback.

Only the incompressible route prepares a constraint-sized Gram, which is the
only dense volume object. `FoamDynamicsPlan.maximum_constraint_entries`,
`maximum_constraint_rank_actions`, and
`maximum_constraint_preparation_bytes` admit that preparation before any
derivative action. `PreparedFoamDynamics.constraint_basis.evidence` records the
topological partitions, numerical rank, logical retained bytes, required work,
and declared ceilings; a refusal raises
`FoamConstraintBasisPreparationError` carrying that evidence. Compartment-gas
dynamics instead records `NOT_APPLICABLE` with zero constraints, rank actions,
and basis bytes while retaining the matrix-free all-finite-volume JVP/VJP used
for gas pressure and work. Runtime native solve or resource-refusal status
remains available as `FoamDynamicsEvidence.constraint_linear_status`.

Both routes first limit the step by shortest-edge displacement and a
face-altitude swept-motion bound. The host `advance` boundary then certifies
every accepted leg with full-surface inclusion CCD and rolls a failed or
capacity-exhausted search back unchanged; the evidence carries the minimum
time of impact and certification flag. `advance(..., events=...)` applies E3/E4
proposals only after those fixed-topology legs and transfers compressible gas
by stable region lineage; a failed gas split/merge rolls the event back.
`fixed_topology_step` is the separately evidenced JVP/VJP surface. Host CCD and
every topology event set `derivative_available=False` on `advance`.

## Vortex-sheet air

`VortexSheetAirPlan` is the experimental, non-default air-inertia route. Its
state stores one circulation potential `Gamma` on each
`(vertex, region-pair)` sheet slot. The gauge is explicit: every initialization
and update subtracts the arithmetic mean independently on each region-pair
sheet. No circulation smoothing, filter, viscosity, or clipping is applied.
Setting `surface_tension_scale=0` is the exact zero-tension control and leaves
gauge-fixed circulation unchanged.

On each triangle the piecewise-linear intrinsic gradient gives
`gamma = n_pair cross grad_s Gamma`, where `n_pair` points out of the
canonical pair's first region. Da et al. use the opposite, higher-to-lower
normal; both `Gamma` and its source therefore change sign relative to their
equations (1) and (10), while the physical sheet strength is unchanged.
Face-centroid particles carry the one-point-quadrature integrated vector
vorticity `gamma A`. The units close without a fitted coefficient:
`[Gamma] = m^2/s`, `[gamma] = m/s`, `[gamma A] = m^3/s`, and the
`(gamma A) cross r / (4 pi |r|^3)` Biot--Savart term has units `m/s`.

The free-space velocity is evaluated by the canonical `VortexFMMPlan`; the
prepared octree is reused while geometry moves within its declared reference
envelope and is rebuilt after a topology epoch. A bounded
`GaussianErfDirectVortexPlan3D` run is an explicit qualification evaluator,
not a fallback hidden under the FMM route. Da et al. equation (12) used a
Rosenhead kernel with `alpha` equal to half the global mean edge length.
Phydrax instead declares its different canonical Gaussian-standard-deviation
core as
`max(minimum_core_radius, core_radius_fraction * local_mean_face_edge)`.
Consequently no finite-core coefficient is calibrated to the Kornek target;
qualification must demonstrate a shrinking core spread as this local radius
vanishes under refinement. Evidence reports the core range, FMM interaction
count, geometric tail bound, reference displacement, stale/overflow status,
and bounded direct/FMM relative error.

Surface tension follows Da et al. equations (10)--(11), rather than recovering
a signed scalar from a projected vector curvature. For each edge and incident
region, `K_i^e = |e| (pi - wedge_i)`; half is assigned to each endpoint.
The canonical-pair source is
`dGamma_ij/dt = +(sigma/(rho A_v)) (K_i^v - K_j^v)`, where the plus sign is
the normal-convention reversal noted above. `FoamMaterialPlan.soap_film`
stores the effective two-interface tension `gamma_film = 2 sigma`, so the
implementation uses exactly `gamma_film / 2`, not a free scale. Its units are
`[(sigma/rho)(K/A)] = m^2/s^2 = [dGamma/dt]`. The pair gauge removes only
the arbitrary integration constant. Symplectic Euler updates circulation
before evaluating velocity. Positions and velocities then advance through
`PreparedFoamDynamics.constrained_kinematic_step`, which uses the E6
SHAKE/RATTLE volume and wire constraints rather than a second projection
implementation. The result carries circulation/gauge, regularized kinetic and
surface energy/work, projection work, volume rank/conditioning, CCD, FMM, and
epoch evidence.

Host preparation builds a fixed-capacity sparse relation from each edge-wedge
contribution to the side of every incident canonical pair slot containing that
region. Runtime accumulation is therefore `(vertex_capacity, slot_width, 2)`;
it never forms a `(vertex_capacity, region_capacity)` intermediate. The
relation retains exact required-route and logical-byte evidence. An explicit
`maximum_curvature_routes` limit is refused before FMM preparation when the
topology requires more routes; otherwise the declared edge, valence, and slot
capacities provide the bound.

Before an E3/E4 event, `Gamma A_slot` is inserted as an extensive circulation
field and transferred by the event engine's sparse conservative relation.
After commit it is divided by the new slot areas and regauged; the raw
extensive transfer defect remains in evidence. The committed result marks
`rebuild_required`, because FMM and E6 structures belong to the old epoch.
Topology transitions and the host CCD boundary are nondifferentiable.

`PreparedVortexSheetAir.apply_rupture` performs the same injection through the
public E5 BURST transaction and adds vanished-sheet circulation to the explicit
`removed_circulation` ledger; surviving content plus that ledger closes the
signed circulation balance.


Only incompressible target-volume cells are admitted in this slice.
Compressible vortex-sheet cells, universal core independence, and a
production qualification claim are not inferred. The candidate profile stays
experimental until `tools/foam_qualification.py --scenario
vortex-sheet-bubble-modes` shows both mesh and core-radius convergence.

## Plateau-border drainage and B-on-E films

`prepare_film_sheet_slots` is the B-on-E adapter. It never makes the
non-manifold multiregion complex into a `TriangleTopology`. Instead, every
region pair is prepared as its own manifold `PreparedFilmSurface`, with a
fixed sparse gather/scatter map to E's `(vertex, region-pair)` slots. The
adapter exposes every sheet-boundary half-edge and its global E edge. A route
is Plateau-border-supported only when that global edge has exactly three
incident films. B1 lubrication, B2 symmetric surfactant transport and B4
moving-surface geometry therefore use their existing manifold content and
operator contracts. `SurfaceLubricationStepResult.boundary_exchange_m3` is the
per-vertex reservoir exchange (positive into the film);
`distribute_slot_boundary_rate` conservatively partitions that aggregate over
the explicit half-edge routes.

`PlateauBorderPlan` prepares only physical valence-three edges. Border liquid
volume and surfactant amount are extensive; cross-section is derived as
`A_e = V_e / L_e`. With declared triangular-channel resistance `C`, the
edge-cell flux is

`q = -A^2 / (C mu) (dp/ds - rho g dot t)`.

The capillary closure is the declared
`p_l - p_g = -c_sigma sigma / sqrt(A)`. At every network vertex one hydraulic
potential is the conductance-weighted incident value. Eliminating this scalar
makes the signed incident volume flux sum to roundoff; a degree-four network
vertex is the dry-foam quad-point mass and pressure balance. Preparation owns
fixed border/quad capacities and sparse endpoint relations. It refuses absent
physical borders, insufficient capacities and a triple edge lacking all six
sheet half-edge routes. Runtime contains no topology search or all-pairs
materialization.

`PlateauBorderState` stores sheet liquid/surfactant on E slots, border
liquid/surfactant on the fixed border axis, and the unresolved-rim scalar.
`PlateauBorderBoundaryFlux` declares signed sheet-to-border liquid and
surfactant rates separately. Sheet and border evaporation rates are explicit
nonnegative liquid sinks; a nonzero sink is refused unless
`evaporation_declared=True`, and evaporation never removes surfactant
implicitly. `PlateauBorderEvidence` reports the separate transfer and sink
ledgers, total film-plus-border conservation, quad residuals, Courant number,
minimum cross-section, support, positivity and rollback status.

`rates` and `step` are JVP/VJP-compatible on one fixed topology and matching
geometry revision. B4 motion uses `PreparedFilmSheetSlots.refresh` and
`PreparedPlateauBorder.refresh`; topology events require host re-preparation.
`apply_foam_rupture_with_borders` returns a source-epoch conservative rim
transfer record, then the consumer rebuilds the target-epoch border network.
No derivative is claimed across rupture, remeshing or any physical event.

### Border content across topology events

`PreparedPlateauBorder.reprepare_after_events(event, surface, film_slots,
state)` consumes the `SurfaceEventPassEvidence` of a committed pass that
started from the network's own epoch and geometry, prepares the network on the
target epoch with the same plan, and returns a `PlateauBorderAdaptation`. Its
`transition` is the nondifferentiable `TopologyEpochTransition` of one
extensive border field (liquid volume or surfactant amount) built on a
`ConservativeFieldTransfer`, so `transition.composition_transport(source,
target)` binds it into a `phx.lifecycle` rebind. Only two border rules are
declared:

- a border whose two stable vertex IDs survive keeps its content;
- a border split at a new vertex (parents are exactly its endpoints) gives each
  child `L_child / (L_1 + L_2)` of its content, which keeps the cell's uniform
  cross-section `V / L` and surfactant concentration.

Accepted T1 pops, pinches, merges, region splits and bursts raise
`PlateauBorderTransportError` before any target network is prepared, and so
does any collapse that removes or coarsens a border: a redistribution of border
liquid is not inferred from lineage. Sheet-slot content crosses the same pass
through E's own `SurfaceEventPassResult.transition`.
`examples/foam_junction_rebind.py` drains three double-bubble sheets into the
border ring (a declared first-order boundary suction, not a lubrication
model), declares the three sheets as one junction `InterfaceBinding`, splits a
border through one accepted composition rebind that reprepares the geometry,
sheet views, junction, network and rendering observation, renders the
published thickness, and shows the rupture refusal. Liquid and surfactant are
conserved to roundoff across the exchange and the rebind.

## Iridescent rendering

E remains the geometry and content authority. For a `PlateauBorderState`, each
manifold view in `PreparedFilmSheetSlots` obtains thickness from its own
`sheet_liquid_m3 / vertex_area` and orientation from its own
`PreparedFilmSurface.vertex_normal`. Passing those plain arrays and an explicit
support mask to `rendering.thin_film_surface_colors` yields linear and encoded
sRGB fields consumable by `SurfaceImagePlan`. It does not modify E state or
turn a rendered color into thickness evidence.

Plateau borders and junctions have liquid content and cross-section, not a
sheet thickness. They are therefore rendered with a separate support/status
mask; rejected and unsupported optical values stay NaN. The source-epoch
rupture frame in `examples/advanced_foam_iridescence.py` uses the canonical
rupture proposal to mark the removed sheet, overlays border support explicitly,
and verifies that physical content hashes are unchanged by rendering.

The example's alpha composition of independently exact sheet images is a
diagnostic display, not a calibrated transparent-foam radiance or
multiple-reflection model.

## Qualification

- Laplace pressure of a soap bubble converges at second order under icosphere
  refinement; the virial identity holds to roundoff.
- `StandardDoubleBubble` gives the closed-form equal-tension double bubble
  (Hutchings, Morgan, Ritoré and Ros, 2002): three spherical caps at 120 degrees
  with `1 / R_P = 1 / R_2 - 1 / R_1`. Discrete pressures agree to about 0.3 % at
  18 junction points and 0.2 % at 24.
- Unequal tensions recover the Herring/Neumann angles exactly for planar sheets
  on wires: `cos theta_k = (gamma_k^2 - gamma_i^2 - gamma_j^2) / (2 gamma_i gamma_j)`.
- Catenoid convention (Goldstein, Pesci, Raufaste and Shemilt, Phys. Rev. E 104,
  035105, 2021): rings of radius `R` at `z = +/- d`, `D = d / R`,
  `alpha cosh(D / alpha) = 1`; the stable branch ends at `x_c tanh x_c = 1`,
  `D_c = 0.66274`, `alpha_c = 0.55243`. `examples/advanced_catenoid_collapse.py`
  continues the discrete equilibrium to the stability limit and then, at
  `D = 0.7`, collapses the band by overdamped relaxation with remeshing until
  the neck pinches into two disks spanning the rings (region lineage
  `core -> core/0, core/1`) that relax toward flat discs. The collapse is
  quasi-static and friction-dominated; pinch-off dynamics with inertia are not
  modelled.
- Dynamics unit controls close incompressible volume residuals, native
  rank/conditioning/status, bounded volume JVP/VJP agreement and adjoint
  duality, constraint-resource refusal, overdamped energy behavior,
  SHAKE/RATTLE position and velocity constraints, compressible amount and
  caloric work ledgers, rollback, and a fixed-topology JVP against centered
  differences.
- The burst workflow triggers exactly at the accepted threshold, refuses
  rejected film evidence, resolves removed liquid onto a completely supported
  rim/border network, retains the unresolved ledger when support is absent,
  merges gas-region extensive fields with lineage, and rolls failed candidate
  validation back unchanged.
- `plateau-border-drainage` runs gravity drainage of double-bubble border
  loops at 6, 12 and 24 edges and records film/border conservation,
  surfactant conservation, Courant support, sparse boundary routes and runtime.
  Manufactured quad-point and fixed-topology derivative controls remain in the
  permanent tests.
  The recorded CPU campaign passed: relative liquid and surfactant residuals
  were zero at all three resolutions, maximum Courant numbers were
  `2.18e-8`, `4.81e-8`, and `9.84e-8`; first-step wall times were
  7.74 s, 6.76 s, and 6.92 s on the loaded qualification host.
- The vortex-sheet campaign derives its reference from Kornek et al. equation
  (2): a single interface has
  `omega_l^2 = gamma (l-1)l(l+1)(l+2) /
  (R^3 (rho_i(l+1) + rho_o l))`. A soap film has `gamma = 2 sigma`, and equal
  air densities on both sides give the modal added-mass denominator
  `rho_a(2l+1)`. The seed is rescaled so `R` is the same equal-volume sphere
  radius at every mesh level. The resulting `l = 2` reference is
  `omega_l^2 = 2 sigma (l-1) l (l+1) (l+2) /
  (rho_a R^3 (2l+1))`; `FoamDynamicsPlan.areal_mass` is only the E6 constraint
  projection metric and is not counted again as air mass. The campaign records
  mode purity, a post-transient time-series fit, joint mesh/time-step
  refinement, local-core sensitivity, bounded-direct/FMM separation, gauge,
  volume, work, and resources.
- Kelvin (5.306) and Weaire–Phelan (5.288) cell areas are pinned references in
  `tools/foam_qualification.py`. They are periodic foams and are **not
  computed**: periodic volumes are refused until unwrapped coordinates exist.

## Nonclaims

- Irregular meshes may equilibrate at tangential saddles (reported as
  `INDEFINITE`); remeshing between solves improves but does not certify mesh
  quality.
- No compressible vortex-sheet cells or core-independent vortex-sheet claim;
  vortex air remains candidate until its Kornek mesh/core campaign converges.
- The Plateau-border closure is a declared one-dimensional triangular-channel
  model. Node-dominated resistance, dynamic meniscus shape, soluble bulk
  surfactant inside borders and post-burst target-epoch rim motion are not
  inferred. Unsupported burst rims remain in `unresolved_rim_content`.
- No random rupture or universal rupture-threshold/similarity-law claim.
- No derivatives across topology event passes.
- No periodic foams.

## References

- J. Plateau, Statique expérimentale et théorique des liquides (1873).
- J. E. Taylor, Ann. Math. 103, 489-539 (1976).
- K. A. Brakke, Experimental Mathematics 1, 141-165 (1992).
- M. Hutchings, F. Morgan, M. Ritoré, A. Ros, Ann. Math. 155, 459-489 (2002).
- R. E. Goldstein, A. I. Pesci, C. Raufaste, J. D. Shemilt, Phys. Rev. E 104,
  035105 (2021).
- D. Weaire, R. Phelan, Phil. Mag. Lett. 69, 107-110 (1994); R. Kusner,
  J. M. Sullivan, Forma 11, 233-242 (1996).
- J.-P. Ryckaert, G. Ciccotti, H. J. C. Berendsen, J. Comput. Phys. 23,
  327-341 (1977); H. C. Andersen, J. Comput. Phys. 52, 24-34 (1983).
- U. Kornek, F. Müller, K. Harth, A. Hahn, S. Ganesan, L. Tobiska,
  R. Stannarius, New J. Phys. 12, 073031 (2010), equations (2) and (9).
- F. Da, C. Batty, C. Wojtan, E. Grinspun, ACM Trans. Graph. 34(4), 149
  (2015), equations (1), (2), and (8)--(12); no code copied.
- S. A. Koehler, S. Hilgenfeldt, H. A. Stone, Langmuir 16, 6327-6341 (2000),
  doi:10.1021/la9913147.
- J. Nocedal, S. J. Wright, Numerical Optimization, 2nd ed. (2006).
