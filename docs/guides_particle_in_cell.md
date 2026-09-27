# Particle-in-cell methods

Phydrax particle-in-cell methods compose stable material-particle supports, measure-aware
structured splats, compatible cochains, native linear solves, and transactional fixed-step
state. They do not introduce a second particle or mesh runtime.

## Charged particle support

`ChargedParticlePlan` attaches one extensive macrocharge to every slot of an existing
`ParticleDiscretization`. Active charges are finite and nonzero; inactive charges are exactly
zero. Macro mass, activity, stable ID, support, and position/velocity field-space identity remain
owned by `ParticleDiscretization`.

PIC state stores spatial position and three-component proper velocity. `RelativisticPushPlan`
converts proper velocity to physical velocity, applies one relativistic pusher selected by
`method` (`RelativisticPusher`), and reports finite and subluminal evidence in
`RelativisticPushResult`:

- `"boris"`: the relativistic Boris map; it evaluates the magnetic rotation at the Lorentz factor
  after the first electric half kick, so crossed fields with E + v×B = 0 acquire a spurious
  drift-frame velocity when the rest-mass cyclotron frequency is unresolved;
- `"vay"`: Vay (Phys. Plasmas 15, 056701, 2008), which keeps the E×B drift exact at any γ but is
  not phase-space volume preserving;
- `"higuera-cary"`: Higuera and Cary (Phys. Plasmas 24, 052104, 2017), which keeps the E×B drift
  exact and preserves phase-space volume.

All three conserve |u| in a pure magnetic field. The speed of light is the exact
`speed_of_light` of the plan's `RelativityScaleContract`; fields, specific charge, and step size
are expressed in that scale and no unit conversion is inferred. Callers holding an
`ElectromagneticScaleContract` pass its `relativity`. PIC plans constructed without a pusher use
`"boris"` in `PIC_CODE_RELATIVITY`, a declared code-unit system with c = 1.

## Instantaneous particle–cochain transfer

`PICParticleCochainTransferPlan` prepares only existing `ParticleGridSplatPlan` instances:

- vertices for degree-zero charge;
- oriented edges for electric field gather;
- oriented faces for magnetic field gather.

Charge deposition returns extensive vertex charge, charge density under the vertex dual measure,
and the packed degree-zero cochain. Electric and magnetic gather first use
`StructuredCochainBridge` to recover physical edge-tangent and face-normal fields, then gather each
component from its exact tensor location.

Ordinary endpoint splatting is not current deposition.

## Compatible electrostatics

`CochainElectrostaticPlan` solves

```text
-delta(epsilon d phi) = rho
```

with the supplied cochain exterior derivative, codifferential, Hodge pairing, and `phydrax.linalg`.
Fully periodic problems require explicitly neutral particle-plus-background charge and use a
zero-mean potential gauge. Bounded problems initially support homogeneous Dirichlet potential.
Nonneutral periodic charge is rejected rather than silently mean-subtracted.

`ElectrostaticPICPlan` stores synchronized particle and field state and advances it with
kick–drift–kick. Each step deposits new endpoint charge, solves one new electrostatic field,
gathers E, reports kinetic/field energy, and commits only if transfer, solve, pusher, displacement,
and finite-state checks all pass.

## Charge-conserving electromagnetic coupling

`ChargeConservingCurrentPlan` currently supports uniform periodic 3-D grids and trajectories that
cross at most one cell per axis in one step. It splits a straight trajectory at crossed faces and
integrates cubical Whitney edge forms. The resulting degree-one current satisfies

```text
(rho_new - rho_old) / dt + delta(J_mid) = 0
```

under the exact `StructuredCochainBridge.codifferential`. Segment overflow, dropped support, or a
continuity defect rejects the step.

`PICMaxwellCurrentSourcePlan` is the native dynamic Maxwell source plan. The
same `PICMaxwellCurrentArguments` instance is passed to Maxwell stepping and
diagnostics, so both see the same deposited midpoint current through the
prepared `sources` collection.

`ElectromagneticPICPlan` keeps Maxwell as the sole owner of D, B, charge, material, boundary,
observer, CFL, and constraint updates. It owns only particle staggering and coupling:

1. gather E and B at integer-time particle positions;
2. push half-step proper velocity with the plan's relativistic pusher;
3. drift particles;
4. deposit midpoint current and endpoint charge;
5. advance `PreparedCompatibleMaxwell`;
6. compare deposited charge with Maxwell charge and certify continuity, Gauss, magnetic, energy,
   CFL, and displacement evidence;
7. commit the entire particle/field candidate atomically.

The base `ElectromagneticPICPlan` scope is fixed-population,
lossless, instantaneous-material, periodic 3-D without PML or material boundaries. Construction
rejects any bridge that is not three-dimensional with every structured axis periodic, and
`PreparedMaxwellCPML` rejects nonzero CPML width on a periodic axis, so the full 3-D plan cannot
carry an absorbing CPML layer. The advanced plans in
[Advanced particle-grid physics](guides_advanced_particle_grid.md) add bounded population
changes, collisions, ionization, reduced 1-D/2-D open electromagnetic PIC (the only PIC route that
accepts `MaxwellCPMLPlan`), moving windows, unstructured electrostatic/electromagnetic PIC, and
semi-implicit response without changing this base contract.

## Differentiation and limits

Weights and payloads differentiate inside a fixed route and segment program. Cell crossings,
periodic-image selection, segment count, support changes, solver failure, and step acceptance are
stopped branch decisions. No derivative is claimed through particle creation/deletion, collisions,
ionization, moving windows, repartitioning, or adaptive topology.

Support is configuration-specific rather than inherited across those plans. Quasi-cylindrical
PSATD and cross-device particle sharding remain unsupported; each advanced configuration requires
its own conservation, capacity, solver, and differentiation evidence.
