# Particle-in-cell methods

Phydrax particle-in-cell methods compose stable material-particle supports, measure-aware
structured splats, compatible cochains, native linear solves, and transactional fixed-step
state. They do not introduce a second particle or mesh runtime.

## Charged particle support

`ChargedParticlePlan` attaches one extensive macrocharge to every slot of an existing
`ParticleDiscretization`. Active charges are finite and nonzero; inactive charges are exactly
zero. Macro mass, activity, stable ID, support, and position/velocity field-space identity remain
owned by `ParticleDiscretization`.

PIC state stores spatial position and three-component proper velocity. `RelativisticBorisPlan`
converts proper velocity to physical velocity, applies the relativistic Boris map, and reports
finite and subluminal evidence. Its `speed_of_light` belongs to the caller's explicit code-unit
system; no unit conversion is inferred.

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

`PICMaxwellCurrentSource` is the stable `CompatibleMaxwellPlan.current_source` adapter. The same
`PICMaxwellCurrentArguments` instance is passed to Maxwell stepping and diagnostics, so both see the
same deposited midpoint current.

`ElectromagneticPICPlan` keeps Maxwell as the sole owner of D, B, charge, material, boundary,
observer, CFL, and constraint updates. It owns only particle staggering and coupling:

1. gather E and B at integer-time particle positions;
2. push half-step proper velocity with Boris;
3. drift particles;
4. deposit midpoint current and endpoint charge;
5. advance `PreparedCompatibleMaxwell`;
6. compare deposited charge with Maxwell charge and certify continuity, Gauss, magnetic, energy,
   CFL, and displacement evidence;
7. commit the entire particle/field candidate atomically.

The initial electromagnetic scope is fixed-population, lossless, instantaneous-material, periodic
3-D without PML or material boundaries.

## Differentiation and limits

Weights and payloads differentiate inside a fixed route and segment program. Cell crossings,
periodic-image selection, segment count, support changes, solver failure, and step acceptance are
stopped branch decisions. No derivative is claimed through particle creation/deletion, collisions,
ionization, moving windows, repartitioning, or adaptive topology.

Not yet supported: true 1D3V/2D3V Maxwell reduction, nonperiodic electromagnetic current loss,
collisions, ionization, PML/open PIC, unstructured PIC, hybrid or semi-implicit PIC, quasi-cylindrical
PSATD, moving windows, or particle sharding.
