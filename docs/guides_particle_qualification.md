# Particle method qualification

Particle methods carry one of four maturity levels:

```text
experimental
qualified
production
certified
```

Execution success, numerical constraints, evidence claims, and production status
are separate. A finite trajectory does not imply that density, divergence,
pressure complementarity, walls, or free-surface constraints are satisfied.

`ParticleConstraintResiduals` uses dimensionless original-equation metrics,
including relative density and divergence L∞/L₂ errors, pressure
complementarity, wall constraints, and free-surface Dirichlet pressure.
`ParticleQualificationProfile` defines fixed thresholds. IISPH and DFSPH report
these residuals independently of their internal projected-iteration residuals.

`ParticleQualificationResult` can satisfy the production gate only when:

- execution succeeded;
- original constraints passed;
- every requested claim has satisfied evidence;
- maturity is production or certified.

The current advanced IISPH/DFSPH reference artifact remains experimental: its
steps execute successfully, but approximately 1.24% density residual does not
meet the default production profile. This is an intentional false-success guard,
not a benchmark failure.

Qualification artifacts retain method, configuration, source, precision,
backend, and evidence identities. Thresholds must be declared before evaluating
the benchmark.

## DEM qualification

`DEMQualificationProfile` applies contact-specific thresholds to
`DEMConstraintResiduals`: net internal force and torque, relative energy
balance, negative dissipation, friction-cone defect, maximum relative overlap,
wall action/reaction, and contact-history continuity.

`DEMDifferentiabilityMargins` separately records distance to contact activation,
distance to the friction stick/slip switch, and route-capacity success.
`DEMDiagnostics` additionally carries capillary constitutive-domain and
near-rupture extrapolation margins, conserved-liquid balance, cumulative
evaporation, and deforming-cell work/acceptance through the accepted state.
These margins qualify only the executed branch; they do not claim a smooth
derivative through a changed collision sequence.

Soft-contact DEM remains experimental. Dense/cell parity, restitution
refinement, oblique frictional collision, restart equivalence, and
fixed-capacity failure evidence remain required. New evidence covers fitted
capillary force--potential consistency, bridge birth/rupture and inventory
conservation, segment-virial balance and Galilean invariance, sparse
multilevel/dense pair parity with stable history keys, and periodic cell
conditioning/work rollback. These checks do not promote unrelated
configurations. Superquadric contact, multicontact correction, wall traction,
Finnie wear, and distributed execution retain independent status; distributed
claims remain absent.

## Particle-conversion qualification

Particle conversion is qualified by exact radial measure identities, thermodynamic inversion residual, interior heat/species cancellation, element balance, phase inventory, accepted-step energy closure, and agreement between reference Rosenbrock and structured tridiagonal backends. `tools/particle_conversion_qualification.py` records these cases as one machine-readable campaign.

Reactive coupling adds particle/fluid momentum, energy, and species closure; subsystem success flags; coupling iteration residual; and atomic rollback. `tools/reactive_cfd_dem_qualification.py` exercises both Strang and iterated schedules. `ParticlePhysicsSupportMatrix` reports claims compositionally: a successful DEM claim does not imply thermochemistry, superquadric, radiation, or distributed support.
