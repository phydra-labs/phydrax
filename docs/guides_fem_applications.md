# Finite-element applications

`phydrax.applications` contains executable, single-device workflows built on the
finite-element form, nonlinear, accepted-step, material-state, and adaptation
substrates. Application packages do not implement independent assemblers or
linear/nonlinear/time solvers.

Pin-jointed force-density form-finding is a discrete algebraic application, not a
finite-element constitutive workflow. See the
[force-density guide](guides_force_density.md).

## Phase field

`phydrax.applications.phase_field` provides backward-Euler Allen–Cahn and mixed
Cahn–Hilliard forms and step functions. The Allen–Cahn result reports free
energy before and after the accepted solve. The Cahn–Hilliard result reports the
finite-element mass before and after the mixed solve. Current energy evidence
uses the supplied double-well model and the executed nonlinear root; arbitrary
free-energy splitting is not inferred.

## Crystal plasticity

`phydrax.applications.crystal_plasticity` provides finite-strain multiplicative
crystal plasticity with oriented slip systems, power-law slip, isotropic
hardening, an implicit local slip root, first-Piola stress, candidate material
state, admissibility, and a suggested substep factor. The topology, grain map,
and accepted material state remain fixed during one global nonlinear attempt.

## Contact

`phydrax.applications.contact` combines collision surfaces, deterministic
fixed-capacity candidate epochs, an area-weighted physical barrier, conservative
linear-trajectory CCD, optional T3/T4 inversion bounds, static equilibrium, and
transactional implicit Newmark dynamics. Lagged isotropic Coulomb friction and
fixed-route implicit sensitivities remain explicit extensions of the same
stationarity problem. Search, feature selection, CCD, and lag refresh are
discrete derivative boundaries.

## Fracture and XFEM

`phydrax.applications.fracture` keeps sharp and diffuse fracture separate.
`PhaseFieldFractureModel` and `PhaseFieldAcceptedState` own diffuse degradation,
history, bounds, and accepted-step irreversibility. `CrackFrontGeometry`,
`SharpCrackTopology`, crack-side/tip quadrature, shifted enrichment, and
interaction-integral evidence own sharp cracks. Growth and crack-face contact are
accepted topology transactions; derivatives apply only inside one frozen
history/search/topology epoch.

## Static hyperelasticity

`phydrax.applications.solid_mechanics.neo_hookean_form` constructs a
`CellEnergyAction` from the canonical pointwise hyperelastic law. Plane strain,
three-dimensional, and block-diagonal plane-stress adapters therefore share the
same energy, first-Piola stress, tangent, admissibility, and Nanson conventions.

Exact or finite-bulk incompressibility uses `MixedHyperelasticModel`,
`mixed_hyperelastic_form`, a certified displacement/pressure space, and a
nonlinear root. It is not routed through `FunctionalSolver` minimization.
Conservative dead or certified pressure loads may contribute potential energy;
general follower loads use `MechanicalLoadAction`, preserve their nonsymmetric
tangent, and route through virtual work.

## Fixed-mesh topology optimization

`TopologyMechanicsProblem` composes one physical `DensityTransform`,
`MaterialInterpolation`, one or more `LoadCase`s, an explicit aggregation, and
an authoritative mechanics state solver. Generic conic filtering and tanh
projection remain in `phydrax.optim`; application density/material/load semantics
live in solid mechanics.

Every candidate carries independently recomputed state and adjoint defect
evidence. `NeuralVariationalStateSolver` may propose an initial state, but native
FE residual and transpose equations remain authoritative. Multi-load aggregation,
periodic homogenization with Hill–Mandel evidence, nonlinear branch gates, and
fixed-epoch contact/fracture admission are explicit contracts.

`TopologyReanalysisPlan` transfers the accepted design and performs mandatory
independent FE state/adjoint reanalysis. A finite optimizer result without this
evidence does not establish mesh-independent or physically admissible
performance.

Contact-search changes, crack initiation/growth, and undeclared branch changes
invalidate an ordinary reduced gradient. Learned operators remain proposal-only
at every design.

## Accepted-step boundary

`FiniteElementAcceptedStepSchedule` promotes fields and material trials exactly
once after acceptance. Rejected attempts preserve the previous fields,
material version, schedule cursor, and topology identity. Local mesh changes use
`FiniteElementTopologyTransaction`; candidate transfer or certification failure
retains the accepted mesh and state.

## Current scope

These workflows are single-device. Deformable contact currently uses a host
sweep-and-prune/nonlinear loop with JAX-local energies and supports certified
simplex inversion bounds only for nodal T3/T4 fields. Candidate capacity and
topology changes require an accepted host boundary. XFEM currently classifies
fitted T3 cells against one fixed two-dimensional crack segment. Topology
decisions, retry branches, active sets, and crack propagation are not
advertised as smooth operations.
