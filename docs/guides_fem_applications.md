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

`phydrax.applications.contact` provides persistent pair IDs and a frictionless
penalty law. Search/pair selection and active-set changes are discrete accepted-
step decisions. The tangent differentiates the current active branch; it does
not differentiate pair search.

## Fracture and XFEM

`phydrax.applications.fracture` provides a coupled displacement/damage form,
volumetric-positive tensile energy, irreversible history promotion, damage
bounds as accepted-state invariants, fixed-segment crack classification, and
Heaviside enrichment evaluation. Crack classification and enrichment activation
are discrete topology events; derivatives apply only for a fixed crack and
active enrichment layout.

## Static hyperelasticity

`phydrax.applications.solid_mechanics.neo_hookean_form` constructs a
`CellEnergyAction` for the logarithmic compressible Neo-Hookean reference energy.
The finite-element executor differentiates that scalar energy into the internal
residual, so the residual and stored-energy definitions cannot drift.

A two-component field on a two-dimensional mesh is embedded as plane strain with
unit out-of-plane stretch. A three-component field is fully three-dimensional.
Compose conservative dead loads separately with `BoundaryLoadAction`, apply strong
Dirichlet constraints through the finite-element constraint map, and solve the
compiled nonlinear residual. Plane stress and exact incompressibility are not
implicit modes.

## Fixed-mesh topology optimization

`phydrax.applications.solid_mechanics` provides a fixed-topology compliance
workflow over one cell density per finite-element cell. `DensityFilterPlan`
constructs a sparse, physical-radius, constant-preserving filter from cell
centers and measures. `SIMPInterpolation` maps the filtered density to material
modulus. `ComplianceTopologyProblem` then composes a caller-supplied physical
state residual, fixed load, volume inequality, converged state solver, and
`ReducedMMA`.

The application retains raw density, filtered physical density, modulus, volume
ratio, state-design KKT evidence, and every state/adjoint solve count. It does
not own a second finite-element assembler: a compiled FE residual binds through
the `state_residual(state, modulus, args)` callback.

`reanalyse_topology_design` is the independent honesty check. A caller supplies
a density transfer to a reference discretization and a reference compliance
solve. The report separates the ordinary coarse/fine discretization ratio from
excess stiffness over-report by the optimized discretization. A finite optimizer
result without this reanalysis does not establish mesh-independent compliance.

Current scope is steady linear compliance, fixed topology, scalar cell density,
one volume constraint, and a prescribed load. Stress, buckling, eigenfrequency,
manufacturing, and moving-mesh constraints remain outside this contract.

## Accepted-step boundary

`FiniteElementAcceptedStepSchedule` promotes fields and material trials exactly
once after acceptance. Rejected attempts preserve the previous fields,
material version, schedule cursor, and topology identity. Local mesh changes use
`FiniteElementTopologyTransaction`; candidate transfer or certification failure
retains the accepted mesh and state.

## Current scope

These workflows are single-device. Contact uses supplied persistent pairs rather
than a general self-contact search. XFEM currently classifies fitted T3 cells
against one fixed two-dimensional crack segment. Topology decisions, retry
branches, active sets, and crack propagation are not advertised as smooth
operations.
