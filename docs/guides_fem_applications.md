# Finite-element applications

`phydrax.applications` contains executable, single-device workflows built on the
finite-element form, nonlinear, accepted-step, material-state, and adaptation
substrates. Application packages do not implement independent assemblers or
linear/nonlinear/time solvers.

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
