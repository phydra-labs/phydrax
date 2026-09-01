# Electrokinetics

PHYDRAX discretizes passive Poisson–Nernst–Planck dynamics directly on a compatible
cochain complex. Concentrations and potential are degree-zero cochains; electric fields
and ionic fluxes are oriented degree-one cochains. The Hodge codifferential returns a
conservative nodal rate without a separate finite-volume/cochain interpolation.

## Electrostatics

`CochainElectrostaticBoundaryPlan` supports periodic, prescribed-potential, prescribed
normal-displacement, and mixed boundaries. Gauge-constrained periodic and pure-Neumann
systems validate total charge/flux compatibility. Dirichlet data is applied through an
exact lift. `CochainElectrostaticPlan` retains matrix-free self-adjoint positive-definite
linear-solve evidence.

## Electrochemical closure and flux

`IdealDiluteElectrochemicalClosure` evaluates mixing free energy, chemical and
electrochemical potentials, osmotic pressure, and charge density from one species
schema. `PreparedCochainElectrochemicalFlux` uses a cancellation-safe Bernoulli
function and exponential-fitted oriented flux. The discretization preserves constant
and discrete Boltzmann equilibria, exact inter-node transfer, and per-species mass.

`PoissonNernstPlanckPlan` solves potential, evaluates ionic flux, and reports free
energy, charge-rate defect, and an explicit positivity restriction. Its transactional
explicit step accepts only positive, finite, conservative, non-energy-increasing
candidates.

## Electrodes and flow

`ReactiveElectrodePlan` evaluates Butler-Volmer mechanisms on declared boundary nodes,
evolves surface species and capacitive surface charge, and exposes bulk boundary flux
and Faradaic current with a charge-current ledger.

`CochainMACTransferPlan` maps integrated edge electric force to physical MAC-axis force
while retaining a discrete power identity. `MultiphaseElectrolyteClosure` composes
binary phase, solvation, ionic, and dielectric energy before deriving chemical
potentials and total stress, avoiding force double counting.
