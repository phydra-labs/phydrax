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

The codifferential is the positive Hodge adjoint. With physical electric field
`E = -d(phi)`, Poisson is `delta(epsilon*d(phi)) = rho`, and Gauss's law is
`-delta(D) = rho`. Physical tail-to-head electrical current obeys
`rho_dot - delta(J) = 0`. The compatible Maxwell and PIC paths use the same signs.

## Electrochemical closure and flux

`IdealDiluteElectrochemicalClosure` evaluates mixing free energy, chemical and
electrochemical potentials, osmotic pressure, and charge density from one species
schema. `PreparedCochainElectrochemicalFlux` uses a cancellation-safe Bernoulli
function and exponential-fitted oriented flux. The discretization preserves constant
and discrete Boltzmann equilibria, exact inter-node transfer, and per-species mass.

The Bernoulli argument is the **drift-only** dimensionless potential. It must not
include the ideal `log(concentration)` term already represented by the endpoint
densities. Full electrochemical potential is used separately for dissipation.
`scharfetter_gummel_flux` exposes the geometry-independent edge law;
`stable_bernoulli` keeps both primal values and AD branches finite at zero and
large finite arguments.

`PoissonNernstPlanckPlan` solves potential, evaluates ionic flux, and reports free
energy, charge-rate defect, and an explicit positivity restriction. Its transactional
explicit step accepts only positive, finite, conservative, non-energy-increasing
candidates.

## Electrodes and flow

`ReactiveElectrodePlan` evaluates Butler-Volmer mechanisms on declared boundary nodes,
evolves surface species and capacitive surface charge, and exposes bulk boundary flux
and Faradaic current with a charge-current ledger.
`MACReactiveElectrodeBinding` maps those declared boundary slots to fixed MAC
boundary-adjacent cells, applies bulk species flux with the correct outward sign,
and retains the Faradaic current/charge defect. A configured resolved
electroosmotic plan advances concentrations and electrode surface/charge state in
the same candidate transaction.

`CochainElectrohydrodynamicForcePlan` retains an edge-cochain force and power
identity without claiming that primal edge arrays are MAC normal-face arrays.
`MultiphaseElectrolyteClosure` composes binary phase, solvation, ionic, and
dielectric energy before deriving chemical potentials and total stress, avoiding
force double counting.

## MAC-native PNP

`MACElectrostaticPlan`, `PreparedMACElectrochemicalFlux`, and
`MACPoissonNernstPlanckPlan` place potential and concentrations on MAC cells and
ionic flux on the exact MAC normal-face layout. The same Scharfetter-Gummel kernel
is reused; conservative MAC advection is added once. Periodic and homogeneous
Neumann electrostatics retain gauge and total charge/flux compatibility evidence.

`MACElectrohydrodynamicForcePlan` evaluates `rho_e E - grad(pi_osmotic)` once on
MAC faces and uses the MAC dual measure for mechanical power. Electric and osmotic
components remain separately observable.

## Electroosmotic flow

`ResolvedElectroosmoticStokesPlan` performs a fixed-capacity coupled PNP and
quasi-steady Stokes fixed-point candidate. Ionic content, Poisson solve, force,
viscous momentum, pressure projection, positivity, convergence, and ledgers commit
atomically. A failed candidate restores concentration, potential, velocity,
pressure, and time.

`ThinEDLElectroosmoticSlipPlan` is a separate DC Helmholtz-Smoluchowski model.
Admission checks Debye-length ratio, Dukhin number, bulk electroneutrality, finite
unit normal, and parameter support. Its boundary value forbids simultaneous
volumetric electrohydrodynamic forcing. AC, induced-charge, reactive, moving,
free-surface, and initially curved embedded-wall cases are outside this model.
