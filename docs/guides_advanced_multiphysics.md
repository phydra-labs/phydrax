# Advanced compressible multiphysics

This guide covers the post-transport-unification capabilities whose implementation state
is recorded in `docs/data/multiphysics_capabilities.json`. An implementation is not a
qualification claim: every capability retains an explicit `implemented`, `experimental`,
`qualified`, or `unsupported` status.

## Accepted transport evidence

`BalanceLawTransportAdvance.accepted_integrals` owns the actual accepted transport
integrals. Ordinary finite-volume adapters expose their conservative face-flux ledger.
Constrained MHD exposes `ConstrainedMHDAcceptedIntegralLedger`, containing stage-weighted
integrated face fluxes, edge electromotive circulation, cell-content change, and magnetic
face-flux change. Rejected steps contain exactly zero accepted integrals.

`AbstractPreparedAcceptedStepCoupling` runs only after both source halves and transport
have produced a provisional accepted state. Its complete update remains transactional.
`ConservativeGravityEnergyCoupling` uses this hook to account for the change in discrete
gravitational potential energy.

## Dimension-generic constrained MHD

`ConstrainedMagneticStateLayout` defines state ownership:

- one dimension: normal magnetic flux is cochain-owned; two transverse components are
  cell-owned;
- 2.5 dimensions: in-plane normal magnetic flux is cochain-owned; the out-of-plane
  component is cell-owned;
- three dimensions: all magnetic components are face-flux cochains.

`StructuredCochainBridge.pack_normal_flux` and `unpack_normal_flux` provide one
orientation-safe interface in all three dimensions. Two-dimensional Faraday evolution
uses a degree-zero electromotive cochain; three-dimensional evolution uses degree-one
edge circulation.

## Reconstruction and UCT

`MHDPrimitiveReconstructionPlan` supports piecewise constant, PLM, WENO-Z, TENO, and
MP5 primitive reconstruction. The face-normal magnetic field is injected from the
cochain after reconstruction and before conservative conversion.

`HLLUCTElectromotivePlan` adds multidimensional upwind magnetic-jump dissipation to the
flux-CT reference. `MHDCTUPredictorPlan` supplies a coupled half-step cell/cochain
predictor. Characteristic reconstruction is represented by
`MHDCharacteristicReconstructionPlan` and requires a qualified face-local eigensystem.

## Physical boundaries and robustness

`ConstrainedMHDBoundarySet` combines cell traces with magnetic and electromotive boundary
semantics. Initial policies cover conducting slip walls, controlled outflow, and
prescribed inflow. Boundary normal magnetic flux remains authoritative.

`LocalMHDPositivityPlan` computes cell-local convex factors against an admissible
low-order state. `DualEnergyMHDPlan` retains material internal-energy and entropy evidence
for magnetically or kinetically dominated states. Their support remains experimental
until local edge-factor and boundary-crossing cases graduate.

## Source physics

`NewtonianSelfGravityPlan` accepts periodic, homogeneous Dirichlet, homogeneous Neumann,
and mixed transform boundaries. Nullspace projection is used only for periodic/all-Neumann
operators. `ConservativeGravityEnergyCoupling` accounts for gas plus gravitational energy
in an accepted transaction.

`TabulatedCoolingCurve` exposes an exact cumulative inverse-cooling coordinate for its
piecewise power-law table. `RadiativeCoolingProcessPlan(integration="exact")` uses that
coordinate; constant heating remains on the implicit signed-source route.

`ModalOUForcingPlan` accepts a geometry-neutral `ModalForcingBasis` and exact
`OrnsteinUhlenbeckRealization`. `BalanceLawCompositionPlan` declares process subcycles
and explicit, exact, implicit, or exact-stochastic integration identities.

`StoichiometricReactionNetwork` and `ThermochemistryProcessPlan` advect and react species
through `MultispeciesEulerSystem`, validate invariant nullspaces, and account for reaction
energy. `GrayRadiationDiffusionPlan` provides gray diffusion plus exact local
radiation-matter exchange. `MultigroupM1RadiationSystem` supplies realizability-preserving
hyperbolic moment transport.

## Non-ideal, AMR, mapped, and unstructured MHD

`NonIdealMHDPlan` builds resistive, Hall, and ambipolar edge electromotive corrections;
every magnetic update remains a cochain exterior derivative. Magnetic dissipation is
returned to material energy.

`DivergenceFreeMagneticTransferPlan`, `ElectromotiveForceRegister`, and
`ConstrainedMHDAMRSynchronizationPlan` provide nested-grid transfer and reflux-curl
operations. `AMRTopologyReplayPlan` fixes topology epochs for replay.

`MappedALEConstrainedTransportPlan` integrates magnetic flux through physical face-area
vectors and electric plus mesh-motion circulation along physical edges.
`GLMIdealMHDSystem` is the cell-centered unstructured route; it does not inherit CT
claims. `UnstructuredConstrainedTransportPlan` supplies topology-exact Faraday evolution
when a compatible cochain complex is available.

## Cosmology, inference, and learned closures

`phydrax.applications.cosmology` provides a parameter-differentiable FLRW background
with radiation, matter, signed spatial curvature, and CPL dark energy. Fixed quadrature
supplies radial/transverse comoving distance, angular/luminosity distance, volume, and
finite-epoch time integrals. Native first/second Lagrangian growth remains an explicitly
flat, smooth-dark-energy, matter-era Newtonian approximation.

Every tabulated product carries a dynamic physical-realization signature, static
producer/provenance, an enforced differentiability policy, and semantic field metadata.
`MatterPowerTable` distinguishes cold+baryon, total-matter, and massive-neutrino fields,
auto/cross power, gauge, linear/nonlinear stage, shot noise, and spatial dimension.
`LinearTransferTable` preserves signed field transfers. Precision Boltzmann and
massive-neutrino perturbations remain isolated external calculations through
`SubprocessLinearTheoryBackend`; their arrays are immutable constant products, not
custom-gradient primitives.

Native product consumers include model-card-checked multiplicative power corrections,
spherical-overdensity/NFW/linear-variance halo foundations, flat Limber density and
lensing predictions, linear Kaiser/AP multipoles, canonical CMB spectra/transforms, and
small-N periodic force qualification. Phydrax does not silently select Halofit, HMcode,
an emulator, a halo fit/finder, a survey likelihood, or a complete CMB solver.

Particles use comoving position `x` and canonical momentum `p = m a^2 dx/dt`.
`ParticleMeshGravityPlan.acceleration` solves the rescaled potential `psi = a Phi` from
comoving density, `nabla_x^2 psi = 4 pi G (rho_com - mean(rho_com))`. Periodic LPT,
KDK, collisionless PM, and gas-particle gravity reject nonzero spatial curvature.

`CosmologicalGasParticleGravityPlan` physically couples adiabatic comoving Euler cell
averages and collisionless particles through one shared periodic density, Poisson solve,
and endpoint force refresh. It commits or rolls back gas and particles atomically.
Cooling, chemistry, stars, feedback, MHD, radiation, tree gravity, AMR, and distributed
particle migration are separate unsupported capabilities.

`WhitenedFieldInferencePlan`, `ParticleMarginalLikelihoodPlan`, and
`SimulationSensitivityReport` compose existing inference substrates with field-valued
multiphysics simulations. `StructurePreservingFaceClosurePlan` and
`ConstrainedMHDClosurePlan` provide dissipative face corrections, edge-EMF corrections,
and explicit OOD fallback.

## Astrodynamics and astrophysical observations

`phydrax.applications.astrodynamics` binds scale, two-part epoch, and frame identity to
Cartesian states while keeping numerical solver time relative and dense. Universal
Kepler, modified-equinoctial conversion, Lambert branches, native force composition,
tabulated ephemerides, orbital events, direct and nearly-Keplerian N-body dynamics,
CR3BP, rigid-spacecraft loads, and measurement Jacobians delegate general integration,
geometry, particles, roots, and control to their existing owners.

`phydrax.applications.astrophysics` projects those states into an observer frame and
composes polynomial limb-darkened occultation with immutable photon-counting
bandpasses. Binned, image, frequency, ray-transfer, and complex-field response objects
are concrete array operators. Existing likelihood, state-space, posterior, and
BlackJAX contracts remain the only inference boundary.

Contacts, transfer-branch selection, event brackets, collisions, classical-element
singularities, provider loading, and capacity changes are discrete or singular
boundaries. Smooth fixed branches retain ordinary JAX derivatives; every other path
must be judged through its explicit validity/status evidence.

## Differentiability boundary

Gradients are exact for the realized discrete program. Temporal nodes, topology epochs,
process subcycles, stochastic paths, limiter regions, Riemann branches, and fallback
activation remain explicit discrete evidence. Exact stochastic marginalization requires
particle or pseudo-marginal methods; conditioning on one path is not marginalization.
