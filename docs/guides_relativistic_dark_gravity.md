# Relativistic dark-sector gravity

Phydrax keeps Newtonian PM, weak-field relativistic PM, and full Einstein--Vlasov Z4c
as separate scientific profiles. A density-only Poisson solve cannot be promoted by
attaching relativistic particle velocities.

## Units and local frames

`RelativisticUnitContract` uses `(E, c p_x, c p_y, c p_z)` and binds c, hbar, metric,
phase-space and scattering normalizations. `LocalRelativisticFramePlan.from_adm`
constructs one Eulerian orthonormal frame per accepted ADM stage. Runtime state binds the
static `frame_id` and exact `frame_token`, time, scale factor and realization digest.

## Stress-energy particle transfer

`RelativisticParticleState` stores stable IDs, weights, comoving positions, local and
covariant momenta, species, masks, incarnations and lineage. `RelativisticStressDepositPlan`
uses the existing particle-grid routes to deposit the ADM projection `(E,S_i,S_ij)` and
to gather metric/tetrad fields through the matched route. It reports shell, tensor,
source-integral, support and deposit/gather-adjoint evidence.

## Weak-field profile

`WeakFieldRelativisticPMPlan` evolves the full Poisson-gauge scalar, transverse-vector
and transverse-traceless sectors. It deposits endpoint stress, solves metric sectors,
advances relativistic geodesics and recomputes sources before atomic commit. Scalar-only
execution is admitted only when omitted momentum/anisotropic-stress channels satisfy its
bound. Gauge, spectral support, weak-field, metric, timestep and resource failures roll
back particles and metric.

## Einstein--Vlasov profile

`EinsteinVlasovMatterPlan` is a distinct Z4c matter participant. Every stage deposits
stress at one frame token, proposes Z4c geometry, gathers the stage metric/tetrad,
advances particles, recomputes endpoint stress and checks ADM constraints/source
exchange before committing both states. Initial data requires independent Hamiltonian
and momentum-constraint evidence.

`EinsteinVlasovAMRStressTransferPlan` and `EinsteinVlasovParticleMigrationPlan` use
existing NR AMR ownership and checkpoint owners. Repartition changes storage, never
stable IDs or physical stage identity.

## Differentiation

Fixed-topology smooth metric/geodesic kernels may expose narrow derivatives. Particle
migration, topology, source capacity, horizon/surface events, checkpoint restore and
profile switching are nondifferentiable.
