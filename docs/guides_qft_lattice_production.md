# Lattice fermions, RHMC, and QCD production

Production lattice-QCD workflows compose canonical lattice topology, matrix
gauge links, representation-aware matter transport, matrix-free Dirac
operators, pseudofermion actions, split molecular dynamics, distributed
fields, measurements, archives, and ensemble manifests.

## Fermion representations

Wilson, Wilson-clover, twisted-mass, staggered/HISQ, domain-wall/Mobius, and
overlap operators are separate concrete types. Every operator identifies its
spinor space, gauge field, boundary phases, parity layout, stencil/halo depth,
adjoint relation, and approximation evidence. No fermion-action enum activates
incompatible dormant fields.

Even-odd lowering is admitted only after a bipartite-stencil certificate. Its
Schur solution is always checked against the original operator. Clover and
smearing plans carry gauge/configuration dependency IDs and differentiated
pullbacks.

## Pseudofermions and rational actions

Two-flavor, Hasenbusch-ratio, and fractional-power pseudofermion terms have
separate refresh, proposal-action, force, and acceptance-action roles. Rational
coefficients are admitted only with a spectral interval and measured
approximation bound. Existing shifted-system solvers perform the multi-pole
action.

## Molecular dynamics

Force partitions lower to palindromic leapfrog or Omelyan compositions. A
trajectory records force/solve evidence, reversibility, Hamiltonian error,
group membership, and exact acceptance energy. Adaptation and learned
preconditioning stop before frozen production.

Checkpoints are published only after accept/reject. Candidate links, transient
momenta, force buffers, and solver workspaces are never restart state.

## Distributed execution

The lattice decomposition owns global site/link/face IDs, parity, local
extents, halos, and periodic/twisted neighbors. Execution distinguishes
packing, communication start, interior work, communication completion,
boundary work, and reductions. Projected spinor halos and block right-hand
sides are operator-specific lowerings; backend memory layouts remain private.

Native JAX semantics are authoritative. Optional kernel providers expose
granular capabilities and must agree with native finite references.

## Observables and ensembles

Gauge loops, flow observables, topology, propagators, meson/baryon
correlators, condensates, stochastic disconnected terms, and finite-temperature
observables each declare source placement, normalization, solve policy,
measurement randomness, and autocorrelation treatment.

Ensemble segments merge only when theory, action, algorithm, precision,
provider, RNG namespace, thermalization exclusion, and trajectory ranges are
compatible. Interchange archives are not restart checkpoints.

## Continuum claims

A continuum or thermodynamic-limit study tracks lattice spacing, volume,
anisotropy, masses, improvement, scale setting, renormalization, topology,
chain error, and fit/systematic variations independently. Orchestration never
automatically certifies the limit.
