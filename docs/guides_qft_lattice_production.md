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
separate refresh, proposal-action, force, and acceptance-action solve policies.
Rational coefficients are admitted only with a structural spectral interval
and measured approximation evidence. Each role may explicitly select streaming
three-term shifted Lanczos; generic defaults retain reusable projections.
Streaming admits a pole `z` only when `z` lies below the certified spectral
lower endpoint, so the exact zero pole is valid only with a strictly positive
lower bound.

Shifted solves report direct residual and solution-error evidence through
rational actions, pseudofermion results, trajectories, transitions, and sample
results. The force differentiates the physical `D†D` action while treating
solve vectors as stopped. Its evidence bounds shifted-solution error, not force
error; a force-error certificate would additionally require a bound on the
link derivative of `D†D`. Measured rational-approximation error remains separate
from solve error.

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
