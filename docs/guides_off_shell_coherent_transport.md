# Coherent and off-shell dark transport

Phydrax exposes a staged family rather than one quantum-transport switch.

## On-shell coherent QKE

`CoherentDensityMatrixState` stores Hermitian positive-semidefinite internal density
matrices on fixed cell/momentum support. `CoherentTransportPlan` uses a Cayley
Hamiltonian commutator and explicit local Kraus collision maps. Trace, charge,
Hermiticity, PSD, Hamiltonian/frame identity and rollback are evidence. No post-step
Hermitization, eigenvalue clipping or trace normalization is permitted.

## Quasiparticle off-shell profile

`QuasiparticleOffShellState` stores spectral function A, occupation F, real retarded
self-energy, width and lesser/greater self-energies. `OffShellTransportPlan` checks
retarded causality, spectral positivity/sum rules, Dyson residual, KMS and narrow-width
moments.

## Kadanoff--Baym profile

`KadanoffBaymTransportPlan` implements first-gradient Wigner drift/backflow plus a
fixed-depth finite-memory convolution ring. Checkpoints bind exact support, frame,
initial-correlation policy and history. `continue_kadanoff_baym_epoch` uses the durable
dark-sector epoch runtime for semantically unbounded memory evolution while each
compiled ring remains finite.

## Gauge-covariant Wigner profile

`GaugeCovariantWignerPlan` uses existing gauge-link spaces and Wilson-line transport.
Abelian covariance and real-adjoint representations are admitted; unsupported
non-Abelian representations fail closed rather than dropping the link.
