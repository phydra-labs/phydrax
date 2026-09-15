# Semiconductor lasers

Phydrax keeps passive bidirectional wave propagation, semiconductor optical
response, reduced carrier-rate laser dynamics, and full drift-diffusion devices
as distinct models. There is no universal laser state and no automatic switch
between these regimes.

## Harmonic and power conventions

All optical amplitudes use the package convention `exp(-i omega t)`. Angular
frequency is always in rad/s. Bidirectional coupled-mode amplitudes are
power-normalized: `abs(a_plus)**2` and `abs(a_minus)**2` are watts at the named
reference planes. A power-gain coefficient therefore contributes half that
coefficient to the complex field-amplitude generator.

Left and right boundary ordering is explicit:

- left incoming drives the forward amplitude at the left reference plane;
- right incoming drives the backward amplitude at the right reference plane;
- left outgoing is the backward amplitude at the left reference plane;
- right outgoing is the forward amplitude at the right reference plane.

A bare scalar reflectivity is insufficient. Facets own complex amplitude,
phase/reference-plane, power-metric, and passivity semantics.

## Passive coupled-mode optics

`BidirectionalCoupledModePlan` is the passive piecewise-constant longitudinal
model for Bragg gratings, DBR/DFB sections, and related two-direction guided
structures. Preparation fixes the longitudinal support, carrier frequency,
reference propagation constant, detuning, reciprocal coupling, passive loss,
boundary convention, tolerances, and resource limits.

The prepared solver composes exact local two-by-two section relations through a
stable scattering representation rather than multiplying a long transfer-matrix
chain. Its result retains full forward/backward profiles, two-port scattering,
directional powers, distributed loss, residual, reciprocity, passivity,
conditioning, and provenance. Negative passive attenuation is rejected; active
gain belongs to a semiconductor application model.

## Frozen semiconductor optical response

Semiconductor optical response is owned by
`phydrax.applications.semiconductor`, not by the passive refractive-index law.
A response plan consumes explicit carrier-pair density, lattice temperature,
angular frequency, active-region projection, and provenance. It returns modal
power gain, field gain, propagation-constant shift, internal loss, validity,
and support identities.

Two initial response families are deliberately bounded:

- a caller-parameterized local linearized response around a declared operating
  point;
- a tabulated response on explicit carrier-density, temperature, and frequency
  supports.

Neither is a microscopic quantum-well theory or bundled process design kit.
Out-of-support queries fail rather than clip. Illustrative semiconductor
constants are not laser calibration data.

## Reduced traveling-wave dynamics

`TravelingWaveSemiconductorLaserPlan` is an explicitly reduced longitudinal
carrier-rate model. Its authoritative state contains forward/backward optical
amplitudes and local electron-hole pair density. Injection, A/B/C recombination,
gain compression, carrier-induced phase, passive loss, reciprocal grating
coupling, stimulated pair depletion, and complex facets are named mechanisms.
It is not a multidimensional drift-diffusion solve.

Stimulated optical gain removes equal electron and hole populations. The result
reports injected pairs, each recombination channel, stimulated transfer,
boundary optical power, passive loss, stored optical change, photon-energy
transfer, net-charge defect, and numerical residual. A carrier-source model
alone cannot certify complete electrical or thermal energy closure; those claims
require the authoritative electrical and thermal device owners.

Zero optical input and zero initial field remain exactly zero, including above
threshold. Deterministic startup therefore requires a declared seed. Spontaneous
startup is a separate stochastic contract with an explicit JAX key, realization
identity, covariance convention, and replay evidence.

## Threshold

Threshold is a deterministic homogeneous problem at the plan's declared fixed
carrier angular frequency. The zero-coupling branch uses the exact Fabry-Perot
round-trip power condition. A distributed grating uses bounded matrix-free
two-vector subspace iteration on the frozen optical step and bisects carrier
density until the dominant modal multiplier has unit magnitude.

Results retain the normalized forward/backward mode, modal eigenvalue, residual,
conservative modal gap, threshold carrier density/current/gain, mode and map
iteration counts, and grating identity/unitarity evidence. A grating threshold is
accepted only when endpoint values bracket threshold and the selected dominant
mode is converged and isolated. The solver does not search optical frequency,
track a continuation branch, or imply a multimode threshold.

No derivative through threshold bisection, subspace selection, or a mode crossing
is claimed. Above-threshold steady operation with spatial hole burning is a
separate nonlinear problem; it is not inferred from the small-signal threshold
result.

## Full-device boundary

A prepared drift-diffusion device may provide a frozen operating point to an
explicit active-region projection and optical-response lowering. This supports
small-signal cavity and threshold studies without pretending that the optical
field updates the device state.

Transient full-device coupling is not currently claimed. Such a model requires
stimulated recombination in the authoritative electron/hole residuals,
conservative optical-section/device-volume transfer, lattice and electrical
energy partition, a coupled block solver, and monolithic or certified waveform
residuals. A loose one-way loop is not equivalent.

## Qualification and limitations

Permanent qualification covers passive grating analytics, power and reciprocity
ledgers, transparency, gain-sign conventions, analytic Fabry-Perot threshold,
DFB mode/threshold behavior, carrier-photon exchange, refinement, deterministic
replay, and stochastic replay where enabled.

The reduced model does not claim transverse spatial hole burning, polarization
competition, microscopic bandstructure, quantum-well gain, thermal lensing,
multimode spontaneous-emission statistics, full Maxwell feedback, or process-PDK
accuracy unless a later concrete model and reference artifact establish those
capabilities.
