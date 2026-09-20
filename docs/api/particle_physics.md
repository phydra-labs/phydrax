# Particle-physics semantics

::: phydrax.particle_physics.ParticleCatalogReference

::: phydrax.particle_physics.ParticleSpeciesTable

::: phydrax.particle_physics.ParticleEventPlan

::: phydrax.particle_physics.HostEventRecord

::: phydrax.particle_physics.pack_host_events


::: phydrax.particle_physics.EventWeightSet

::: phydrax.particle_physics.CrossSectionLedger

::: phydrax.particle_physics.AssociationTable

::: phydrax.particle_physics.HEPRunContext

::: phydrax.particle_physics.ProcessNormalization

::: phydrax.particle_physics.SystematicSource


::: phydrax.particle_physics.HEPProviderBinding

::: phydrax.artifacts.DerivativeEvidence

::: phydrax.particle_physics.build_hep_qualification_bundle

## Particle spectra

`SpectrumApproximationProfile` binds the model, renormalization scheme,
RGE/threshold/pole orders, matching and electroweak scale rules, correction
sources, and primary-source identities. `SpectrumDiagnostics` keeps numerical,
provider, root, vacuum, perturbativity, running-tachyon, pole-tachyon, and
approximation-warning axes separate. A physically admissible result with an
approximation warning is not silently promoted to success.

`SpectrumCalculationResult` retains labeled/unit-bearing observables, the full
reported running-scale trajectory, approximation profile, provider identity,
input/output artifact identities, diagnostics, and content-derived result ID.

`ExternalSpectrumProvider` requires one pinned executable and an admitted
`particle-spectrum.external` `HEPProviderBinding`. Execution is host-only,
bounded, and fail-closed. There is no automatic calculator fallback.

`ScaleBVPPlan` is the native finite reference for equations
`d p / d log(Q) = beta(p)`: fixed-step log-scale RK4, explicit low/high
constraint dimensions, native Jacobian Newton correction, bounded
backtracking, full trajectory, residual history, accepted steps, and terminal
status. It supplies no MSSM formulas or loop corrections; those require
separately sourced model implementations.

`tools/particle_spectrum_qualification.py` combines strict SLHA round-trip and
an analytic exponential-flow BVP. `benchmarks/particle_spectrum.py` separates
SLHA parsing/extraction and scale-BVP work over integration resolution.

## Native SM/MSSM spectrum closure

`NativeSpectrumModelPlan` supplies source-pinned one-loop SM and
CP-conserving, third-family MSSM beta functions. The adaptive log-scale
Dormand–Prince route lands exactly on affine threshold maps and keeps accepted
steps and threshold indices. Multi-seed trust-region boundary solves retain
every distinct root and failure.

The MSSM surface adds tree electroweak breaking, neutralino/chargino/stop
matrices, the leading top/stop light-Higgs correction, and bounded CCB/tachyon
diagnostics. Parametric, scale, perturbative-order, and provider uncertainties
remain separate covariance components. Scans preserve failures.

SLHA parsing now preserves block header arguments, including QNUMBERS PDG
identities, scaled duplicate blocks, complex mixing partners, decays, unknown
blocks, and semantic warning/error records. Calculator adapters remain pinned
process boundaries; cross-qualification never selects a fallback provider.
The native profile is not a collider-exclusion or global-vacuum proof.
