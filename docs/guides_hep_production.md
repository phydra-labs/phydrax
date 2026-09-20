# HEP production profiles

Phydrax closes named HEP production profiles by composing bounded native kernels with pinned external authorities. It does not claim one universal HEP engine.

## Ownership

`phydrax.particle_physics` owns shared particle-catalog references, event identities, bounded generator truth, named weights, associations, provider capabilities, reproducibility grades, and complete cross-section accounting. Material particles, accelerator macroparticles, detector tracks, hits, digits, clusters, reconstructed particles, and QCD thermodynamic states remain distinct.

`phydrax.applications.relativistic_scattering` owns the native two-to-two hard-process path. A `HardProcessPlan` binds a `ScatteringProcess`, two-particle center-of-momentum beam, scales, PDG identities, matrix-element identity, spin/color averages, symmetry factor, and fiducial cosine support. `produce_hard_events` returns the existing fixed-leg weighted stream, a bounded truth event, the generated/selected masks, derivative evidence, and a cross-section ledger. `integrate_hard_process` uses the existing frozen-grid VEGAS implementation.

General matrix elements, PDFs, NLO calculations, matching/merging, showers, hadronization, and unrestricted decays remain external. `ExternalHEPProvider` uses the existing pinned executable runtime and refuses capabilities absent from the provider contract.

Particle spectra are a separate staged profile. The internal result binds one
approximation profile, running trajectory, pole observables, calculator or
native-provider identity, artifacts, and distinct numerical/physical/warning
statuses. External calculators require a pinned
`particle-spectrum.external` capability and have no fallback. The native
`ScaleBVPPlan` supplies only caller-defined beta functions and boundary
constraints; it does not embed MSSM/NMSSM loop formulas.

## Event and weight invariants

Event status codes retain their provider namespace. The normalized `ParticleRole` vocabulary only distinguishes beam, incoming, intermediate, and outgoing roles. Mother and color relations retain explicit integer fields.

Every shard records attempted, generated, selected, positive, negative, and zero-weight counts; sum of weights; sum of absolute weights; sum of squared weights; filter efficiency; cross-section estimate; uncertainty; and overflow. An overflowed event is invalid and contributes to no observable.

Native RNG can claim exact semantic replay. External providers declare event-stable, seed-replayable, statistical, or uncontrolled reproducibility instead of inheriting a native guarantee.

## Collision environment

`CollisionEnvironmentPlan` declares mean and maximum pileup, luminosity, bunch spacing, source profile, and beam backgrounds. `assign_collision_pileup` folds the caller key by stable event identity. Primary and pileup truth remain separate in `CollisionEventBundle`; composition does not erase their source status namespaces or event identities.

## Interchange

`phydrax.interchange.hep` supplies schema-specific host mappings. The native
LHEF profile imports particles, mothers, color, momenta, masses, provider
status, and nominal weights. Alternative weights or unsupported production
vertices are reported as declared losses. The HepMC3 ASCII writer is
deliberately narrower and reports its missing vertex topology. ROOT import
requires an exact `HEPColumnProfile`; ROOT and Awkward are not treated as
schemas. The SLHA profile preserves generic block/decay ordering, scales,
numeric tokens, comments, and unknown blocks; duplicate entries and malformed
decays fail rather than being resolved by precedence.

## Derivatives

JAX transformability does not imply a physics derivative. Cuts, topology, status choices, pileup counts, provider calls, thresholding, and capacity decisions are stopped events. Generic `DerivativeEvidence` names differentiable and discrete parameters, estimator kind, stopped events, support, and evidence while retaining the shared `DifferentiationContract`.

## End-to-end simulated-collider profile

The composition root is:

```text
hard process or pinned event provider
→ LHEF/HepMC admission
→ collision overlay
→ detailed transport provider or bounded native propagation
→ hits and digits
→ tracking and calorimetry
→ reconstructed particles
→ weighted selections, response, unfolding, and likelihood
```

Each composed workflow requires independent qualification and cannot exceed the weakest component or adapter claim.
