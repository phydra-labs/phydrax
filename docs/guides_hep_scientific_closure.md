# Full scientific HEP closure

Phydrax separates authoritative host records, bounded JAX execution views, native scientific kernels, provider adapters, and governed qualification. It does not replace experiment frameworks, event generators, detector transport, reconstruction, distributed services, or live controls.

## Host records and bounded execution

`phydrax.particle_physics.HostEventRecord` retains ragged particles, vertices, incidence, named weights, provider status, and opaque namespaced attributes. `pack_host_events` lowers an exact declared subset into `ParticleEventBatch`, preserving production/end vertex indices and returning `EventPackingReport`. Capacity, relation, and source-event overflow reject affected records; opaque attributes are a declared projection loss.

Canonical host records have optional Awkward conversion in `phydrax.interchange.hep`. Awkward is a host representation, not the Phydrax scientific authority.

## Runs, conditions, normalization, and systematics

Operational coordinates and half-open intervals live in `phydrax.measurement`. A `ResolvedConditionSnapshot` records immutable payload references, external tags, checksums, authorities, dependencies, resolution time, missing requirements, and overlaps. Exposure records distinguish delivered, recorded, and certified luminosity, protons on target, live time, target mass, and detector exposure.

`HEPRunContext` composes exact run coordinates, beam conditions, resolved conditions, data quality, exposure, campaign, and stream. `ProcessNormalization` records process-level attempted/generated/accepted counts, signed-weight statistics, cross section, uncertainty, units, and provider. `SystematicSource` distinguishes normalization, event-weight, kinematic, multiplicity, replica, Hessian, envelope, provider, and calibration variations with explicit correlation scope.

## Provider ownership

`HEPProviderBinding` binds a generic released `CapabilityProfile` to one configuration, input/output profiles, units, frames, device/dtype support, side effects, licenses, reproducibility, and generic derivative evidence. Native applications depend on this provider-neutral binding; concrete SDKs remain under `phydrax.interchange.hep`.

## Statistics

`phydrax.particle_physics.statistics` provides a limited shared core:

- binned Poisson models with explicit additive/exponential modifiers and Gaussian constraints;
- native bounded fitting through `phydrax.optim`;
- unbinned datasets and extended mixtures evaluated from externally normalized component densities.

RooFit, RooStats, HistFactory, Combine, pyhf, iminuit, and HS3 remain external authorities/profiles. Unsupported semantics refuse conversion.

## Collider physics

`collider_theory` owns process/model/order/provider contracts, provider-supplied theory predictions, replica/Hessian/envelope covariance, and bounded quadratic EFT morphing. Metadata never promotes a provider to NLO, NNLO, or resummed correctness.

`collider_analysis` adds:

- native bounded anti-kT, kT, and Cambridge/Aachen reference clustering;
- provider-bound jet contracts for production FastJet;
- probabilistic fuzzy-jet reference clustering with explicit pileup responsibility and local-optimum evidence;
- multidimensional signed histograms;
- correction maps;
- ABCD background estimates;
- collider-stage lineage records.

Hard sequential clustering is derivative-invalid at merge-order changes. Fuzzy clustering is differentiable only within fixed component/initialization support.

## Detector and online replay

Detector extensions include governed candidate or external-official calibration payloads, primary-vertex reference fitting, and fixed nearest-association particle flow. Detailed transport, full tracking, PID, production particle flow, and experiment calibrations remain external.

`trigger_replay` is offline-only. It supports raw-fragment completeness/corruption, event-building replay, trigger DAGs, deterministic prescales, stream membership, latency/resource admission, and buffer/dead-time replay. It exposes no live endpoints, credentials, run control, prescale updates, hardware programming, or trigger deployment.

## Accelerator physics

The accelerator application adds admitted symplectic transfer maps, closed-orbit/tune evidence, multi-turn ring tracking with aperture loss, and a binned longitudinal wake reference. MAD-X, Xsuite, SixTrack, FLUKA, live controls, settings management, and interlocks remain external.

## QCD and heavy ion

Finite-density QCD adds typed transport-coefficient tables for shear viscosity over entropy, bulk viscosity over entropy, and baryon diffusion. `heavy_ion` binds independently qualified initial-state, pre-equilibrium, hydrodynamic, particlization, and afterburner providers to exact EoS/transport tables. Native functions audit energy-momentum/B-Q-S conservation and calculate weighted flow vectors and two-particle cumulants.

## Neutrino physics

The neutrino application provides three-flavor vacuum and constant-density matter oscillations, unitarity evidence, POT-governed fluxes, rate prediction, migration, and near/far transfer. Interaction generators, nuclear effects, detector response, and production reconstruction remain external.

## Flavor physics

Flavor support includes coherent amplitude composition and interference fractions, neutral-meson mixing, calibrated mistag probabilities, Gaussian time-resolution damping, and tagged time-dependent rates. Line-shape normalization, acceptance, experiment tagging/PID, and official fits remain provider-owned.

## Cosmological phase transitions

`cosmology.phase_transitions` provides a thermal quartic potential, analytic stationary points, thin-wall thermal action, and a spherical trapped-particle bubble reference. Particle-wall crossings are localized analytically; reflection/transmission use wall-frame kinematics; wall backreaction and energy residuals are explicit. The implementation is a clean-room scientific reference based on the published model and does not copy the unlicensed `bubbleSim` source. Compactness is not black-hole certification.

## Fixed-target and long-lived particles

The fixed-target application composes POT exposure, beam/target declarations, production and transport providers, weighted stage accounting, and cylindrical decay-volume acceptance. It does not clone generator, shielding, transport, or experiment frameworks.

## Distributed preservation

HEP interchange records closed dataset snapshots, checksummed files and replicas, source/container/CVMFS/SBOM identity, read-only workload exports for HTCondor, Slurm, Kubernetes, PanDA, DIRAC, or REANA, and preservation bundles tying run conditions, data, environment, statistical models, outputs, qualification, rights, and external approvals together. Phydrax emits immutable plans; it does not operate those services.

## Qualification

`build_hep_qualification_bundle` produces one exact `SupportTuple`, generic `CapabilityProfile`, and `ScientificClaimProfile`. Governed release gates cover contract, source, numerical, interchange, physics, uncertainty, derivative, resource, and preservation evidence. Experiment approval, calibration sign-off, data-quality certification, machine protection, and physics review remain external authority.
