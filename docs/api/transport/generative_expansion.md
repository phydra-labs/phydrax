# Advanced generative transport

This page describes the capability families built above the core real-vector VP/VE
score-diffusion path. They remain separate contracts because they use different event
coordinates, reference measures, stochastic kernels, and density semantics.

## Structured events and Wiener operators

`ArrayEventLayout`, `ComplexEventLayout`, and `PyTreeEventLayout` map public events to
stable real coordinates without inferring event axes from rank. `WienerNoiseLayout`
records ordered named Brownian blocks. A `WienerTerm` may be dense, diagonal, or an
explicit Lineax operator; mixed blocks lower to one matrix-free block-column action in
the canonical Diffrax backend.

## Matrix and state-dependent Itô diffusion

`MatrixGaussianDiffusion` provides exact constant-coefficient affine Gaussian
transitions. `StateDependentItoDiffusion` exposes state-dependent factors and computes
the exact Euclidean covariance divergence. Its reverse-coordinate drift is

```text
-b(x,t) + div(a)(x,t) + a(x,t) score(x,t),
```

and probability flow uses one half of the covariance-divergence and score corrections.
`general_reverse_diffusion_problem` and `general_probability_flow_system` lower those
fields into existing solver and dynamics contracts.

## Guidance

`ScoreContext` binds named conditioning values to one realization. Guidance results
record exact, approximate, or heuristic semantics:

- `TimeConditionedLikelihoodGuidance` is exact for a true noised-state likelihood;
- `PotentialGuidance` retains caller-declared approximation semantics;
- `ClassifierFreeGuidance` is marked heuristic when its interpolation weight differs
  from one;
- `GuidedScoreField` keeps every correction and identity ordered and visible.

A hard constraint is not implemented by projecting or clipping reverse steps.

## Discrete Gaussian and categorical diffusion

`DiscreteGaussianDiffusionSchedule` constructs cancellation-sensitive schedule values
in host certification precision. `AncestralGaussianDiffusion` and `DDIMTransport`
consume explicit epsilon, clean-state, score, or velocity predictions without a model
factory or implicit clipping. Samples record whether their standard-Normal terminal
reference is exact, approximate, or assumed.

`CategoricalDiffusionSchedule` owns exact finite transition and cumulative kernels.
`CategoricalReverseDiffusion` stores its terminal probabilities, relationship, and
identity; its default is explicitly the terminal pushforward of a uniform clean-state
prior. Its predictor returns clean-state logits, which the schedule converts into a
normalized reverse transition through the exact forward kernels. Categorical and
Gaussian diffusion remain separate: categories use counting measure and normalized
transition rows rather than dequantized Euclidean noise.

## Subspace and field laws

`AffineSubspaceLayout` provides metric projection, synthesis, and volume correction.
`SubspaceGaussianLaw` is a Hausdorff-density law; it does not fabricate an ambient
Lebesgue score. `FieldNoiseGeometry` builds the same coefficient contract from a
quadrature-orthonormal `SpatialNoiseBasis`, and `FieldGaussianDiffusion` corrupts those
coefficients rather than applying mesh-dependent IID nodal noise.

Cross-mesh transfer requires identical ordered mode IDs and retained covariance
spectrum.

## Manifold and complex events

`ManifoldProbabilityLaw` defines density relative to Riemannian volume.
`RiemannianScoreField` requires tangent scores. `IsotropicRiemannianDiffusion` supplies
intrinsic reverse and probability-flow drifts; the initial sampler is explicitly a
fixed-step retraction-Euler approximation.

`ComplexNormalLaw` uses real-coordinate Lebesgue density for proper circular complex
Gaussians. `ComplexVariancePreservingDiffusion` exposes both real-packed and declared
Wirtinger score views. A real log density is not labeled holomorphic.

## Scientific compositions

- `LatentDiffusion` composes an explicit latent sampler with a typed encoder/decoder
  representation. It exposes decoder-likelihood or sample-only capability and does not
  infer an induced normalized data density.
- `FixedTopologyGraphDiffusion` corrupts one floating node or edge payload while
  preserving the exact `GraphIR` topology.
- `AtomisticCoordinateDiffusion` works in mass-centered coordinate space;
  `AtomisticHybridDiffusion` combines continuous coordinates with categorical species.
- `PathCoefficientDiffusion` corrupts trajectory basis or innovation coordinates and
  distinguishes global from causal score dependencies.

## Implicit model contracts

`EnergyTarget` is unnormalized and never implements a normalized probability law
without a partition function. `PersistentContrastiveDivergence` uses fixed-capacity
replayable device particles and unclipped Langevin transitions.

`AutoregressiveLaw` is normalized by construction from ordered scalar conditional
laws. `ImplicitGenerator` is sample-only. Wasserstein adversarial evaluation never
claims a density for that generator.

## Deliberate boundaries

The APIs reject unsupported measure or geometry combinations. They do not introduce
image/text modality registries, diffusion-specific model factories, universal
backbones, hidden Gaussian dequantization, or balanced-to-unbalanced fallback.
