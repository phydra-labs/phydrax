# Phydrax

Top-level package namespace. Most functionality lives in subpackages:

- `phydrax.domain`: domains, geometry, sampling, and domain functions
- `phydrax.discretization`: finite topology, support, field spaces, measures,
  prepared tensor/spectral/cochain/FEM/FV methods, transfers, temporal meshes,
  and approximation bundles/hierarchies
- `phydrax.signal`: differentiable windows, framing, finite convolution, causal
  FIR state, periodic Fourier and fixed wavelet transforms, and finite/streaming
  polyphase rate conversion
- `phydrax.topology`: exact finite-complex homology, validated filtrations,
  persistent homology, fixed-capacity diagram layouts, and topology–Hodge evidence
- `phydrax.metrix`: coordinate charts, tensors, Riemannian metrics, curvature,
  embedded geometry, and metric-aware stochastic calculus
- `phydrax.data_utils`: CSV loading and array scaling helpers
- `phydrax.operators`: differential/integral operators on `DomainFunction`s
- `phydrax.conditions`: residual, moment, observation, and physical condition declarations
- `phydrax.terms`: penalty terms and specialized numerical/data terms
- `phydrax.integration`: integration targets, sources, reductions, and realizations
- `phydrax.variational`: representation-independent scalar functionals, local
  field jets, semantic regions, and ordered functional evaluation evidence
- `phydrax.weighting`: exact and quadratically reconciled relative-entropy
  moment calibration with operator-native geometry, audited status, and implicit
  derivatives
- `phydrax.transport`: balanced finite-measure transport, Sinkhorn divergence,
  exact and sliced Wasserstein distances, and differentiable order operations
- `phydrax.kernels`: composable positive-definite covariance functions shared by
  Gaussian-process and coreset algorithms
- `phydrax.ein`: exact optimized contraction dispatch plus named static JAX
  rearrangement, reduction, and repetition
- `phydrax.linalg`: paired vector spaces, composable dense/matrix-free/block
  operators, linear problem contracts, reusable solve and factorization plans,
  and standard/generalized eigensolvers
- `phydrax.qualification`: exact support tuples, current evidence matrices,
  unsigned candidates, signed release indexes, and fail-closed admission
- `phydrax.lifecycle`: resolved run identities, explicit configuration migration,
  transactional repositories, topology-aware direct restart, and provenance
- `phydrax.service`: in-process reference orchestration, durable stores, injected
  scheduler/identity providers, tenant isolation, observability, and signing/trust
- `phydrax.closure_data`: identified flow series, filters and closure targets,
  chunked datasets, leakage-safe partitions, train-only normalization, and
  artifact-bound learned deployment
- `phydrax.statistical_dynamics`: finite CE2/GCE2 cumulants, beta-plane quadratic
  coordinates, segmented NILSS, and logical distributed/restart layouts
- `phydrax.backends`: explicit lazy PETSc, SLEPc, PyAMGCL, and NVIDIA AmgX
  lifecycle bridges with capability, transfer, convergence, and provenance evidence
- `phydrax.enforcement`: exact condition transforms and enforcement programs
- `phydrax.optim`: finite and continuous scalar, least-squares, proximal,
  constrained, state/design, and stochastic optimization with explicit solution maps
- `phydrax.nonlinear`: nonlinear systems, fixed points, preconditioning,
  multigrid, complementarity, and implicit root derivatives
- `phydrax.continuation`: generic parameterized residual curves, stability and event
  evidence, branch switching, and fold/Hopf/pitchfork workflows
- `phydrax.nn`: neural network components and structured models
- `phydrax.ml`: case-aware native machine-learning batches, fitted models,
  workflows, model selection, metrics, inspection, and artifact interoperability
- `phydrax.sparse`: JAX-native sparse relations, routing, reductions, and linear actions
- `phydrax.solver`: functional solvers and direct ODE/SDE, jump, hybrid, and
  semidiscrete SPDE integration
- `phydrax.dynamics`: typed continuous/map systems, evolution and trajectory
  contracts, DMD/EDMD, sparse equation discovery, periodic orbits, and nonlinear
  dynamics/chaos analysis
- `phydrax.control`: finite-horizon control contracts, linear-system analysis,
  trajectory optimization, compiled QPs, and MPC
- `phydrax.stochastic`: global Wiener/Poisson/composite realizations, random
  fields, process laws, and explicit path coupling
- `phydrax.uq`: posterior inference, uncertainty propagation, calibration, filtering,
  smoothing, and sensitivity analysis
- `phydrax.export`: deployment helpers for learned inference functions

## Sparse execution substrate

`phydrax.sparse` factors out the gather–message–reduce mechanics shared by
fixed interpolation stencils, case-local neighborhoods, graph edges, and
cochain incidence maps. `EdgeRelation` represents arbitrary source-to-target
routes; `RowRelation` represents fixed-width target rows with explicit case
boundaries. Invalid capacity slots are numerically inert under routing and
reduction, including when their stored payloads are non-finite.

`SparseLinearMap` attaches scalar route coefficients and exposes forward,
transpose, and conjugate-adjoint actions while preserving trailing payload
dimensions. `SparseCoordinateOperator` binds the same route algebra to
`phydrax.linalg` spaces and pairing-aware adjoints. Provider-neutral
`SparsePattern`, `SparseColoring`, and `SparseDerivativePlan` artifacts support
native compressed JAX evaluation. ASDEX supplies compile-time global detection
and optimized coloring when a pattern is not already known.
Dense and SciPy conversions remain explicit interoperability operations, not
execution fallbacks. Sparse-grid
quadrature, sparse Gaussian-process approximations, stochastic probes, and
generic matrix-free callables remain in their semantic subsystems.

::: phydrax.sparse.EdgeRelation

---

::: phydrax.sparse.RowRelation

---

::: phydrax.sparse.SparseLinearMap

---


::: phydrax.sparse.SparseCoordinateOperator

---

::: phydrax.sparse.compile_sparse_jacobian

---

::: phydrax.sparse.compile_sparse_hessian

---


::: phydrax.sparse.gather_routes

---

::: phydrax.sparse.route_reduce

## Stochastic realizations and fields

`phydrax.stochastic` owns replayable randomness and coupling semantics. A PRNG
key alone is not a realization identity: Wiener paths, Gaussian field modes,
and coefficient processes retain explicit realization and coupling IDs.

::: phydrax.stochastic.WienerRealization

---

::: phydrax.stochastic.SpatialBasisSynthesis

---

::: phydrax.stochastic.GaussianCoefficientRealization

---

::: phydrax.stochastic.StaticGaussianRandomField

---

::: phydrax.stochastic.TransformedRandomField

---

::: phydrax.stochastic.GaussianFieldCoupling

---

::: phydrax.stochastic.gaussian_field_diagnostics

## Stochastic process contracts

Pathwise transitions consume named driver segments. Marginal transition laws
integrate that driver out. These interfaces remain separate so a transition
density is never mistaken for one reusable stochastic flow.

::: phydrax.stochastic.AbstractPathwiseTransition

---

::: phydrax.stochastic.AbstractMarginalTransitionLaw

---

::: phydrax.stochastic.AbstractProcessDistribution

---

::: phydrax.stochastic.ProcessRealization

---

::: phydrax.stochastic.LatentGaussianCoefficientProcess
    options:
        members:
            - __init__
            - realize
            - evaluate
            - pathwise_transition
            - marginal_transition

---

::: phydrax.stochastic.GaussianProcessDistribution

---

::: phydrax.stochastic.process_query_consistency

---

::: phydrax.stochastic.cocycle_objective

---

::: phydrax.stochastic.semigroup_objective

---

::: phydrax.stochastic.process_sample_statistics

---

::: phydrax.stochastic.gaussian_process_diagnostics

## Finite-activity jumps and composite realizations

`PoissonClockRealization` owns prefix-stable unit-rate Poisson thresholds and
mark keys. `JumpEventBatch` is the canonical masked event representation:
every path has an explicit status, valid-prefix mask, event time, channel,
mark, and optional left/right state. Algorithms must use the mask and status;
padding values are never event evidence.

`JumpProcess` accepts callable intensities, jump maps, and optional mark
sampling. `ChemicalJumpProcess` supplies mechanism-derived combinatorial propensities
and stoichiometric updates without a separate reaction schema.
`CompositeStochasticRealization` combines named Wiener and Poisson
realizations with one sample layout and support. Its path labels, realization
ID, and coupling ID include every named component.

::: phydrax.stochastic.CompositeStochasticRealization
    options:
        members:
            - __init__
            - num_paths
            - component
            - path_labels

---

::: phydrax.stochastic.AbstractJumpProcess

---

::: phydrax.stochastic.JumpProcess
    options:
        members:
            - __init__
            - intensities
            - jump
            - sample_mark

---

::: phydrax.stochastic.ChemicalJumpProcess
    options:
        members:
            - __init__
            - intensities
            - jump
            - conservation_residual

---

::: phydrax.stochastic.PoissonClockRealization
    options:
        members:
            - __init__
            - extend
            - thresholds
            - mark_keys
            - path_keys

---

::: phydrax.stochastic.JumpEventBatch
    options:
        members:
            - __init__
            - counts
            - successful
            - states_at

---

::: phydrax.stochastic.jump_status_name

---

::: phydrax.stochastic.StochasticTrajectory

## Qualification, lifecycle, and release admission

Candidate evidence is not release evidence. `QualificationMatrix.evaluate()` resolves
every named predicate against current `QualificationEvidence`.
`CapabilityProfile(released=False)` remains an unsigned candidate; `ReleaseIndex` and
`require_profile` are the separate signed, time-scoped, exact-support admission
boundary.

::: phydrax.qualification.SupportTuple

---

::: phydrax.qualification.QualificationEvidence

---

::: phydrax.qualification.ObservedResourceRecord

---

::: phydrax.qualification.ForecastResourceRecord

---

::: phydrax.qualification.QualificationMatrix

---

::: phydrax.qualification.CapabilityProfile

---

::: phydrax.qualification.ReleaseIndex

---

::: phydrax.qualification.require_profile

---

::: phydrax.qualification.ReferenceArtifactManifest

`ResolvedRunSpec` binds support dependencies and execution identities before launch.
Migration follows only explicit acyclic forward edges and retains complete lineage.
Repositories expose transactional immutable chunks; direct restart validates canonical
source ranges and destination ownership before injected range I/O.

::: phydrax.lifecycle.ResolvedRunSpec

---

::: phydrax.lifecycle.CompatibilityRegistry

---

::: phydrax.lifecycle.MigrationReport

---

::: phydrax.lifecycle.POSIXArtifactRepository

---

::: phydrax.lifecycle.S3ArtifactRepository

---

::: phydrax.lifecycle.TopologyRestartRelation

---

::: phydrax.lifecycle.prepare_direct_restore

---

::: phydrax.lifecycle.execute_direct_restore

## Service and security providers

`InProcessReferenceService` is a synchronous reference implementation, not a network
server. Durable storage, Slurm/Kubernetes scheduling, JWKS retrieval, KMS, and
certificate validation are explicit injected providers. Ed25519, asymmetric JWT, and
X.509 operations require optional cryptography support; no provider fallback is hidden.

::: phydrax.service.InProcessReferenceService

---

::: phydrax.service.SQLiteServiceStore

---

::: phydrax.service.ReleaseIndexDependencyAdmitter

---

::: phydrax.service.OIDCJWKSTokenValidator

---

::: phydrax.service.HTTPSJWKSProvider

---

::: phydrax.service.SlurmScheduler

---

::: phydrax.service.KubernetesScheduler

---

::: phydrax.service.LocalSecretHandleBroker

---

::: phydrax.service.Ed25519Signer

---

::: phydrax.service.KMSSigner

---

::: phydrax.service.SigningTrustStore

## Closure-data plane

Closure data keeps simulation ownership external. `FlowStateSchema`,
`ClosureSnapshot`, and `ClosureSeries` bind physical components, units, mesh, case,
trajectory, realization, and time identities. Filters and target constructors create
an immutable `ClosureAnalysisDAG`. `ChunkedClosureDatasetManifest` verifies complete,
non-overlapping sample/byte coverage and delegates storage only through
`ClosureArtifactRepository`.

`LeakageSafePartitionPlan` groups by declared case/trajectory/realization/time-block
identity. `TrainOnlyNormalizer` records the exact training assignments used for its
statistics. `LearnedClosureBindingPlan` binds predictor ABI, model artifact, component
ordering, normalizer provenance, and differentiability. Conservative-face deployment
uses the finite-volume closure owner. Spectral drift is explicitly dealiased,
projected, Hermitian constrained, and energy checked; invalid prediction produces zero
drift together with a `SpectralFallbackArtifact`, never a hidden fallback.

::: phydrax.closure_data.FlowStateSchema

---

::: phydrax.closure_data.ClosureSeries

---

::: phydrax.closure_data.FilterSpec

---

::: phydrax.closure_data.ClosureAnalysisDAG

---

::: phydrax.closure_data.ChunkedClosureDatasetManifest

---

::: phydrax.closure_data.LeakageSafePartitionPlan

---

::: phydrax.closure_data.TrainOnlyNormalizer

---

::: phydrax.closure_data.LearnedClosureBindingPlan

---

::: phydrax.closure_data.PreparedSpectralDriftHook

---

::: phydrax.closure_data.SpectralFallbackArtifact
