# Phydrax

Top-level package namespace. Most functionality lives in subpackages:

- `phydrax.domain`: domains, geometry, sampling, and domain functions
- `phydrax.metrix`: coordinate charts, tensors, Riemannian metrics, curvature,
  embedded geometry, and metric-aware stochastic calculus
- `phydrax.data_utils`: CSV loading and array scaling helpers
- `phydrax.operators`: differential/integral operators on `DomainFunction`s
- `phydrax.conditions`: residual, moment, observation, and physical condition declarations
- `phydrax.terms`: penalty terms and specialized numerical/data terms
- `phydrax.integration`: integration targets, sources, reductions, and realizations
- `phydrax.kernels`: composable positive-definite covariance functions shared by
  Gaussian-process and coreset algorithms
- `phydrax.special`: JAX-native named special functions and integral primitives
- `phydrax.enforcement`: exact condition transforms and enforcement programs
- `phydrax.optim`: structured residual KFAC and domain-neutral optimization
  configurations consumed by bounded geometry and posterior workflows
- `phydrax.nn`: neural network components and structured models
- `phydrax.sparse`: JAX-native sparse relations, routing, reductions, and linear actions
- `phydrax.solver`: functional solvers and direct ODE/SDE, jump, hybrid, and
  semidiscrete SPDE integration
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
dimensions. Dense and SciPy conversions are explicit interoperability
operations, not execution fallbacks. Sparse-grid quadrature, sparse Gaussian
process approximations, stochastic probes, and generic matrix-free callables
remain in their semantic subsystems.

::: phydrax.sparse.EdgeRelation

---

::: phydrax.sparse.RowRelation

---

::: phydrax.sparse.SparseLinearMap

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
sampling. `MassActionJumpProcess` provides combinatorial propensities and
stoichiometric updates without requiring SciPy reaction objects.
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

::: phydrax.stochastic.MassActionJumpProcess
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
