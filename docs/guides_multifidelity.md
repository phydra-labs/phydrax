# Multi-fidelity scientific machine learning

PhydraX represents fidelity as a graph of identified models, not as an integer input
feature. A workflow names one authoritative target, defines directed information or
refinement relations, and compares every model through one canonical observable
contract. Native solver states may still use different discretizations.

## Ownership

- `phydrax.fidelity` owns hierarchy, case, observation, split, and archive contracts.
- `phydrax.integration` owns control variates, coupled sampling, and MLMC allocation.
- `phydrax.uq` owns correlated fidelity models and information-per-cost acquisition.
- `phydrax.nn.operator` owns field-valued correction models and their training data.
- `phydrax.rom` exposes an executable reduced model as an ordinary fidelity evaluator.

A multiresolution neural architecture is not automatically a multi-fidelity workflow.
Fidelity requires separate model identities, costs, observations, and a declared target.

## Declare a hierarchy

```python
import phydrax as phx

coarse = phx.fidelity.FidelityLevelSpec(
    "coarse",
    problem_id="diffusion",
    observable_id="terminal-energy",
    model_id="coarse-solver",
    approximation_id="mesh-32",
    observable_contract_id="scalar-energy",
)
fine = phx.fidelity.FidelityLevelSpec(
    "fine",
    problem_id="diffusion",
    observable_id="terminal-energy",
    model_id="fine-solver",
    approximation_id="mesh-256",
    observable_contract_id="scalar-energy",
)
hierarchy = phx.fidelity.FidelityHierarchy(
    (coarse, fine),
    (phx.fidelity.FidelityRelation("coarse", "fine"),),
    target_level_id="fine",
)
path = hierarchy.linear_path()
```

The graph rejects cycles, disconnected models, target levels with outgoing relations,
and mismatched problem or observable contracts. `path` refuses ambiguous routes rather
than selecting one silently.

## Sparse heterogeneous observations

`FidelityDataset` stores physical cases separately from observed case/pair/level rows.
Missing high-fidelity observations need no imputation. Invalid evaluations remain in the
corpus with `valid=False`; fitting adapters ignore them explicitly.

`split_fidelity_dataset` partitions `split_group_id` values. Every observation of one
physical case therefore remains in the same train, validation, or test partition even
when several fidelities or stochastic realizations exist.

Archive restoration requires the expected hierarchy and PyTree template. The reader
checks every case, evaluation, and dataset content identity and never unpickles Python
objects.

## Coupled MLMC

`fidelity_multilevel_target` binds a prefix-stable input sampler and a level evaluator.
For correction level `l`, it evaluates only levels `l` and `l - 1` on the same sampled
input. Different correction levels receive independent random-key namespaces.

Choose the estimand explicitly:

- `estimand="finest_level"` estimates the expectation of the declared target model and
  has zero truncation bias relative to that discrete estimand.
- `estimand="limit"` requires either at least three stochastic refinement levels for an
  asymptotic bias estimate or an explicit `terminal_bias_bound`.

Unavailable spatial, temporal, solver, transfer, or model-form errors are never
silently folded into the statistical RMSE.

## Fidelity Gaussian processes

`AutoregressiveFidelityKernel` implements a positive-definite correction chain with one
independent spatial kernel per level and one transfer coefficient per relation.
`FidelityGaussianProcess` flattens active scalar observations into the existing
heterotopic multi-output GP substrate and always conditions an explicitly named level.
`condition_target` cannot return a cheaper channel as the target prediction.

Use existing ICM or LMC kernels instead when the sources have no justified
autoregressive ordering. Fidelity is a discrete output identity, never a continuous
coordinate with an invented distance.

## Information per cost

`TargetVarianceAcquisitionPolicy` declares target query points, weights, one positive
cost per level, and a common cost unit. `select_fidelity_acquisition` greedily chooses
input-level pairs by target posterior-variance reduction divided by cost, updating the
posterior covariance after every selected candidate. It does not optimize mixed-fidelity
objective values and is not mislabeled as Bayesian optimization.

## Physics-informed neural fields

Multi-fidelity PINNs use the same `FunctionalSolver`, residual penalties, exact
enforcement, collocation, balancing, and checkpoint machinery as ordinary PINNs.
`prepare_fidelity_observation_penalty` converts active rows at one level into an exact
fixed observation term; evaluation cost never becomes a loss weight.

The default child stage freezes a selected parent field and trains a correction against
target data and target physics. `prepare_fidelity_pinn_stage` accepts only the immediate
successor on a `FidelityPath`, records the relation and observation identities in the
functional discretization bundle, and exposes deterministic term roles. A nonlinear
correction may additionally consume the frozen parent prediction through
`condition_fidelity_correction`.

Selection terms may use only held-out target-fidelity groups. Final
`evaluate_fidelity_pinn` rejects any target test group observed in training or
validation. Composed fidelity neural fields currently use Optax or SOAP; KFAC refuses
until an explicit affine curvature layout exists. See the
[multi-fidelity PINN cookbook](cookbook/multifidelity_pinn.md).

## Field correction operators

`FidelityCorrectionOperator` composes a baseline operator and a correction operator:

```text
target prediction = baseline prediction + correction prediction
```

Both operators consume the same canonical `OperatorBatch` and must return the same
output contract and target shape. Existing `fit_operator` and `SupervisedOperatorLoss`
train the composed target prediction; no parallel optimizer or fidelity-specific loss is
introduced. Select only correction parameters with the existing parameter-subspace
control when the baseline must remain frozen.

`prepare_fidelity_operator_dataset` joins low and target datasets through an explicit
physical-case identity. It reports low-only and target-only cases and, by default,
refuses target cases without a low-fidelity pair.

## Reduced models

A ROM is one possible fidelity evaluator. `ROMFidelityEvaluator` accepts only profiles
with an executable reduced online solve, disables truth fallback, and retains the ROM
artifact and lifecycle evidence. Control-variate and MLMC algorithms remain in
`phydrax.integration`; they are not ROM training profiles.

## Validation rules

- Select models using held-out target-fidelity risk.
- Treat cheaper-level validation scores as diagnostics only.
- Keep observation noise, GP uncertainty, cross-fidelity discrepancy, discretization
  error, solver error, and sampling error separate.
- Reject changed hierarchy, transfer, target, model, or dataset identities on restore.
- Never promote a surrogate result to authoritative science without a separately
  declared target evaluation or qualification policy.
