# Multi-fidelity physics-informed neural fields

PhydraX implements multi-fidelity PINNs as staged `FunctionalSolver` problems. A
selected parent field is frozen, transferred into the target support, and corrected
against target-fidelity data and physics. This is distinct from
`FidelityCorrectionOperator`, which composes function-to-function neural operators.

## Prepare leakage-safe data

Declare a fidelity hierarchy and store every observed level against one physical case.
Request target observations in every split used for training, selection, or final
evaluation.

```python
import jax
import optax

import phydrax as phx
```

```python
split = phx.fidelity.split_fidelity_dataset(
    dataset,
    train_fraction=0.6,
    validation_fraction=0.2,
    seed=11,
    requirements=phx.fidelity.FidelitySplitRequirements(
        train={"low": 2, "high": 2},
        validation={"low": 1, "high": 1},
        test={"high": 1},
    ),
)
```

Convert each partition into an exact fixed observation penalty. Case probability
weights and optional observation standard deviations determine statistical weights;
evaluation cost never changes the loss.

```python
geometry = phx.domain.Interval1d(-1.0, 1.0)
component = geometry.component()
low_train = phx.terms.prepare_fidelity_observation_penalty(
    split.train,
    "low",
    field="u",
    component=component,
)
low_validation = phx.terms.prepare_fidelity_observation_penalty(
    split.validation,
    "low",
    field="u",
    component=component,
)
```

The prepared target is indexed by the fixed physical `PointBatch`. It refuses another
batch rather than inventing an interpolation through sparse observations. Invalid
observations remain in `rejected_evaluation_ids`.

## Train and bind the low-fidelity field

```python
low_solver = phx.solver.FunctionalSolver(
    functions={"u": low_model},
    terms=(low_physics, low_train.term),
    evaluation_terms=(low_validation.term,),
).solve(
    num_iter=2_000,
    optim=optax.adam(1e-3),
    seed=4,
    log_every=100,
)
parent = phx.solver.bind_fidelity_pinn_level(
    dataset.hierarchy.linear_path(),
    "low",
    low_solver,
    training_observations=(low_train,),
    validation_observations=(low_validation,),
)
```

`parent.functions` contains the exactly enforced ansatz when the low solver has an
`EnforcementProgram`.

## Prepare the target correction

Target observations must come from the target level and disjoint physical groups.
Target physics sees the same composed field as target data.

```python
high_train = phx.terms.prepare_fidelity_observation_penalty(
    split.train,
    "high",
    field="u",
    component=component,
)
high_validation = phx.terms.prepare_fidelity_observation_penalty(
    split.validation,
    "high",
    field="u",
    component=component,
)
correction = geometry.Model("x")(
    phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(64, 64, 64),
        activation=jax.nn.tanh,
        key=jax.random.key(5),
    )
)
stage = phx.solver.prepare_fidelity_pinn_stage(
    parent,
    "high",
    {"u": correction},
    (target_pde, target_boundary),
    training_observations=(high_train,),
    validation_observations=(high_validation,),
    epsilon=0.1,
    enforcement=target_enforcement,
)
trained_stage = stage.training_solver.solve(
    num_iter=4_000,
    optim=optax.adam(1e-3),
    seed=6,
    log_every=100,
)
result = stage.finalize(trained_stage)
```

The parent evaluator and all of its derivative rules are nontrainable in the child
stage. `stage.term_roles` identifies target-physics and target-data term indices for
existing gradient-norm or NTK term balancing. The stage inserts its hierarchy, relation,
parent, transfer, scale, and data identity into the functional discretization bundle,
so ordinary functional checkpoints reject incompatible resumes.

Target-owned inverse parameters may be supplied through `replacement_functions`; they
remain trainable and are not aliased to frozen parent parameters.

## Condition a nonlinear correction on the parent

A correction can depend on both physical coordinates and the frozen parent value:

```python
conditioned_delta = phx.solver.condition_fidelity_correction(
    parent.functions["u"],
    correction_model_with_coordinate_and_parent_inputs,
    correction_id="high-nonlinear-correction",
)
```

Coordinate AD differentiates through the parent prediction while parameter updates
remain restricted to the correction model. This path currently supports ordinary
Optax/pointwise AD training. It is not advertised for KFAC curvature lowering.

## Evaluate target evidence once

```python
high_test = phx.terms.prepare_fidelity_observation_penalty(
    split.test,
    "high",
    field="u",
    component=component,
)
evidence = phx.solver.evaluate_fidelity_pinn(
    result,
    (high_test,),
    physics_terms=(target_pde,),
)
```

Evaluation rejects any test `split_group_id` seen during training or validation. The
result reports target RMSE, relative L2 error, accuracy, fixed target-physics losses,
and complete result/observation identities. A deterministic residual is not calibrated
uncertainty.

## Optimizer support

- Optax and SOAP use the frozen-parent correction normally.
- Existing functional selection, balancing, checkpointing, and exact enforcement remain
  available.
- KFAC currently rejects composed fidelity corrections because their trainable neural
  state is outside a declared affine curvature layout. The refusal occurs before an
  update; no diagonal or identity-curvature fallback is substituted.
