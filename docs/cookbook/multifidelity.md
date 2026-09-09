# Multi-fidelity target prediction

This recipe combines non-nested scalar observations from a cheap approximation and an
expensive target model, conditions an autoregressive GP at the target fidelity, and
selects the next evaluation by target information per cost.

```python
import jax.numpy as jnp
import phydrax as phx

low = phx.fidelity.FidelityLevelSpec(
    "low",
    problem_id="response",
    observable_id="qoi",
    model_id="coarse-solver",
    approximation_id="coarse",
    observable_contract_id="scalar-qoi",
)
high = phx.fidelity.FidelityLevelSpec(
    "high",
    problem_id="response",
    observable_id="qoi",
    model_id="fine-solver",
    approximation_id="fine",
    observable_contract_id="scalar-qoi",
)
hierarchy = phx.fidelity.FidelityHierarchy(
    (low, high),
    (phx.fidelity.FidelityRelation("low", "high"),),
    target_level_id="high",
)
path = hierarchy.linear_path()

points = jnp.linspace(-1.0, 1.0, 9)
cases = tuple(
    phx.fidelity.FidelityCaseSpec(
        jnp.asarray([x]),
        case_id=f"case-{index}",
        split_group_id=f"physical-{index}",
    )
    for index, x in enumerate(points)
)


def low_model(x):
    return 0.9 * jnp.sin(2.0 * jnp.pi * x) + 0.1


def target_model(x):
    return jnp.sin(2.0 * jnp.pi * x)


evaluations = []
for index, x in enumerate(points):
    evaluations.append(
        phx.fidelity.FidelityEvaluation(
            low_model(x),
            case_id=f"case-{index}",
            pair_id=f"case-{index}",
            level_id="low",
            evaluator_id="response-models",
            valid=True,
            cost=1.0,
            cost_unit="relative",
        )
    )
    if index in (1, 4, 7):
        evaluations.append(
            phx.fidelity.FidelityEvaluation(
                target_model(x),
                case_id=f"case-{index}",
                pair_id=f"case-{index}",
                level_id="high",
                evaluator_id="response-models",
                valid=True,
                cost=20.0,
                cost_unit="relative",
            )
        )

dataset = phx.fidelity.FidelityDataset(
    hierarchy,
    cases,
    tuple(evaluations),
)
model = phx.uq.FidelityGaussianProcess(path, dataset)
kernel = phx.uq.AutoregressiveFidelityKernel(
    path,
    (
        phx.kernels.SquaredExponentialKernel(length_scale=0.3),
        phx.kernels.AmplitudeKernel(
            phx.kernels.SquaredExponentialKernel(length_scale=0.3),
            0.2,
        ),
    ),
    transfer_coefficients=jnp.asarray([1.0]),
)
state = phx.uq.MultiOutputGaussianProcessLikelihoodState(
    kernel=kernel,
    noise_scale=jnp.asarray([0.01, 0.01]),
)

query = jnp.linspace(-0.95, 0.95, 65)[:, None]
target_prediction = model.condition_target(query, state=state)

candidate_x = jnp.asarray([[-0.5], [-0.5], [0.5], [0.5]])
selection = phx.uq.select_fidelity_acquisition(
    model,
    state,
    candidate_x,
    ("low", "high", "low", "high"),
    phx.uq.TargetVarianceAcquisitionPolicy(
        path,
        query,
        jnp.asarray([1.0, 20.0]),
        batch_size=2,
    ),
)
```

`target_prediction` contains only target-fidelity query rows. `selection` reports the
selected candidate indices, level IDs, marginal target-variance reductions, scores, and
initial/final integrated target variance. Evaluate those pairs with the declared model,
extend the immutable fidelity dataset, and refit before the next acquisition round.

For field-valued outputs, train a low-fidelity operator first and wrap it with
`FidelityCorrectionOperator`. Prepare target cases through
`prepare_fidelity_operator_dataset` so unpaired or leaked cases remain explicit.
