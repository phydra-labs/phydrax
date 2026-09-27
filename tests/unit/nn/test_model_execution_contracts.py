from typing import Any

import equinox as eqx
import jax

from phydrax import (
    CapabilityEvidenceKind,
    ComponentAuthority,
    DerivativeRegularity,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
    RegularityPolicy,
)
from phydrax._model import FrozenModel
from phydrax.nn.activations import squared_relu
from phydrax.nn.layers import inference_mode
from phydrax.nn.models import EquinoxModel, InputConvexNetwork, MLP


INPUT = DerivativeSurface.INPUT
PARAMETER = DerivativeSurface.MODEL_PARAMETER
_C0_LINEAR = DerivativeRegularity.piecewise_polynomial(continuity=0, degree_bound=1)
_ALLOW_AE = RegularityPolicy(allow_almost_everywhere=True)


def _mlp(**overrides: Any) -> MLP:
    fields = dict(
        in_size=2,
        out_size=1,
        width_size=4,
        depth=2,
        key=jax.random.key(0),
    )
    fields.update(overrides)
    # ty: ignore[invalid-argument-type]
    return MLP(**fields)


def _admit(
    model: MLP,
    order: int,
    policy: RegularityPolicy = _ALLOW_AE,
) -> Any:
    request = DifferentiationRequest(
        (INPUT,),
        order=order,
        authority=ComponentAuthority.MODEL,
    )
    return model.model_execution_contract().derivative.admit(request, policy=policy)


def test_model_execution_contracts_scenario_1() -> None:
    smooth = _mlp()
    smooth_contract = smooth.model_execution_contract()
    assert smooth_contract.regularity == DerivativeRegularity.smooth()
    assert smooth_contract.derivative.level(INPUT) is GradientLevel.SMOOTH
    assert smooth_contract.derivative.level(PARAMETER) is GradientLevel.SMOOTH
    assert _admit(smooth, 3, policy=RegularityPolicy()).supported

    relu = _mlp(activation=jax.nn.relu)
    relu_contract = relu.model_execution_contract()
    assert relu_contract.regularity == _C0_LINEAR
    assert relu_contract.derivative.level(INPUT) is GradientLevel.ALMOST_EVERYWHERE
    assert _admit(relu, 1).level(INPUT) is GradientLevel.ALMOST_EVERYWHERE
    relu_second = _admit(relu, 2)
    assert not relu_second.supported
    assert relu_second.level(INPUT) is GradientLevel.NONE
    assert relu_second.reasons == ("regularity-degenerate",)

    piecewise_smooth = _mlp(
        activation=jax.nn.relu,
        final_activation=jax.nn.tanh,
    )
    assert piecewise_smooth.model_execution_contract().regularity == (
        DerivativeRegularity.piecewise_smooth(continuity=0)
    )
    second = _admit(piecewise_smooth, 2)
    assert second.supported
    assert second.level(INPUT) is GradientLevel.ALMOST_EVERYWHERE
    assert "singular-part-ignored" in second.conditions
    assert not _admit(piecewise_smooth, 2, policy=RegularityPolicy()).supported

    polynomial = _mlp(activation=squared_relu)
    assert polynomial.model_execution_contract().regularity == (
        DerivativeRegularity.piecewise_polynomial(continuity=1, degree_bound=4)
    )
    assert _admit(polynomial, 4).supported
    assert not _admit(polynomial, 5).supported
    assert (
        _mlp(
            activation=jax.nn.relu,
            skip_connection=True,
        )
        .model_execution_contract()
        .regularity
        == _C0_LINEAR
    )

    unknown = _mlp(activation=lambda value: value * value * value)
    unknown_contract = unknown.model_execution_contract()
    assert unknown_contract.regularity is None
    assert unknown_contract.randomness is None
    assert unknown_contract.derivative.level(INPUT) is GradientLevel.CONDITIONAL
    assert _admit(unknown, 1).reasons == ("regularity-undeclared",)
    deterministic = _mlp()
    contract = deterministic.model_execution_contract()
    dtype = deterministic.layers[0].weight.dtype.name
    assert contract.precision is not None
    assert contract.randomness is not None
    assert (contract.precision.parameter_dtype, contract.precision.compute_dtype) == (
        dtype,
        dtype,
    )
    assert contract.precision.residual_floor(1.0) is None
    assert contract.randomness.mode == "deterministic"
    assert not contract.randomness.requires_inference_state

    stochastic = _mlp(dropout=0.2)
    stochastic_randomness = stochastic.model_execution_contract().randomness
    assert stochastic_randomness is not None
    assert stochastic_randomness.requires_inference_state
    frozen_randomness = inference_mode(stochastic).model_execution_contract().randomness
    assert frozen_randomness is not None
    assert not frozen_randomness.requires_inference_state

    relu = _mlp(activation=jax.nn.relu)
    assert (
        FrozenModel(relu).model_execution_contract().contract_id
        == relu.model_execution_contract().contract_id
    )
    cases = (
        ("softplus", DerivativeRegularity.smooth()),
        ("relu", _C0_LINEAR),
        (
            "squared_relu",
            DerivativeRegularity.piecewise_polynomial(continuity=1, degree_bound=4),
        ),
    )
    for activation, regularity in cases:
        model = InputConvexNetwork(
            in_size=2,
            width_size=4,
            depth=2,
            activation=activation,
            key=jax.random.key(1),
        )
        contract = model.model_execution_contract()
        certificate = model.input_convex_certificate()
        assert contract.regularity == regularity, activation
        assert contract.certificates == (
            (
                "input-convex",
                certificate.certificate_id,
                CapabilityEvidenceKind.CONSTRUCTED,
            ),
        ), activation


def test_equinox_model_contracts_are_derived_structurally() -> None:
    relu = eqx.nn.MLP(
        2,
        1,
        4,
        2,
        activation=jax.nn.relu,
        key=jax.random.key(0),
    )
    assert (
        EquinoxModel(
            relu,
            in_size=2,
            out_size=1,
        )
        .model_execution_contract()
        .regularity
        == _C0_LINEAR
    )

    pieces = eqx.nn.MLP(
        2,
        1,
        4,
        2,
        activation=jax.nn.relu,
        final_activation=jax.nn.tanh,
        key=jax.random.key(0),
    )
    assert EquinoxModel(
        pieces,
        in_size=2,
        out_size=1,
    ).model_execution_contract().regularity == DerivativeRegularity.piecewise_smooth(
        continuity=0
    )

    unknown = eqx.nn.MLP(
        2,
        1,
        4,
        1,
        activation=lambda value: value * value * value,
        key=jax.random.key(0),
    )
    assert (
        EquinoxModel(
            unknown,
            in_size=2,
            out_size=1,
        )
        .model_execution_contract()
        .regularity
        is None
    )
    embedding = EquinoxModel(
        eqx.nn.Embedding(4, 2, key=jax.random.key(0)),
        in_size="scalar",
        out_size=2,
    )
    assert embedding.model_execution_contract().regularity is None

    sequential = eqx.nn.Sequential(
        [
            eqx.nn.Linear(2, 3, key=jax.random.key(0)),
            eqx.nn.Lambda(jax.nn.tanh),
            eqx.nn.Dropout(0.2),
            eqx.nn.Linear(3, 1, key=jax.random.key(1)),
        ]
    )
    sequential_contract = EquinoxModel(
        sequential,
        in_size=2,
        out_size=1,
    ).model_execution_contract()
    assert sequential_contract.regularity == DerivativeRegularity.smooth()
    assert sequential_contract.randomness is not None
    assert sequential_contract.randomness.requires_inference_state
