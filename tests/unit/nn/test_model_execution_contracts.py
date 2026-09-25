#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import pytest

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


def _mlp(**overrides):
    fields = dict(in_size=2, out_size=1, width_size=4, depth=2, key=jax.random.key(0))
    fields.update(overrides)
    return MLP(**fields)


def _admit(model, order, policy=_ALLOW_AE):
    request = DifferentiationRequest(
        (INPUT,), order=order, authority=ComponentAuthority.MODEL
    )
    return model.model_execution_contract().derivative.admit(request, policy=policy)


def test_tanh_mlp_is_smooth_to_every_order():
    contract = _mlp().model_execution_contract()
    assert contract.regularity == DerivativeRegularity.smooth()
    assert contract.derivative.level(INPUT) is GradientLevel.SMOOTH
    assert contract.derivative.level(PARAMETER) is GradientLevel.SMOOTH
    admission = _admit(_mlp(), 3, policy=RegularityPolicy())
    assert admission.supported
    assert admission.level(INPUT) is GradientLevel.SMOOTH


def test_relu_mlp_with_linear_output_rejects_second_derivatives():
    model = _mlp(activation=jax.nn.relu)
    contract = model.model_execution_contract()
    assert contract.regularity == _C0_LINEAR
    assert contract.derivative.level(INPUT) is GradientLevel.ALMOST_EVERYWHERE
    assert _admit(model, 1).level(INPUT) is GradientLevel.ALMOST_EVERYWHERE
    laplacian = _admit(model, 2)
    assert not laplacian.supported
    assert laplacian.level(INPUT) is GradientLevel.NONE
    assert laplacian.reasons == ("regularity-degenerate",)


def test_relu_mlp_with_tanh_output_has_smooth_pieces():
    model = _mlp(activation=jax.nn.relu, final_activation=jax.nn.tanh)
    assert model.model_execution_contract().regularity == (
        DerivativeRegularity.piecewise_smooth(continuity=0)
    )
    second = _admit(model, 2)
    assert second.supported
    assert second.level(INPUT) is GradientLevel.ALMOST_EVERYWHERE
    assert "singular-part-ignored" in second.conditions
    assert not _admit(model, 2, policy=RegularityPolicy()).supported


def test_depth_two_squared_relu_bounds_piece_degree_by_four():
    model = _mlp(activation=squared_relu)
    regularity = model.model_execution_contract().regularity
    assert regularity == DerivativeRegularity.piecewise_polynomial(
        continuity=1, degree_bound=4
    )
    assert _admit(model, 4).supported
    assert not _admit(model, 5).supported


def test_skip_connection_keeps_the_piecewise_linear_bound():
    model = _mlp(activation=jax.nn.relu, skip_connection=True)
    assert model.model_execution_contract().regularity == _C0_LINEAR


def test_unknown_activation_leaves_regularity_undeclared():
    contract = _mlp(activation=lambda x: x * x * x).model_execution_contract()
    assert contract.regularity is None
    assert contract.randomness is None
    assert contract.derivative.level(INPUT) is GradientLevel.CONDITIONAL
    rejected = _admit(_mlp(activation=lambda x: x * x * x), 1)
    assert rejected.reasons == ("regularity-undeclared",)


def test_network_precision_follows_parameter_dtype():
    precision = _mlp().model_execution_contract().precision
    dtype = _mlp().layers[0].weight.dtype.name
    assert (precision.parameter_dtype, precision.compute_dtype) == (dtype, dtype)
    assert precision.residual_floor(1.0) is None


def test_active_dropout_requires_the_inference_state():
    deterministic = _mlp().model_execution_contract().randomness
    assert deterministic.mode == "deterministic"
    assert not deterministic.requires_inference_state
    stochastic = _mlp(dropout=0.2)
    assert stochastic.model_execution_contract().randomness.requires_inference_state
    frozen = inference_mode(stochastic)
    assert not frozen.model_execution_contract().randomness.requires_inference_state


@pytest.mark.parametrize(
    ("activation", "regularity"),
    [
        ("softplus", DerivativeRegularity.smooth()),
        ("relu", _C0_LINEAR),
        (
            "squared_relu",
            DerivativeRegularity.piecewise_polynomial(continuity=1, degree_bound=4),
        ),
    ],
)
def test_input_convex_network_carries_its_certificate(activation, regularity):
    model = InputConvexNetwork(
        in_size=2, width_size=4, depth=2, activation=activation, key=jax.random.key(1)
    )
    contract = model.model_execution_contract()
    assert contract.regularity == regularity
    certificate = model.input_convex_certificate()
    assert contract.certificates == (
        (
            "input-convex",
            certificate.certificate_id,
            CapabilityEvidenceKind.CONSTRUCTED,
        ),
    )


def test_equinox_mlp_regularity_is_derived_structurally():
    module = eqx.nn.MLP(2, 1, 4, 2, activation=jax.nn.relu, key=jax.random.key(0))
    wrapped = EquinoxModel(module, in_size=2, out_size=1)
    assert wrapped.model_execution_contract().regularity == _C0_LINEAR
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
        pieces, in_size=2, out_size=1
    ).model_execution_contract().regularity == DerivativeRegularity.piecewise_smooth(
        continuity=0
    )


def test_equinox_unknown_layers_stay_undeclared():
    module = eqx.nn.MLP(2, 1, 4, 1, activation=lambda x: x * x * x, key=jax.random.key(0))
    wrapped = EquinoxModel(module, in_size=2, out_size=1)
    assert wrapped.model_execution_contract().regularity is None
    embedding = EquinoxModel(
        eqx.nn.Embedding(4, 2, key=jax.random.key(0)), in_size="scalar", out_size=2
    )
    assert embedding.model_execution_contract().regularity is None


def test_equinox_sequential_dropout_requires_the_inference_state():
    module = eqx.nn.Sequential(
        [
            eqx.nn.Linear(2, 3, key=jax.random.key(0)),
            eqx.nn.Lambda(jax.nn.tanh),
            eqx.nn.Dropout(0.2),
            eqx.nn.Linear(3, 1, key=jax.random.key(1)),
        ]
    )
    contract = EquinoxModel(module, in_size=2, out_size=1).model_execution_contract()
    assert contract.regularity == DerivativeRegularity.smooth()
    assert contract.randomness.requires_inference_state


def test_frozen_model_delegates_its_contract():
    model = _mlp(activation=jax.nn.relu)
    assert (
        FrozenModel(model).model_execution_contract().contract_id
        == model.model_execution_contract().contract_id
    )
