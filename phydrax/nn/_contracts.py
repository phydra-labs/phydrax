#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Regularity, precision, and randomness declarations of network families."""

from collections.abc import Callable, Iterable
from typing import Any

import equinox as eqx
import jax

from .._differentiation import DerivativeRegularity
from .._model import AbstractArrayModel
from .._model._array import native_parameter_precision, value_derivative_contract
from .._model._component import ModelExecutionContract, RandomnessContract
from .activations import activation_regularity


AFFINE = DerivativeRegularity.smooth(degree_bound=1)
SMOOTH = DerivativeRegularity.smooth()


def compose_regularity(
    *stages: DerivativeRegularity | None,
) -> DerivativeRegularity | None:
    """Regularity of applying `stages` in order; undeclared if any stage is."""
    if not stages or any(stage is None for stage in stages):
        return None
    result = stages[0]
    for stage in stages[1:]:
        result = result.compose(stage)
    return result


def sum_regularity(
    terms: Iterable[DerivativeRegularity | None], /
) -> DerivativeRegularity | None:
    """Regularity of a sum or juxtaposition of `terms`; undeclared if any is."""
    terms_ = tuple(terms)
    if not terms_ or any(term is None for term in terms_):
        return None
    result = terms_[0]
    for term in terms_[1:]:
        result = result.add(term)
    return result


def product_regularity(
    factors: Iterable[DerivativeRegularity | None], /
) -> DerivativeRegularity | None:
    """Regularity of an elementwise product of `factors`; undeclared if any is."""
    factors_ = tuple(factors)
    if not factors_ or any(factor is None for factor in factors_):
        return None
    result = factors_[0]
    for factor in factors_[1:]:
        result = result.multiply(factor)
    return result


def gradient_regularity(
    regularity: DerivativeRegularity | None, /
) -> DerivativeRegularity | None:
    """Regularity of the value gradient of a map of `regularity`.

    The gradient loses one continuity order and one polynomial degree; that of a
    `C^0` map with declared pieces jumps across the non-smooth locus. It is
    undeclared for discontinuous or piece-free `C^0` maps, whose distributional
    gradient has a singular part.
    """
    if regularity is None:
        return None
    continuity = regularity.continuity
    if continuity != "smooth":
        if continuity < 0 or (continuity == 0 and regularity.pieces == "none"):
            return None
        continuity -= 1
    degree = regularity.degree_bound
    return DerivativeRegularity(
        continuity=continuity,
        pieces=regularity.pieces,
        degree_bound=None if degree is None else max(degree - 1, 0),
        conditions=regularity.conditions,
        support=regularity.support,
    )


def activated_affine(activation: Callable[..., Any], /) -> DerivativeRegularity | None:
    """Regularity of an affine map followed by `activation`."""
    return compose_regularity(AFFINE, activation_regularity(activation))


def model_regularity(model: Any, /) -> DerivativeRegularity | None:
    """Declared value regularity of a child array model (`None` otherwise)."""
    if not isinstance(model, AbstractArrayModel):
        return None
    return model.model_execution_contract().regularity


def _dropout_layers() -> tuple[type, ...]:
    # Imported here: the dropout layer module imports this package's base model.
    from .layers._dropout import Dropout

    return (Dropout, eqx.nn.Dropout)


def network_randomness(tree: Any, /) -> RandomnessContract:
    """Deterministic randomness of a network, recording active dropout.

    Dropout with `p > 0` outside inference mode is stochastic; the network is
    then deterministic only in its inference state, which the owner must bind.
    """
    dropouts = _dropout_layers()
    active = any(
        isinstance(node, dropouts) and node.p > 0.0 and not node.inference
        for node in jax.tree_util.tree_leaves(
            tree, is_leaf=lambda node: isinstance(node, dropouts)
        )
    )
    return RandomnessContract("deterministic", requires_inference_state=active)


def network_execution_contract(
    model: AbstractArrayModel,
    regularity: DerivativeRegularity | None,
    /,
) -> ModelExecutionContract:
    """Execution contract of a network family with declared value `regularity`.

    Undeclared regularity keeps the conservative `AbstractArrayModel` default.
    Otherwise derivatives follow `value_derivative_contract(regularity)`,
    precision is native in the uniform parameter dtype (floors undeclared), and
    randomness is deterministic, conditioned on the inference state when the
    network holds active dropout.
    """
    if regularity is None:
        return AbstractArrayModel.model_execution_contract(model)
    return model._execution_contract(
        value_derivative_contract(regularity),
        precision=native_parameter_precision(model),
        randomness=network_randomness(model),
    )


__all__ = [
    "AFFINE",
    "SMOOTH",
    "activated_affine",
    "compose_regularity",
    "gradient_regularity",
    "model_regularity",
    "network_execution_contract",
    "network_randomness",
    "product_regularity",
    "sum_regularity",
]
