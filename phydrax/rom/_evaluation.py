#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._affine import (
    AffineLinearROMEvaluation,
    NamedROMObservation,
    PreparedAffineLinearROM,
)


class AffineLinearROMAudit(StrictModule, NonTrainableState):
    """Independent physical-norm and observable audit of one affine ROM result."""

    state_error_norm: Array
    state_relative_error: Array
    observation_errors: tuple[NamedROMObservation, ...]
    valid: Array
    model_id: str = eqx.field(static=True)


def _norm(space, vector, /) -> Array:
    value = space.validate(vector)
    return jnp.sqrt(jnp.maximum(jnp.real(space.inner(value, value)), 0.0))


def audit_affine_linear_rom(
    model: PreparedAffineLinearROM,
    evaluation: AffineLinearROMEvaluation,
    truth_state: PyTree[Array],
    /,
    *,
    truth_observations: Mapping[str, PyTree[Array]] | None = None,
) -> AffineLinearROMAudit:
    """Compare a reconstructed result to truth without changing model execution."""
    if not isinstance(model, PreparedAffineLinearROM):
        raise TypeError("model must be PreparedAffineLinearROM.")
    if not isinstance(evaluation, AffineLinearROMEvaluation):
        raise TypeError("evaluation must be AffineLinearROMEvaluation.")
    if evaluation.model_id != model.model_id:
        raise ValueError("Evaluation does not belong to the supplied ROM model.")
    if evaluation.reconstructed_state is None:
        raise ValueError("State audit requires evaluation with reconstruct=True.")
    space = model.reduction.trial.full_space
    truth = space.validate(truth_state)
    error = jax.tree.map(
        lambda approximate, reference: approximate - reference,
        evaluation.reconstructed_state,
        truth,
    )
    error_norm = _norm(space, error)
    truth_norm = _norm(space, truth)
    relative = error_norm / jnp.maximum(truth_norm, jnp.finfo(error_norm.dtype).tiny)
    expected = {} if truth_observations is None else dict(truth_observations)
    approximate = {item.name: item for item in evaluation.observations}
    if set(expected) != set(approximate):
        raise ValueError(
            "truth_observations must name exactly the evaluated ROM observations."
        )
    observation_errors = []
    prepared = {item.name: item for item in model.observations}
    for name in sorted(expected):
        output_space = prepared[name].output_space
        reference = output_space.validate(expected[name])
        difference = jax.tree.map(
            lambda left, right: left - right,
            approximate[name].value,
            reference,
        )
        observation_errors.append(
            NamedROMObservation(
                _norm(output_space, difference),
                name,
                approximate[name].observation_id,
            )
        )
    finite = jnp.isfinite(error_norm) & jnp.isfinite(relative)
    return AffineLinearROMAudit(
        error_norm,
        relative,
        tuple(observation_errors),
        evaluation.valid & finite,
        model.model_id,
    )


__all__ = ["AffineLinearROMAudit", "audit_affine_linear_rom"]
