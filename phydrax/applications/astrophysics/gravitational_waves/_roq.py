#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from phydrax.ein import contract

from ....rom import PreparedEmpiricalInterpolation
from ._approximation import (
    LikelihoodApproximationPolicy,
    LinearQuadraticCompressedLikelihood,
    QualifiedGravitationalWaveLikelihood,
    qualify_likelihood,
)
from ._likelihood import GravitationalWaveLikelihoodPlan


def prepare_reduced_order_quadrature_likelihood(
    exact: GravitationalWaveLikelihoodPlan,
    linear_interpolation: PreparedEmpiricalInterpolation,
    quadratic_interpolation: PreparedEmpiricalInterpolation,
    validation_parameters: Sequence[PyTree[Any]],
    validation_ids: Sequence[str],
    /,
    *,
    policy: LikelihoodApproximationPolicy | None = None,
) -> QualifiedGravitationalWaveLikelihood:
    if not isinstance(exact, GravitationalWaveLikelihoodPlan):
        raise TypeError("exact must be GravitationalWaveLikelihoodPlan.")
    if not isinstance(
        linear_interpolation, PreparedEmpiricalInterpolation
    ) or not isinstance(quadratic_interpolation, PreparedEmpiricalInterpolation):
        raise TypeError(
            "ROQ requires prepared linear and quadratic empirical interpolation."
        )
    frequency_count = int(exact.network.frequency.size)
    if (
        linear_interpolation.reconstruction_matrix.shape[0] != frequency_count
        or quadratic_interpolation.reconstruction_matrix.shape[0] != frequency_count
    ):
        raise ValueError(
            "ROQ empirical bases must use the exact likelihood frequency grid."
        )
    linear_order = np.argsort(np.asarray(linear_interpolation.node_indices))
    quadratic_order = np.argsort(np.asarray(quadratic_interpolation.node_indices))
    linear_indices = linear_interpolation.node_indices[jnp.asarray(linear_order)]
    quadratic_indices = quadratic_interpolation.node_indices[jnp.asarray(quadratic_order)]
    linear_reconstruction = linear_interpolation.reconstruction_matrix[
        :, jnp.asarray(linear_order)
    ]
    quadratic_reconstruction = quadratic_interpolation.reconstruction_matrix[
        :, jnp.asarray(quadratic_order)
    ]
    inverse_variance = exact.network.inverse_variance
    linear_functional = 2.0 * jnp.conj(exact.network.strain) * inverse_variance
    quadratic_functional = 2.0 * inverse_variance
    linear_weights = contract("df,fn->dn", linear_functional, linear_reconstruction)
    quadratic_weights = contract(
        "df,fn->dn", quadratic_functional, quadratic_reconstruction
    )
    candidate = LinearQuadraticCompressedLikelihood(
        exact,
        exact.network.frequency[linear_indices],
        exact.network.frequency[quadratic_indices],
        linear_weights,
        quadratic_weights,
        approximation_id=(
            "reduced-order-quadrature:"
            f"{linear_interpolation.artifact_id}:{quadratic_interpolation.artifact_id}"
        ),
    )
    return qualify_likelihood(
        exact,
        candidate,
        validation_parameters,
        validation_ids,
        policy=policy,
    )


__all__ = ["prepare_reduced_order_quadrature_likelihood"]
