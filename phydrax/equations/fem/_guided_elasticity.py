#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

import phydrax.ein as ein

from ...linalg import OperatorProperties
from .._finite_element_variational import CellResidualAction, FiniteElementForm
from ._operators import symmetric_gradient


def guided_elasticity_form(
    field_name: str,
    lame_lambda: ArrayLike,
    shear_modulus: ArrayLike,
    axial_wavenumber: float,
    /,
    *,
    form_id: str = "guided-elasticity",
) -> FiniteElementForm:
    """Build the fixed-q cross-section form for three-dimensional elasticity.

    The trial field is a vector displacement on a cross-section. Its omitted
    longitudinal derivative is exactly ``i q u`` under the ``exp(i q z)``
    convention. The test derivative is conjugated, hence the corresponding
    weak term uses ``-i q``. No plane-strain reduction is made.
    """
    name = str(field_name)
    identifier = str(form_id)
    if not name or not identifier:
        raise ValueError("field_name and form_id must be non-empty.")
    lambda_host = np.asarray(lame_lambda)
    mu_host = np.asarray(shear_modulus)
    q = float(axial_wavenumber)
    if (
        np.iscomplexobj(lambda_host)
        or np.iscomplexobj(mu_host)
        or np.any(~np.isfinite(lambda_host))
        or np.any(~np.isfinite(mu_host))
    ):
        raise ValueError("Guided elastic Lamé coefficients must be finite and real.")
    if np.any(mu_host <= 0.0) or np.any(lambda_host + 2.0 * mu_host / 3.0 <= 0.0):
        raise ValueError("Guided elastic Lamé coefficients must define positive energy.")
    if not math.isfinite(q):
        raise ValueError("axial_wavenumber must be finite.")
    lambda_ = jnp.asarray(lambda_host)
    mu = jnp.asarray(mu_host)

    def kernel(values, gradients, points, weights, test_basis, test_gradients, context):
        del points, context
        displacement = values[0]
        transverse_gradient = gradients[0]
        transverse_dimension = transverse_gradient.shape[-1]
        displacement_dimension = displacement.shape[-1]
        if displacement_dimension != transverse_dimension + 1:
            raise ValueError(
                "Guided elasticity requires one longitudinal displacement component "
                "in addition to the cross-section dimensions."
            )
        longitudinal_gradient = (1j * q * displacement)[..., None]
        full_gradient = jnp.concatenate(
            (transverse_gradient, longitudinal_gradient), axis=-1
        )
        strain = symmetric_gradient(full_gradient)
        trace = jnp.trace(strain, axis1=-2, axis2=-1)
        identity = jnp.eye(displacement_dimension, dtype=strain.dtype)
        stress = (lambda_ * trace)[..., None, None] * identity + 2.0 * mu[
            ..., None, None
        ] * strain
        transverse_action = ein.contract(
            "cq,cqid,cqad->cia",
            weights,
            test_gradients,
            stress[..., :transverse_dimension],
            backend="jax",
        )
        longitudinal_action = (
            -1j
            * q
            * ein.contract(
                "cq,cqi,cqa->cia",
                weights,
                test_basis,
                stress[..., transverse_dimension],
                backend="jax",
            )
        )
        return transverse_action + longitudinal_action

    return FiniteElementForm(
        identifier,
        name,
        (
            CellResidualAction(
                name,
                (name,),
                kernel,
                action_id="fixed-q-guided-elasticity",
            ),
        ),
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )


__all__ = ["guided_elasticity_form"]
