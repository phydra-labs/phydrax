#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...solver.maxwell.fourier_modal import FrequencyMaxwellMaterial
from ._refractive_index import (
    AbstractRefractiveIndexLaw,
    evaluate_refractive_index,
    RefractiveIndexEvaluation,
)


class GeometricRefractiveIndex(StrictModule):
    """Real geometric-optics lowering with lane-wise rejection evidence.

    Status values are 0 for accepted, 1 when the source evaluation was
    rejected, 2 when a nonzero imaginary part was present, and 3 when the real
    index was nonpositive or non-finite. Rejected values are NaN rather than a
    silently projected real part.
    """

    refractive_index: Array
    accepted: Array
    status: Array
    imaginary_magnitude: Array
    evaluation: RefractiveIndexEvaluation
    loss_tolerance: Array
    law_id: str = eqx.field(static=True)


def lower_to_geometric_index(
    law: AbstractRefractiveIndexLaw,
    angular_frequency: ArrayLike,
    /,
    *,
    loss_tolerance: ArrayLike = 0.0,
) -> GeometricRefractiveIndex:
    """Lower a scalar law only where it is finite, positive, and real."""
    tolerance = jnp.asarray(loss_tolerance)
    if tolerance.ndim != 0 or not jnp.issubdtype(tolerance.dtype, jnp.floating):
        raise TypeError("loss_tolerance must be a real scalar.")
    tolerance = eqx.error_if(
        tolerance,
        (~jnp.isfinite(tolerance)) | (tolerance < 0),
        "loss_tolerance must be finite and nonnegative.",
    )
    evaluation = evaluate_refractive_index(law, angular_frequency)
    real_index = jnp.real(evaluation.refractive_index)
    imaginary_magnitude = jnp.abs(jnp.imag(evaluation.refractive_index))
    real_positive = jnp.isfinite(real_index) & (real_index > 0)
    real_only = imaginary_magnitude <= tolerance
    accepted = evaluation.accepted & real_only & real_positive
    value = jnp.where(accepted, real_index, jnp.asarray(jnp.nan, real_index.dtype))
    status = jnp.where(
        ~evaluation.accepted,
        1,
        jnp.where(~real_only, 2, jnp.where(~real_positive, 3, 0)),
    ).astype(jnp.int32)
    return GeometricRefractiveIndex(
        refractive_index=value,
        accepted=accepted,
        status=status,
        imaginary_magnitude=imaginary_magnitude,
        evaluation=evaluation,
        loss_tolerance=tolerance,
        law_id=law.law_id,
    )


def lower_to_frequency_maxwell_material(
    law: AbstractRefractiveIndexLaw,
    angular_frequency: ArrayLike,
    /,
    *,
    material_id: str,
) -> FrequencyMaxwellMaterial:
    """Lower one resolved scalar index to isotropic nonmagnetic Maxwell data.

    This structural lowering accepts one scalar angular frequency and refuses
    invalid evaluations. Relative permittivity is exactly n**2, relative
    permeability is one, and both magnetoelectric blocks are zero.
    """
    evaluation = evaluate_refractive_index(law, angular_frequency)
    if evaluation.angular_frequency.ndim != 0:
        raise ValueError("Maxwell material lowering requires one scalar frequency.")
    if not bool(evaluation.accepted):
        raise ValueError(
            "Cannot lower a rejected refractive-index evaluation to Maxwell data."
        )
    index = evaluation.refractive_index
    relative_permittivity = index**2
    passive = bool(jnp.imag(relative_permittivity) >= 0)
    return FrequencyMaxwellMaterial(
        relative_permittivity,
        1.0,
        magnetoelectric_xi=0.0,
        magnetoelectric_zeta=0.0,
        material_id=material_id,
        material_role="physical",
        origin_evidence_id=law.provenance.provenance_id,
        passive=passive,
        reciprocal=True,
    )


__all__ = [
    "GeometricRefractiveIndex",
    "lower_to_frequency_maxwell_material",
    "lower_to_geometric_index",
]
