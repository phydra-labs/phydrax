#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Matrix-valued exterior products and finite Chern–Weil character forms."""

from __future__ import annotations

from math import factorial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..exterior._algebra import wedge
from ..exterior._basis import exterior_indices
from ..exterior._form_type import FormType


class ChernCharacterForm(StrictModule):
    """Finite pointwise Chern-character form in one explicit curvature convention."""

    coefficients: Array
    order: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    coordinate_dimension: int = eqx.field(static=True)
    bundle_rank: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    reality_residual: Array
    finite: Array
    form_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def chern_character_form(
    curvature: ArrayLike,
    order: int,
    coordinate_dimension: int,
    /,
    *,
    source_id: str,
) -> ChernCharacterForm:
    """Compute ``Tr[(i F / 2π)^k] / k!`` from curvature two-form coefficients."""
    values = jnp.asarray(curvature)
    dimension = int(coordinate_dimension)
    k = int(order)
    source = str(source_id)
    curvature_components = len(exterior_indices(dimension, 2))
    if k < 1 or 2 * k > dimension:
        raise ValueError("Chern-character order must be positive and fit the dimension.")
    if (
        values.ndim < 3
        or values.shape[-3] != curvature_components
        or values.shape[-2] != values.shape[-1]
    ):
        raise ValueError("Curvature must have exterior-two-form and square matrix axes.")
    if not source:
        raise ValueError("source_id must be non-empty.")
    power = values
    degree = 2
    for _ in range(1, k):
        fiber_shape = values.shape[-2:]
        power = wedge(
            power,
            values,
            FormType(dimension, degree, fiber_shape=fiber_shape),
            FormType(dimension, 2, fiber_shape=fiber_shape),
            product="matrix",
        )
        degree += 2
    traced = jnp.trace(power, axis1=-2, axis2=-1)
    coefficient = (1.0j / (2.0 * np.pi)) ** k / float(factorial(k))
    result = coefficient * traced
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(result)))
    reality = jnp.max(jnp.abs(jnp.imag(result))) / scale
    finite = jnp.all(jnp.isfinite(result))
    convention = "curvature=connection-square;ch_k=Tr[(iF/2pi)^k]/k!"
    identifier = canonical_fingerprint(
        {
            "kind": "chern-character-form",
            "curvature": array_tree_fingerprint(values),
            "order": k,
            "coordinate_dimension": dimension,
            "bundle_rank": values.shape[-1],
            "source_id": source,
            "convention": convention,
        }
    )
    return ChernCharacterForm(
        coefficients=result,
        order=k,
        degree=2 * k,
        coordinate_dimension=dimension,
        bundle_rank=values.shape[-1],
        source_id=source,
        convention=convention,
        reality_residual=reality,
        finite=finite,
        form_id=identifier,
        claim="finite-pointwise-chern-weil-form-requires-separate-closedness-and-topology-audit",
    )


class CharacteristicNumberEvidence(StrictModule):
    normalized_value: Array
    physical_value: Array
    effective_sample_size: Array
    imaginary_residual: Array
    finite: Array
    accepted: Array
    form_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def integrate_top_characteristic_form(
    form: ChernCharacterForm,
    normalized_weights: ArrayLike,
    physical_mass: ArrayLike,
    /,
    *,
    measure_id: str,
    imaginary_tolerance: float = 1e-8,
) -> CharacteristicNumberEvidence:
    """Integrate a sampled top form against one explicit normalized measure."""
    if not isinstance(form, ChernCharacterForm):
        raise TypeError("form must be ChernCharacterForm.")
    if form.degree != form.coordinate_dimension:
        raise ValueError("A characteristic number requires a top-degree form.")
    weights = jnp.asarray(normalized_weights)
    values = form.coefficients
    if values.shape[-1] != 1:
        raise ValueError("A top exterior form must have exactly one basis coefficient.")
    scalar = values[..., 0]
    if weights.shape != scalar.shape[:1]:
        raise ValueError("normalized_weights must match the sample axis.")
    if not measure_id:
        raise ValueError("measure_id must be non-empty.")
    normalized = jnp.sum(weights * scalar)
    physical = jnp.asarray(physical_mass) * normalized
    ess = 1.0 / jnp.sum(weights**2)
    imaginary = jnp.abs(jnp.imag(normalized)) / jnp.maximum(1.0, jnp.abs(normalized))
    finite = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(weights))
        & jnp.isfinite(physical)
        & jnp.isfinite(ess)
    )
    accepted = finite & (imaginary <= float(imaginary_tolerance))
    return CharacteristicNumberEvidence(
        normalized_value=normalized,
        physical_value=physical,
        effective_sample_size=ess,
        imaginary_residual=imaginary,
        finite=finite,
        accepted=accepted,
        form_id=form.form_id,
        measure_id=str(measure_id),
        claim="sampled-chern-weil-estimate-not-an-exact-topological-invariant",
    )


__all__ = [
    "CharacteristicNumberEvidence",
    "ChernCharacterForm",
    "chern_character_form",
    "integrate_top_characteristic_form",
]
