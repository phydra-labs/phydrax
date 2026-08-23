#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._curvature import ricci_tensor
from ._forms import DifferentialForm, exterior_derivative, wedge
from ._kahler import KahlerStructure, validate_kahler_structure


class _ConjugateFormCoefficients(StrictModule):
    form: DifferentialForm

    def __init__(self, form: DifferentialForm, /):
        self.form = form

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.conj(self.form._coefficients_point(coordinates))


class LocalCalabiYauStructure(StrictModule):
    """Local Ricci-flat Kähler candidate with a complex volume form."""

    kahler: KahlerStructure
    holomorphic_volume: DifferentialForm

    def __init__(
        self,
        kahler: KahlerStructure,
        holomorphic_volume: DifferentialForm,
        /,
    ):
        if not isinstance(kahler, KahlerStructure):
            raise TypeError("LocalCalabiYauStructure requires a KahlerStructure.")
        if not isinstance(holomorphic_volume, DifferentialForm):
            raise TypeError("holomorphic_volume must be a DifferentialForm.")
        if not kahler.chart.compatible_with(holomorphic_volume.chart):
            raise ValueError("Kähler and volume-form charts must match.")
        complex_dimension = kahler.chart.dimension // 2
        if holomorphic_volume.degree != complex_dimension:
            raise ValueError(
                "Holomorphic volume form degree must equal the complex dimension."
            )
        self.kahler = kahler
        self.holomorphic_volume = holomorphic_volume

    @property
    def complex_dimension(self) -> int:
        return self.kahler.chart.dimension // 2

    def conjugate_volume(self) -> DifferentialForm:
        return DifferentialForm(
            _ConjugateFormCoefficients(self.holomorphic_volume),
            chart=self.kahler.chart,
            degree=self.holomorphic_volume.degree,
        )

    def kahler_volume_form(self) -> DifferentialForm:
        result = self.kahler.fundamental_form()
        for _ in range(1, self.complex_dimension):
            result = wedge(result, self.kahler.fundamental_form())
        return result


class LocalCalabiYauValidationReport(StrictModule):
    valid: Array
    kahler_valid: Array
    volume_closed: Array
    volume_nonvanishing: Array
    maximum_closure_residual: Array
    minimum_volume_norm: Array
    maximum_compatibility_residual: Array
    maximum_volume_normalization_residual: Array
    maximum_ricci_residual: Array

    def __init__(
        self,
        *,
        valid: ArrayLike,
        kahler_valid: ArrayLike,
        volume_closed: ArrayLike,
        volume_nonvanishing: ArrayLike,
        maximum_closure_residual: ArrayLike,
        minimum_volume_norm: ArrayLike,
        maximum_compatibility_residual: ArrayLike,
        maximum_volume_normalization_residual: ArrayLike,
        maximum_ricci_residual: ArrayLike,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.kahler_valid = jnp.asarray(kahler_valid, dtype=bool)
        self.volume_closed = jnp.asarray(volume_closed, dtype=bool)
        self.volume_nonvanishing = jnp.asarray(volume_nonvanishing, dtype=bool)
        self.maximum_closure_residual = jnp.asarray(maximum_closure_residual)
        self.minimum_volume_norm = jnp.asarray(minimum_volume_norm)
        self.maximum_compatibility_residual = jnp.asarray(maximum_compatibility_residual)
        self.maximum_volume_normalization_residual = jnp.asarray(
            maximum_volume_normalization_residual
        )
        self.maximum_ricci_residual = jnp.asarray(maximum_ricci_residual)


def validate_local_calabi_yau_structure(
    structure: LocalCalabiYauStructure,
    points: ArrayLike,
    /,
    *,
    closure_tolerance: float = 1e-8,
    nonvanishing_tolerance: float = 1e-10,
    compatibility_tolerance: float = 1e-8,
    normalization_tolerance: float = 1e-7,
    ricci_tolerance: float = 1e-7,
    raise_on_error: bool = True,
) -> LocalCalabiYauValidationReport:
    """Validate local SU(n)-structure and Ricci-flat residuals."""
    if not isinstance(structure, LocalCalabiYauStructure):
        raise TypeError("structure must be a LocalCalabiYauStructure.")
    if (
        min(
            closure_tolerance,
            nonvanishing_tolerance,
            compatibility_tolerance,
            normalization_tolerance,
            ricci_tolerance,
        )
        < 0.0
    ):
        raise ValueError("Local Calabi–Yau tolerances must be non-negative.")
    kahler_report = validate_kahler_structure(
        structure.kahler,
        points,
        raise_on_error=False,
    )
    volume_values = structure.holomorphic_volume(points)
    volume_norms = jnp.linalg.norm(volume_values, axis=-1)
    minimum_volume_norm = jnp.min(volume_norms)
    derivative = exterior_derivative(structure.holomorphic_volume)(points)
    closure_residual = jnp.max(jnp.abs(derivative))
    closed = closure_residual <= closure_tolerance
    nonvanishing = minimum_volume_norm > nonvanishing_tolerance

    if structure.complex_dimension == 1:
        compatibility_residual = jnp.asarray(0.0)
    else:
        compatibility = wedge(
            structure.kahler.fundamental_form(), structure.holomorphic_volume
        )
        compatibility_residual = jnp.max(jnp.abs(compatibility(points)))

    complex_volume = wedge(
        structure.holomorphic_volume,
        structure.conjugate_volume(),
    )(points)
    kahler_volume = structure.kahler_volume_form()(points) / float(
        factorial(structure.complex_dimension)
    )
    expected_ratio = float(2**structure.complex_dimension)
    denominator = jnp.maximum(
        jnp.abs(kahler_volume),
        jnp.finfo(kahler_volume.real.dtype).tiny,
    )
    ratio = jnp.abs(complex_volume) / denominator
    normalization_residual = jnp.max(jnp.abs(ratio - expected_ratio))
    ricci_residual = jnp.max(jnp.abs(ricci_tensor(structure.kahler.metric, points)))
    valid = (
        kahler_report.valid
        & closed
        & nonvanishing
        & (compatibility_residual <= compatibility_tolerance)
        & (normalization_residual <= normalization_tolerance)
        & (ricci_residual <= ricci_tolerance)
    )
    report = LocalCalabiYauValidationReport(
        valid=valid,
        kahler_valid=kahler_report.valid,
        volume_closed=closed,
        volume_nonvanishing=nonvanishing,
        maximum_closure_residual=closure_residual,
        minimum_volume_norm=minimum_volume_norm,
        maximum_compatibility_residual=compatibility_residual,
        maximum_volume_normalization_residual=normalization_residual,
        maximum_ricci_residual=ricci_residual,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Local Calabi–Yau validation failed: "
            f"closure={float(jax.device_get(closure_residual))}, "
            f"compatibility={float(jax.device_get(compatibility_residual))}, "
            f"normalization={float(jax.device_get(normalization_residual))}, "
            f"ricci={float(jax.device_get(ricci_residual))}."
        )
    return report


__all__ = [
    "LocalCalabiYauStructure",
    "LocalCalabiYauValidationReport",
    "validate_local_calabi_yau_structure",
]
