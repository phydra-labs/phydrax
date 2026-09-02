#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import permutations
from math import factorial, isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._chart import CoordinateChart
from ._curvature import ricci_tensor
from ._exterior_basis import exterior_indices
from ._forms import DifferentialForm, exterior_derivative, hodge_star, wedge
from ._kahler import KahlerStructure, validate_kahler_structure
from ._metric import euclidean_metric, RiemannianMetric
from .algebra import AlgebraProductPlan, OctonionAlgebraSpec


class _ConjugateFormCoefficients(StrictModule):
    form: DifferentialForm

    def __init__(self, form: DifferentialForm, /):
        self.form = form

    def __call__(self, coordinates: Array, /) -> Array:
        return jnp.conj(self.form._coefficients_point(coordinates))


class LocalSUNStructure(StrictModule):
    """Local Ricci-flat Kähler candidate with a complex volume form."""

    kahler: KahlerStructure
    holomorphic_volume: DifferentialForm
    volume_bidegree: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        kahler: KahlerStructure,
        holomorphic_volume: DifferentialForm,
        /,
        *,
        volume_bidegree: tuple[int, int],
    ):
        if not isinstance(kahler, KahlerStructure):
            raise TypeError("LocalSUNStructure requires a KahlerStructure.")
        if not isinstance(holomorphic_volume, DifferentialForm):
            raise TypeError("holomorphic_volume must be a DifferentialForm.")
        if not kahler.chart.compatible_with(holomorphic_volume.chart):
            raise ValueError("Kähler and volume-form charts must match.")
        complex_dimension = kahler.chart.dimension // 2
        if holomorphic_volume.degree != complex_dimension:
            raise ValueError(
                "Holomorphic volume form degree must equal the complex dimension."
            )
        bidegree = (int(volume_bidegree[0]), int(volume_bidegree[1]))
        if bidegree != (complex_dimension, 0):
            raise ValueError("Local SU(n) volume form must declare bidegree (n, 0).")
        self.kahler = kahler
        self.holomorphic_volume = holomorphic_volume
        self.volume_bidegree = bidegree

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


class LocalSUNValidationReport(StrictModule):
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


def validate_local_su_structure(
    structure: LocalSUNStructure,
    points: ArrayLike,
    /,
    *,
    closure_tolerance: float = 1e-8,
    nonvanishing_tolerance: float = 1e-10,
    compatibility_tolerance: float = 1e-8,
    normalization_tolerance: float = 1e-7,
    ricci_tolerance: float = 1e-7,
    raise_on_error: bool = True,
) -> LocalSUNValidationReport:
    """Validate local SU(n)-structure and Ricci-flat residuals."""
    if not isinstance(structure, LocalSUNStructure):
        raise TypeError("structure must be a LocalSUNStructure.")
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
        raise ValueError("Local SU(n) tolerances must be non-negative.")
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
    report = LocalSUNValidationReport(
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
            "Local SU(n) validation failed: "
            f"closure={float(jax.device_get(closure_residual))}, "
            f"compatibility={float(jax.device_get(compatibility_residual))}, "
            f"normalization={float(jax.device_get(normalization_residual))}, "
            f"ricci={float(jax.device_get(ricci_residual))}."
        )
    return report


def _permutation_sign(values: tuple[int, ...], /) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def _dense_three_form(coefficients: Array, /) -> Array:
    indices = exterior_indices(7, 3)
    source_positions: list[int] = []
    output_positions: list[int] = []
    signs: list[int] = []
    for source, axes in enumerate(indices):
        for ordered in permutations(axes):
            source_positions.append(source)
            output_positions.append(ordered[0] * 49 + ordered[1] * 7 + ordered[2])
            signs.append(_permutation_sign(ordered))
    source_array = jnp.asarray(source_positions, dtype=jnp.int32)
    output_array = jnp.asarray(output_positions, dtype=jnp.int32)
    sign_array = jnp.asarray(signs, dtype=coefficients.dtype)
    values = coefficients[..., source_array] * sign_array
    flat = (
        jnp.zeros(coefficients.shape[:-1] + (7**3,), dtype=coefficients.dtype)
        .at[..., output_array]
        .set(values)
    )
    return flat.reshape(coefficients.shape[:-1] + (7, 7, 7))


def _maximum_or_zero(value: Array, /) -> Array:
    if value.size == 0:
        return jnp.asarray(0.0, dtype=value.dtype)
    return jnp.max(jnp.abs(value))


class _ConstantG2FormCoefficients(StrictModule):
    coefficients: Array

    def __init__(self, coefficients: Array, /):
        self.coefficients = jnp.asarray(coefficients)

    def __call__(self, coordinates: Array, /) -> Array:
        return self.coefficients.astype(coordinates.dtype)


class LocalG2Structure(StrictModule):
    """Local seven-dimensional G2 candidate with an explicit metric and three-form."""

    metric: RiemannianMetric
    associative_form: DifferentialForm
    orientation: int = eqx.field(static=True)

    def __init__(
        self,
        metric: RiemannianMetric,
        associative_form: DifferentialForm,
        /,
        *,
        orientation: int = 1,
    ):
        if not isinstance(metric, RiemannianMetric):
            raise TypeError("LocalG2Structure requires a RiemannianMetric.")
        if not isinstance(associative_form, DifferentialForm):
            raise TypeError("associative_form must be a DifferentialForm.")
        if metric.chart.dimension != 7:
            raise ValueError("A local G2 structure requires a seven-dimensional chart.")
        if associative_form.degree != 3:
            raise ValueError("A local G2 associative form must have degree three.")
        if not metric.chart.compatible_with(associative_form.chart):
            raise ValueError("G2 metric and associative-form charts must match.")
        if orientation not in (-1, 1):
            raise ValueError("G2 orientation must be +1 or -1.")
        self.metric = metric
        self.associative_form = associative_form
        self.orientation = int(orientation)

    def coassociative_form(self) -> DifferentialForm:
        return hodge_star(
            self.associative_form,
            self.metric,
            orientation=self.orientation,
        )


class OctonionG2Bridge(StrictModule):
    """Canonical flat G2 data induced by the declared octonion convention."""

    algebra: OctonionAlgebraSpec
    chart: CoordinateChart
    metric: RiemannianMetric
    product: AlgebraProductPlan
    coefficients: Array
    imaginary_basis_indices: tuple[int, ...] = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    bridge_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebra: OctonionAlgebraSpec,
        chart: CoordinateChart,
        /,
        *,
        orientation: int = 1,
    ):
        if not isinstance(algebra, OctonionAlgebraSpec):
            raise TypeError("OctonionG2Bridge requires an OctonionAlgebraSpec.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("OctonionG2Bridge requires a CoordinateChart.")
        if chart.dimension != 7:
            raise ValueError("Octonion G2 coordinates require a seven-dimensional chart.")
        if orientation not in (-1, 1):
            raise ValueError("G2 orientation must be +1 or -1.")
        imaginary = tuple(
            basis
            for basis in range(algebra.coordinate_dimension)
            if basis != algebra.scalar_basis_index
        )
        if len(imaginary) != 7:
            raise ValueError(
                "The octonion algebra must expose seven imaginary coordinates."
            )
        coefficients = []
        for left, middle, right in exterior_indices(7, 3):
            product = algebra.structure.basis_product(
                imaginary[left],
                imaginary[middle],
            )
            coefficients.append(orientation * product[imaginary[right]])
        self.algebra = algebra
        self.chart = chart
        self.metric = euclidean_metric(chart)
        self.product = algebra.prepare_product(backend="sparse")
        self.coefficients = jnp.asarray(
            [float(value) for value in coefficients],
            dtype=float,
        )
        self.imaginary_basis_indices = imaginary
        self.orientation = int(orientation)
        self.bridge_id = canonical_fingerprint(
            {
                "kind": "octonion-g2-bridge-v1",
                "algebra": algebra.algebra_id,
                "basis": list(imaginary),
                "chart": {
                    "name": chart.name,
                    "coordinates": list(chart.coordinates),
                },
                "orientation": int(orientation),
            }
        )

    def _imaginary_value(self, value: ArrayLike, owner: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape[-1:] != (7,):
            raise ValueError(f"{owner} must have trailing shape (7,).")
        if jnp.iscomplexobj(array):
            raise TypeError(f"{owner} must use real coordinates.")
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError(f"{owner} must use floating coordinates.")
        return array

    def embed_imaginary(self, value: ArrayLike, /) -> Array:
        imaginary = self._imaginary_value(value, "Imaginary octonion value")
        return (
            jnp.zeros(
                imaginary.shape[:-1] + (self.algebra.coordinate_dimension,),
                dtype=imaginary.dtype,
            )
            .at[..., jnp.asarray(self.imaginary_basis_indices, dtype=jnp.int32)]
            .set(imaginary)
        )

    def extract_imaginary(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if array.shape[-1:] != (self.algebra.coordinate_dimension,):
            raise ValueError("Octonion value must have trailing shape (8,).")
        if jnp.iscomplexobj(array):
            raise TypeError("Octonion value must use real coordinates.")
        return array[..., jnp.asarray(self.imaginary_basis_indices, dtype=jnp.int32)]

    def cross(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = self.embed_imaginary(left)
        right_ = self.embed_imaginary(right)
        return self.extract_imaginary(self.product(left_, right_))

    def associative_tensor(self) -> Array:
        """Return the canonical alternating G2 three-tensor."""
        return _dense_three_form(self.coefficients)

    def associative_differential_form(self) -> DifferentialForm:
        return DifferentialForm(
            _ConstantG2FormCoefficients(self.coefficients),
            chart=self.chart,
            degree=3,
        )

    def coassociative_differential_form(self) -> DifferentialForm:
        return hodge_star(
            self.associative_differential_form(),
            self.metric,
            orientation=self.orientation,
        )

    def local_structure(self) -> LocalG2Structure:
        return LocalG2Structure(
            self.metric,
            self.associative_differential_form(),
            orientation=self.orientation,
        )


class LocalG2ValidationReport(StrictModule):
    valid: Array
    algebraically_compatible: Array
    closed: Array
    coclosed: Array
    torsion_free: Array
    ricci_flat: Array
    maximum_metric_compatibility_residual: Array
    maximum_volume_normalization_residual: Array
    maximum_closure_residual: Array
    maximum_coclosure_residual: Array
    maximum_ricci_residual: Array
    required_torsion_free: bool = eqx.field(static=True)
    required_ricci_flat: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: ArrayLike,
        algebraically_compatible: ArrayLike,
        closed: ArrayLike,
        coclosed: ArrayLike,
        torsion_free: ArrayLike,
        ricci_flat: ArrayLike,
        maximum_metric_compatibility_residual: ArrayLike,
        maximum_volume_normalization_residual: ArrayLike,
        maximum_closure_residual: ArrayLike,
        maximum_coclosure_residual: ArrayLike,
        maximum_ricci_residual: ArrayLike,
        required_torsion_free: bool,
        required_ricci_flat: bool,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.algebraically_compatible = jnp.asarray(
            algebraically_compatible,
            dtype=bool,
        )
        self.closed = jnp.asarray(closed, dtype=bool)
        self.coclosed = jnp.asarray(coclosed, dtype=bool)
        self.torsion_free = jnp.asarray(torsion_free, dtype=bool)
        self.ricci_flat = jnp.asarray(ricci_flat, dtype=bool)
        self.maximum_metric_compatibility_residual = jnp.asarray(
            maximum_metric_compatibility_residual
        )
        self.maximum_volume_normalization_residual = jnp.asarray(
            maximum_volume_normalization_residual
        )
        self.maximum_closure_residual = jnp.asarray(maximum_closure_residual)
        self.maximum_coclosure_residual = jnp.asarray(maximum_coclosure_residual)
        self.maximum_ricci_residual = jnp.asarray(maximum_ricci_residual)
        self.required_torsion_free = bool(required_torsion_free)
        self.required_ricci_flat = bool(required_ricci_flat)


def validate_local_g2_structure(
    structure: LocalG2Structure,
    points: ArrayLike,
    /,
    *,
    compatibility_tolerance: float = 1e-8,
    normalization_tolerance: float = 1e-8,
    closure_tolerance: float = 1e-8,
    coclosure_tolerance: float = 1e-8,
    ricci_tolerance: float = 1e-7,
    require_torsion_free: bool = True,
    require_ricci_flat: bool = False,
    raise_on_error: bool = True,
) -> LocalG2ValidationReport:
    """Validate local metric compatibility and requested torsion/Ricci conditions."""
    if not isinstance(structure, LocalG2Structure):
        raise TypeError("structure must be a LocalG2Structure.")
    tolerances = (
        compatibility_tolerance,
        normalization_tolerance,
        closure_tolerance,
        coclosure_tolerance,
        ricci_tolerance,
    )
    if any(not isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("G2 validation tolerances must be finite and nonnegative.")
    if not isinstance(require_torsion_free, bool) or not isinstance(
        require_ricci_flat,
        bool,
    ):
        raise TypeError("G2 validation requirement flags must be Boolean.")

    point_values = jnp.asarray(points)
    coefficients = structure.associative_form(point_values)
    phi = _dense_three_form(coefficients)
    metric = structure.metric(point_values)
    inverse_metric = structure.metric.inverse(point_values)
    contraction = ein.contract(
        "...ikl,...ka,...lb,...jab->...ij",
        phi,
        inverse_metric,
        inverse_metric,
        phi,
        backend="jax",
    )
    metric_residual = jnp.max(jnp.abs(contraction - 6.0 * metric))

    coassociative = structure.coassociative_form()
    volume = wedge(structure.associative_form, coassociative)(point_values)[..., 0]
    expected_volume = (
        7.0 * structure.orientation * structure.metric.volume_density(point_values)
    )
    normalization_residual = jnp.max(jnp.abs(volume - expected_volume))
    algebraically_compatible = (
        (metric_residual <= compatibility_tolerance)
        & (normalization_residual <= normalization_tolerance)
        & jnp.isfinite(metric_residual)
        & jnp.isfinite(normalization_residual)
    )

    closure_residual = jnp.max(
        jnp.abs(exterior_derivative(structure.associative_form)(point_values))
    )
    coclosure_residual = jnp.max(
        jnp.abs(exterior_derivative(coassociative)(point_values))
    )
    closed = jnp.isfinite(closure_residual) & (closure_residual <= closure_tolerance)
    coclosed = jnp.isfinite(coclosure_residual) & (
        coclosure_residual <= coclosure_tolerance
    )
    torsion_free = closed & coclosed

    ricci_residual = jnp.max(jnp.abs(ricci_tensor(structure.metric, point_values)))
    ricci_flat = jnp.isfinite(ricci_residual) & (ricci_residual <= ricci_tolerance)
    valid = algebraically_compatible
    if require_torsion_free:
        valid = valid & torsion_free
    if require_ricci_flat:
        valid = valid & ricci_flat
    report = LocalG2ValidationReport(
        valid=valid,
        algebraically_compatible=algebraically_compatible,
        closed=closed,
        coclosed=coclosed,
        torsion_free=torsion_free,
        ricci_flat=ricci_flat,
        maximum_metric_compatibility_residual=metric_residual,
        maximum_volume_normalization_residual=normalization_residual,
        maximum_closure_residual=closure_residual,
        maximum_coclosure_residual=coclosure_residual,
        maximum_ricci_residual=ricci_residual,
        required_torsion_free=require_torsion_free,
        required_ricci_flat=require_ricci_flat,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Local G2 validation failed: "
            f"compatibility={float(jax.device_get(metric_residual))}, "
            f"normalization={float(jax.device_get(normalization_residual))}, "
            f"closure={float(jax.device_get(closure_residual))}, "
            f"coclosure={float(jax.device_get(coclosure_residual))}, "
            f"ricci={float(jax.device_get(ricci_residual))}."
        )
    return report


class G2DerivationInvarianceReport(StrictModule):
    valid: Array
    derivation_dimension: Array
    maximum_form_invariance_residual: Array
    maximum_metric_skew_residual: Array
    maximum_scalar_mixing_residual: Array
    algebra_id: str = eqx.field(static=True)
    derivation_plan_id: str = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: ArrayLike,
        derivation_dimension: ArrayLike,
        maximum_form_invariance_residual: ArrayLike,
        maximum_metric_skew_residual: ArrayLike,
        maximum_scalar_mixing_residual: ArrayLike,
        algebra_id: str,
        derivation_plan_id: str,
        tolerance: float,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.derivation_dimension = jnp.asarray(
            derivation_dimension,
            dtype=jnp.int32,
        )
        self.maximum_form_invariance_residual = jnp.asarray(
            maximum_form_invariance_residual
        )
        self.maximum_metric_skew_residual = jnp.asarray(maximum_metric_skew_residual)
        self.maximum_scalar_mixing_residual = jnp.asarray(maximum_scalar_mixing_residual)
        self.algebra_id = str(algebra_id)
        self.derivation_plan_id = str(derivation_plan_id)
        self.tolerance = float(tolerance)


def validate_g2_derivations(
    bridge: OctonionG2Bridge,
    derivations,
    /,
    *,
    tolerance: float = 1e-9,
    raise_on_error: bool = True,
) -> G2DerivationInvarianceReport:
    """Validate that prepared octonion derivations infinitesimally preserve G2."""
    from ..linalg import PreparedAlgebraDerivations

    if not isinstance(bridge, OctonionG2Bridge):
        raise TypeError("bridge must be an OctonionG2Bridge.")
    if not isinstance(derivations, PreparedAlgebraDerivations):
        raise TypeError("derivations must be PreparedAlgebraDerivations.")
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("G2 derivation tolerance must be finite and nonnegative.")
    bridge.algebra.require_compatible(derivations.plan.constraint.algebra)

    dimension = bridge.algebra.coordinate_dimension
    basis = derivations.subspace.basis
    capacity = derivations.subspace.capacity
    matrices = jnp.swapaxes(basis, 0, 1).reshape((capacity, dimension, dimension))
    imaginary = jnp.asarray(bridge.imaginary_basis_indices, dtype=jnp.int32)
    restricted = matrices[:, imaginary, :][:, :, imaginary]
    phi = bridge.associative_tensor().astype(basis.dtype)
    action = (
        ein.contract("nai,ajk->nijk", restricted, phi, backend="jax")
        + ein.contract("naj,iak->nijk", restricted, phi, backend="jax")
        + ein.contract("nak,ija->nijk", restricted, phi, backend="jax")
    )
    form_residual = _maximum_or_zero(action)
    skew_residual = _maximum_or_zero(restricted + jnp.swapaxes(restricted, -1, -2))
    scalar = bridge.algebra.scalar_basis_index
    scalar_mixing = jnp.concatenate(
        (
            matrices[:, scalar, :],
            matrices[:, :, scalar],
        ),
        axis=-1,
    )
    scalar_residual = _maximum_or_zero(scalar_mixing)
    finite = (
        jnp.isfinite(form_residual)
        & jnp.isfinite(skew_residual)
        & jnp.isfinite(scalar_residual)
    )
    valid = (
        derivations.converged
        & finite
        & (form_residual <= tolerance_)
        & (skew_residual <= tolerance_)
        & (scalar_residual <= tolerance_)
    )
    report = G2DerivationInvarianceReport(
        valid=valid,
        derivation_dimension=derivations.dimension,
        maximum_form_invariance_residual=form_residual,
        maximum_metric_skew_residual=skew_residual,
        maximum_scalar_mixing_residual=scalar_residual,
        algebra_id=bridge.algebra.algebra_id,
        derivation_plan_id=derivations.plan.plan_id,
        tolerance=tolerance_,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "G2 derivation invariance failed: "
            f"form={float(jax.device_get(form_residual))}, "
            f"metric={float(jax.device_get(skew_residual))}, "
            f"scalar={float(jax.device_get(scalar_residual))}."
        )
    return report


__all__ = [
    "G2DerivationInvarianceReport",
    "LocalG2Structure",
    "LocalG2ValidationReport",
    "LocalSUNStructure",
    "LocalSUNValidationReport",
    "OctonionG2Bridge",
    "validate_g2_derivations",
    "validate_local_g2_structure",
    "validate_local_su_structure",
]
