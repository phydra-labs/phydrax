#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from math import factorial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..linalg import inverse as matrix_inverse
from ._chart import CoordinateChart
from ._forms import DifferentialForm, exterior_derivative, wedge
from ._map import DifferentiableMap
from ._utils import _pointwise_array


class _CanonicalSymplecticCoefficients(StrictModule):
    dimension: int
    pair_positions: tuple[int, ...]

    def __init__(self, dimension: int, /):
        half = dimension // 2
        indices = tuple(combinations(range(dimension), 2))
        lookup = {index: position for position, index in enumerate(indices)}
        self.dimension = dimension
        self.pair_positions = tuple(lookup[(axis, half + axis)] for axis in range(half))

    def __call__(self, coordinates: Array, /) -> Array:
        values = jnp.zeros(
            (self.dimension * (self.dimension - 1) // 2,),
            dtype=coordinates.dtype,
        )
        return values.at[jnp.asarray(self.pair_positions, dtype=jnp.int32)].set(1.0)


class SymplecticForm(StrictModule):
    """A nondegenerate two-form candidate with explicit chart semantics."""

    form: DifferentialForm
    chart: CoordinateChart

    def __init__(self, form: DifferentialForm, /):
        if not isinstance(form, DifferentialForm):
            raise TypeError("SymplecticForm requires a DifferentialForm.")
        if form.degree != 2:
            raise ValueError("A symplectic form must have degree two.")
        if form.chart.dimension % 2:
            raise ValueError("A symplectic chart must have even dimension.")
        self.form = form
        self.chart = form.chart

    def matrix(self, coordinates: ArrayLike, /) -> Array:
        coefficients = self.form(coordinates)
        matrix = jnp.zeros(
            coefficients.shape[:-1] + (self.chart.dimension, self.chart.dimension),
            dtype=coefficients.dtype,
        )
        for position, (left, right) in enumerate(self.form.indices):
            matrix = matrix.at[..., left, right].set(coefficients[..., position])
            matrix = matrix.at[..., right, left].set(-coefficients[..., position])
        return matrix

    def inverse(self, coordinates: ArrayLike, /) -> Array:
        result = matrix_inverse(self.matrix(coordinates))
        return eqx.error_if(
            result.value,
            jnp.any(~result.successful),
            "Symplectic form is degenerate.",
        )


class PoissonStructure(StrictModule):
    """A skew contravariant two-tensor candidate on one coordinate chart."""

    bivector_function: Callable[[Array], Array]
    chart: CoordinateChart

    def __init__(
        self,
        bivector: Callable[[Array], Array],
        /,
        *,
        chart: CoordinateChart,
    ):
        if not callable(bivector):
            raise TypeError("Poisson bivector must be callable.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("Poisson chart must be a CoordinateChart.")
        self.bivector_function = bivector
        self.chart = chart

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        matrix = _pointwise_array(
            self.bivector_function,
            coordinates,
            self.chart.dimension,
        )
        expected = (self.chart.dimension, self.chart.dimension)
        if matrix.shape[-2:] != expected:
            raise ValueError(
                f"Poisson bivector must have trailing shape {expected}; got {matrix.shape}."
            )
        return matrix


class _SymplecticPoissonMap(StrictModule):
    symplectic: SymplecticForm

    def __init__(self, symplectic: SymplecticForm, /):
        self.symplectic = symplectic

    def __call__(self, coordinates: Array, /) -> Array:
        return -self.symplectic.inverse(coordinates)


class _HamiltonianVectorEvaluator(StrictModule):
    hamiltonian: Callable[[Array], Array]
    poisson: PoissonStructure

    def __init__(
        self,
        hamiltonian: Callable[[Array], Array],
        poisson: PoissonStructure,
        /,
    ):
        self.hamiltonian = hamiltonian
        self.poisson = poisson

    def __call__(self, coordinates: Array, /) -> Array:
        value = jnp.asarray(self.hamiltonian(coordinates))
        if value.shape != ():
            raise ValueError("A Hamiltonian must be scalar-valued.")
        return self.poisson.bivector_function(coordinates) @ jax.grad(self.hamiltonian)(
            coordinates
        )


class _PoissonBracketEvaluator(StrictModule):
    left: Callable[[Array], Array]
    right: Callable[[Array], Array]
    poisson: PoissonStructure

    def __init__(
        self,
        left: Callable[[Array], Array],
        right: Callable[[Array], Array],
        poisson: PoissonStructure,
        /,
    ):
        self.left = left
        self.right = right
        self.poisson = poisson

    def __call__(self, coordinates: Array, /) -> Array:
        left_value = jnp.asarray(self.left(coordinates))
        right_value = jnp.asarray(self.right(coordinates))
        if left_value.shape != () or right_value.shape != ():
            raise ValueError("Poisson brackets require scalar functions.")
        return ein.contract(
            "i,ij,j->",
            jax.grad(self.left)(coordinates),
            self.poisson.bivector_function(coordinates),
            jax.grad(self.right)(coordinates),
        )


class SymplecticValidationReport(StrictModule):
    valid: Array
    closed: Array
    nondegenerate: Array
    maximum_closure_residual: Array
    minimum_singular_value: Array

    def __init__(
        self,
        *,
        valid: Array,
        closed: Array,
        nondegenerate: Array,
        maximum_closure_residual: Array,
        minimum_singular_value: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.closed = jnp.asarray(closed, dtype=bool)
        self.nondegenerate = jnp.asarray(nondegenerate, dtype=bool)
        self.maximum_closure_residual = jnp.asarray(maximum_closure_residual)
        self.minimum_singular_value = jnp.asarray(minimum_singular_value)


class PoissonValidationReport(StrictModule):
    valid: Array
    skew: Array
    jacobi: Array
    maximum_skew_residual: Array
    maximum_jacobi_residual: Array

    def __init__(
        self,
        *,
        valid: Array,
        skew: Array,
        jacobi: Array,
        maximum_skew_residual: Array,
        maximum_jacobi_residual: Array,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.skew = jnp.asarray(skew, dtype=bool)
        self.jacobi = jnp.asarray(jacobi, dtype=bool)
        self.maximum_skew_residual = jnp.asarray(maximum_skew_residual)
        self.maximum_jacobi_residual = jnp.asarray(maximum_jacobi_residual)


class _ScaledForm(StrictModule):
    form: DifferentialForm
    scale: float

    def __init__(self, form: DifferentialForm, scale: float, /):
        self.form = form
        self.scale = float(scale)

    def __call__(self, coordinates: Array, /) -> Array:
        return self.scale * self.form._coefficients_point(coordinates)


def canonical_symplectic_form(chart: CoordinateChart, /) -> SymplecticForm:
    """Return ``Σᵢ dqⁱ ∧ dpᵢ`` for coordinates ordered as ``(q, p)``."""
    if chart.dimension % 2:
        raise ValueError("Canonical symplectic coordinates require even dimension.")
    return SymplecticForm(
        DifferentialForm(
            _CanonicalSymplecticCoefficients(chart.dimension),
            chart=chart,
            degree=2,
        )
    )


def symplectic_to_poisson(symplectic: SymplecticForm, /) -> PoissonStructure:
    if not isinstance(symplectic, SymplecticForm):
        raise TypeError("symplectic_to_poisson requires a SymplecticForm.")
    return PoissonStructure(
        _SymplecticPoissonMap(symplectic),
        chart=symplectic.chart,
    )


def hamiltonian_vector_field(
    hamiltonian: Callable[[Array], Array],
    structure: SymplecticForm | PoissonStructure,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the Hamiltonian vector field without calling it a gradient."""
    if not callable(hamiltonian):
        raise TypeError("hamiltonian must be callable.")
    poisson = (
        symplectic_to_poisson(structure)
        if isinstance(structure, SymplecticForm)
        else structure
    )
    if not isinstance(poisson, PoissonStructure):
        raise TypeError("structure must be a SymplecticForm or PoissonStructure.")
    return _pointwise_array(
        _HamiltonianVectorEvaluator(hamiltonian, poisson),
        coordinates,
        poisson.chart.dimension,
    )


def poisson_bracket(
    left: Callable[[Array], Array],
    right: Callable[[Array], Array],
    poisson: PoissonStructure,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``{left, right} = d left · Π · d right``."""
    if not callable(left) or not callable(right):
        raise TypeError("Poisson-bracket operands must be callable.")
    if not isinstance(poisson, PoissonStructure):
        raise TypeError("poisson must be a PoissonStructure.")
    return _pointwise_array(
        _PoissonBracketEvaluator(left, right, poisson),
        coordinates,
        poisson.chart.dimension,
    )


def poisson_jacobi_tensor(
    poisson: PoissonStructure,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the coordinate Schouten/Jacobi residual ``[Π, Π]`` up to scale."""
    if not isinstance(poisson, PoissonStructure):
        raise TypeError("poisson_jacobi_tensor requires a PoissonStructure.")

    def pointwise(point: Array) -> Array:
        bivector = poisson.bivector_function(point)
        derivative = jax.jacfwd(poisson.bivector_function)(point)
        first = ein.contract("il,jkl->ijk", bivector, derivative)
        second = ein.contract("jl,kil->ijk", bivector, derivative)
        third = ein.contract("kl,ijl->ijk", bivector, derivative)
        return first + second + third

    return _pointwise_array(pointwise, coordinates, poisson.chart.dimension)


def casimir_residual(
    function: Callable[[Array], Array],
    poisson: PoissonStructure,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return ``||Π d function||₂`` at each point."""
    vector = hamiltonian_vector_field(function, poisson, coordinates)
    return jnp.linalg.norm(vector, axis=-1)


def symplecticity_residual(
    map: DifferentiableMap,
    source: SymplecticForm,
    target: SymplecticForm,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the maximum residual of ``Jᵀ Ω_target J = Ω_source``."""
    if not isinstance(source, SymplecticForm) or not isinstance(target, SymplecticForm):
        raise TypeError("source and target must be SymplecticForm instances.")
    if not isinstance(map, DifferentiableMap):
        raise TypeError("map must be a DifferentiableMap.")
    if not map.source.compatible_with(source.chart) or not map.target.compatible_with(
        target.chart
    ):
        raise ValueError("Map and symplectic-form charts are incompatible.")
    jacobian = map.jacobian(coordinates)
    target_matrix = target.matrix(map(coordinates))
    pullback = ein.contract("...ai,...ab,...bj->...ij", jacobian, target_matrix, jacobian)
    return jnp.max(jnp.abs(pullback - source.matrix(coordinates)), axis=(-2, -1))


def validate_symplectic_form(
    symplectic: SymplecticForm,
    points: ArrayLike,
    /,
    *,
    closure_tolerance: float = 1e-9,
    nondegeneracy_tolerance: float = 1e-10,
) -> SymplecticValidationReport:
    if not isinstance(symplectic, SymplecticForm):
        raise TypeError("validate_symplectic_form requires a SymplecticForm.")
    if closure_tolerance < 0.0 or nondegeneracy_tolerance < 0.0:
        raise ValueError("Symplectic validation tolerances must be non-negative.")
    if symplectic.chart.dimension == 2:
        closure_residual = jnp.asarray(0.0)
    else:
        derivative = exterior_derivative(symplectic.form)(points)
        closure_residual = jnp.max(jnp.abs(derivative))
    singular_values = jnp.linalg.svd(symplectic.matrix(points), compute_uv=False)
    minimum_singular_value = jnp.min(singular_values)
    closed = closure_residual <= closure_tolerance
    nondegenerate = minimum_singular_value > nondegeneracy_tolerance
    return SymplecticValidationReport(
        valid=closed & nondegenerate,
        closed=closed,
        nondegenerate=nondegenerate,
        maximum_closure_residual=closure_residual,
        minimum_singular_value=minimum_singular_value,
    )


def validate_poisson_structure(
    poisson: PoissonStructure,
    points: ArrayLike,
    /,
    *,
    skew_tolerance: float = 1e-10,
    jacobi_tolerance: float = 1e-9,
) -> PoissonValidationReport:
    if not isinstance(poisson, PoissonStructure):
        raise TypeError("validate_poisson_structure requires a PoissonStructure.")
    if skew_tolerance < 0.0 or jacobi_tolerance < 0.0:
        raise ValueError("Poisson validation tolerances must be non-negative.")
    bivector = poisson(points)
    skew_residual = jnp.max(jnp.abs(bivector + jnp.swapaxes(bivector, -1, -2)))
    jacobi_residual = jnp.max(jnp.abs(poisson_jacobi_tensor(poisson, points)))
    skew = skew_residual <= skew_tolerance
    jacobi = jacobi_residual <= jacobi_tolerance
    return PoissonValidationReport(
        valid=skew & jacobi,
        skew=skew,
        jacobi=jacobi,
        maximum_skew_residual=skew_residual,
        maximum_jacobi_residual=jacobi_residual,
    )


def liouville_volume_form(symplectic: SymplecticForm, /) -> DifferentialForm:
    """Return ``ωⁿ / n!`` for a ``2n``-dimensional symplectic form."""
    if not isinstance(symplectic, SymplecticForm):
        raise TypeError("liouville_volume_form requires a SymplecticForm.")
    half_dimension = symplectic.chart.dimension // 2
    result = symplectic.form
    for _ in range(1, half_dimension):
        result = wedge(result, symplectic.form)
    return DifferentialForm(
        _ScaledForm(result, 1.0 / factorial(half_dimension)),
        chart=symplectic.chart,
        degree=symplectic.chart.dimension,
    )


__all__ = [
    "PoissonStructure",
    "PoissonValidationReport",
    "SymplecticForm",
    "SymplecticValidationReport",
    "canonical_symplectic_form",
    "casimir_residual",
    "hamiltonian_vector_field",
    "liouville_volume_form",
    "poisson_bracket",
    "poisson_jacobi_tensor",
    "symplectic_to_poisson",
    "symplecticity_residual",
    "validate_poisson_structure",
    "validate_symplectic_form",
]
