#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpy.typing as npt
import pytest
from jax import Array

from phydrax.linalg._singular_subspaces import (
    attach_selected_triplet_derivative,
    canonicalize_rows,
    canonicalize_singular_triplets,
    covariance_action,
    covariance_factor,
    make_subspace_response,
    projector_action,
    row_phase_evidence,
    selected_singular_responses,
    singular_value_response,
    SingularSubspaceResponse,
)
from phydrax.linalg.eigen._spectral_derivatives import (
    density_from_projector,
    density_tangent,
    isolated_projector_divided_difference,
)


type HostMatrix = npt.NDArray[np.float64] | npt.NDArray[np.complex128]


def _matrix(
    rows: int, columns: int, spectrum: tuple[float, ...], complex_data: bool
) -> HostMatrix:
    rng = np.random.default_rng(1024 + rows * 7 + columns)
    left = rng.normal(size=(rows, rows))
    right = rng.normal(size=(columns, columns))
    if complex_data:
        left = left + 1j * rng.normal(size=left.shape)
        right = right + 1j * rng.normal(size=right.shape)
    left_q, _ = np.linalg.qr(left)
    right_q, _ = np.linalg.qr(right)
    width = len(spectrum)
    return (left_q[:, :width] * np.asarray(spectrum)[None, :]) @ right_q[
        :, :width
    ].conj().T


def _direction(shape: tuple[int, int], complex_data: bool) -> HostMatrix:
    rng = np.random.default_rng(817)
    values = rng.normal(size=shape)
    if complex_data:
        values = values + 1j * rng.normal(size=shape)
    return values


def _response(
    matrix: Array, count: int
) -> tuple[Array, Array, Array, Array, Array, Array]:
    left, values, right_adjoint = jnp.linalg.svd(matrix, full_matrices=False)
    left, values, right = canonicalize_singular_triplets(
        left, values, jnp.conj(right_adjoint.T)
    )
    left, values, right = jax.tree.map(jax.lax.stop_gradient, (left, values, right))
    left_frame, right_frame, left_core, right_core = selected_singular_responses(
        matrix, left, values, right, jnp.arange(count, dtype=jnp.int32), jnp.asarray(True)
    )
    return (
        left_frame,
        right_frame,
        left_core,
        right_core,
        values[:count],
        right[:, :count],
    )


def _projector(matrix: Array, count: int, left_side: bool) -> Array:
    left, right, left_core, right_core, _, _ = _response(matrix, count)
    frame = left if left_side else right
    core = left_core if left_side else right_core
    response = make_subspace_response(frame, core)
    return projector_action(
        jax.lax.stop_gradient(frame), response, jnp.eye(frame.shape[0], dtype=frame.dtype)
    )


def _numpy_projector(matrix: HostMatrix, count: int, left_side: bool) -> HostMatrix:
    covariance = matrix @ matrix.conj().T if left_side else matrix.conj().T @ matrix
    _, vectors = np.linalg.eigh(covariance)
    frame = vectors[:, -count:]
    return frame @ frame.conj().T


def _numpy_triplets(
    matrix: HostMatrix, count: int
) -> tuple[HostMatrix, npt.NDArray[np.float64], HostMatrix]:
    left, values, right_adjoint = np.linalg.svd(matrix, full_matrices=False)
    right = right_adjoint.conj().T
    pivots = np.argmax(np.abs(right), axis=0)
    entries = right[pivots, np.arange(right.shape[1])]
    phases = entries.conj() / np.abs(entries)
    return (
        (left * phases[None, :])[:, :count],
        np.asarray(values[:count], dtype=np.float64),
        (right * phases[None, :])[:, :count],
    )


def _finite_difference(
    function: Callable[[HostMatrix], HostMatrix],
    matrix: HostMatrix,
    direction: HostMatrix,
) -> HostMatrix:
    step = 2e-5
    return (function(matrix + step * direction) - function(matrix - step * direction)) / (
        2 * step
    )


@pytest.mark.parametrize(
    ("rows", "columns", "spectrum", "count"),
    [
        (5, 3, (4.0, 4.0, 1.0), 2),
        (3, 5, (4.0, 4.0, 1.0), 2),
        (5, 5, (5.0, 3.0, 1.0, 1.0, 0.5), 2),
        (5, 3, (4.0, 2.0, 1.0), 3),
        (3, 5, (4.0, 2.0, 1.0), 3),
    ],
    ids=[
        "tall-retained-repeat",
        "wide-retained-repeat",
        "discarded-repeat",
        "tall-full-thin",
        "wide-full-thin",
    ],
)
@pytest.mark.parametrize("complex_data", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("left_side", [False, True], ids=["right", "left"])
def test_isolated_projector_jvp_and_vjp(
    rows: int,
    columns: int,
    spectrum: tuple[float, ...],
    count: int,
    complex_data: bool,
    left_side: bool,
) -> None:
    matrix = _matrix(rows, columns, spectrum, complex_data)
    direction = _direction(matrix.shape, complex_data)
    array, tangent = jnp.asarray(matrix), jnp.asarray(direction)

    def response(current: Array) -> Array:
        return _projector(current, count, left_side)

    def oracle(current: HostMatrix) -> HostMatrix:
        return _numpy_projector(current, count, left_side)

    primal, actual = jax.jvp(response, (array,), (tangent,))
    expected = _finite_difference(oracle, matrix, direction)
    np.testing.assert_allclose(primal, oracle(matrix), atol=2e-12)
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-8)
    probe = jnp.asarray(_direction(primal.shape, complex_data))

    def loss(current: Array) -> Array:
        return jnp.real(jnp.sum(jnp.conj(probe) * response(current)))

    gradient = jax.grad(loss)(array)
    contraction = jnp.real(jnp.sum(gradient * tangent))
    expected_contraction = np.real(np.sum(np.asarray(probe).conj() * expected))
    np.testing.assert_allclose(contraction, expected_contraction, rtol=3e-7, atol=3e-8)
    if count == (rows if left_side else columns):
        np.testing.assert_array_equal(actual, jnp.zeros_like(actual))


@pytest.mark.parametrize("complex_data", [False, True], ids=["real", "complex"])
def test_selected_covariance_core_and_smooth_factor(complex_data: bool) -> None:
    matrix = _matrix(3, 5, (4.0, 4.0, 1.0), complex_data)
    direction = _direction(matrix.shape, complex_data)

    def covariance(current: Array) -> Array:
        _, frame, _, core, values, _ = _response(current, 2)
        response = make_subspace_response(frame, core)
        return covariance_action(
            jax.lax.stop_gradient(frame),
            values,
            response,
            jnp.eye(frame.shape[0], dtype=frame.dtype),
        )

    def factor_covariance(current: Array) -> Array:
        _, frame, _, core, values, _ = _response(current, 2)
        response = make_subspace_response(frame, core)
        factor = covariance_factor(jax.lax.stop_gradient(frame), values, response)
        return factor @ jnp.conj(factor.T)

    def oracle(current: HostMatrix) -> HostMatrix:
        projector = _numpy_projector(current, 2, False)
        return projector @ (current.conj().T @ current) @ projector

    expected = _finite_difference(oracle, matrix, direction)
    array, tangent = jnp.asarray(matrix), jnp.asarray(direction)
    _, actual = jax.jvp(covariance, (array,), (tangent,))
    factor_primal, factor_tangent = jax.jvp(factor_covariance, (array,), (tangent,))
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=5e-8)
    np.testing.assert_allclose(factor_tangent, expected, rtol=3e-7, atol=5e-8)
    np.testing.assert_allclose(factor_primal, oracle(matrix), atol=2e-11)


def test_covariance_core_retains_noncommuting_selected_perturbations() -> None:
    matrix = jnp.diag(jnp.asarray([4.0, 4.0, 1.0]))
    direction = jnp.asarray([[0.2, 0.7, 0], [-0.3, -0.4, 0], [0, 0, 0.8]])

    def core(current: Array) -> Array:
        _, frame, _, selected_core, _, _ = _response(current, 2)
        return frame @ selected_core @ frame.T

    _, tangent = jax.jvp(core, (matrix,), (direction,))
    # Independent analytical covariance differential at a repeated retained block.
    expected = (
        jnp.zeros((3, 3)).at[:2, :2].set(4 * (direction[:2, :2] + direction[:2, :2].T))
    )
    np.testing.assert_allclose(tangent, expected, atol=2e-12)


@pytest.mark.parametrize("complex_data", [False, True], ids=["real", "complex"])
def test_canonical_basis_triplet_tangent_and_phase_relation(complex_data: bool) -> None:
    matrix = jnp.asarray(_matrix(5, 3, (5.0, 3.0, 1.0), complex_data))
    direction = jnp.asarray(_direction((matrix.shape[0], matrix.shape[1]), complex_data))

    def triplets(current: Array) -> tuple[Array, Array, Array]:
        left, values, right_adjoint = jnp.linalg.svd(current, full_matrices=False)
        left, values, right = jax.tree.map(
            jax.lax.stop_gradient, (left, values, jnp.conj(right_adjoint.T))
        )
        return attach_selected_triplet_derivative(
            current,
            left,
            values,
            right,
            jnp.arange(2, dtype=jnp.int32),
            jnp.asarray(True),
        )

    (left, values, right), (left_tangent, value_tangent, right_tangent) = jax.jvp(
        triplets, (matrix,), (direction,)
    )
    np.testing.assert_allclose(matrix @ right, left * values, atol=2e-12)
    np.testing.assert_allclose(
        direction @ right + matrix @ right_tangent,
        left_tangent * values + left * value_tangent,
        atol=3e-12,
    )
    step = 2e-5
    minus = _numpy_triplets(np.asarray(matrix - step * direction), 2)
    plus = _numpy_triplets(np.asarray(matrix + step * direction), 2)
    for actual, lower, upper in zip(
        (left_tangent, value_tangent, right_tangent), minus, plus, strict=True
    ):
        np.testing.assert_allclose(
            actual, (upper - lower) / (2 * step), rtol=3e-7, atol=3e-8
        )
    pivots = jnp.argmax(jnp.abs(right), axis=0)
    np.testing.assert_allclose(jnp.imag(right[pivots, jnp.arange(2)]), 0, atol=2e-12)
    np.testing.assert_allclose(
        jnp.imag(right_tangent[pivots, jnp.arange(2)]), 0, atol=2e-12
    )
    left_probe = jnp.asarray(_direction(left.shape, complex_data))
    right_probe = jnp.asarray(_direction(right.shape, complex_data))
    value_probe = jnp.asarray([0.7, -0.2])

    def loss(current: Array) -> Array:
        current_left, current_values, current_right = triplets(current)
        return jnp.real(
            jnp.sum(jnp.conj(left_probe) * current_left)
            + jnp.sum(jnp.conj(right_probe) * current_right)
        ) + jnp.sum(value_probe * current_values)

    reverse_direction = jnp.real(jnp.sum(jax.grad(loss)(matrix) * direction))
    finite_difference = np.real(
        np.sum(np.asarray(left_probe).conj() * (plus[0] - minus[0]))
        + np.sum(np.asarray(right_probe).conj() * (plus[2] - minus[2]))
    ) / (2 * step) + np.sum(np.asarray(value_probe) * (plus[1] - minus[1])) / (2 * step)
    np.testing.assert_allclose(reverse_direction, finite_difference, rtol=3e-7, atol=3e-8)


@pytest.mark.parametrize("left_side", [False, True], ids=["source", "target"])
@pytest.mark.parametrize("full_metric", [False, True], ids=["diagonal", "dense-SPD"])
def test_moving_both_metrics_projector_and_values(
    left_side: bool, full_metric: bool
) -> None:
    matrix = jnp.asarray(_matrix(4, 3, (5.0, 3.0, 1.0), True))
    source = jnp.diag(jnp.asarray([1.2, 0.7, 1.8], dtype=jnp.complex128))
    target = jnp.diag(jnp.asarray([0.8, 1.5, 1.1, 1.9], dtype=jnp.complex128))
    if full_metric:
        source = source.at[0, 1].set(0.1 + 0.03j).at[1, 0].set(0.1 - 0.03j)
        target = target.at[0, 2].set(-0.12 + 0.04j).at[2, 0].set(-0.12 - 0.04j)
    source_tangent = jnp.asarray(
        [[0.1, 0.02j, 0.03], [-0.02j, -0.2, 0.01], [0.03, 0.01, 0.15]]
    )
    target_tangent = jnp.asarray(
        [
            [0.2, 0.01, 0.03j, 0.02],
            [0.01, -0.1, 0.02, 0.04j],
            [-0.03j, 0.02, 0.05, 0.01],
            [0.02, -0.04j, 0.01, 0.12],
        ]
    )

    def result(source_metric: Array, target_metric: Array) -> tuple[Array, Array]:
        source_factor = jnp.conj(jnp.linalg.cholesky(source_metric).T)
        target_factor = jnp.conj(jnp.linalg.cholesky(target_metric).T)
        reduced = (
            target_factor
            @ jsp.linalg.solve_triangular(source_factor.T, matrix.T, lower=True).T
        )
        left, values, right_adjoint = jnp.linalg.svd(reduced, full_matrices=False)
        left, values, right = jax.tree.map(
            jax.lax.stop_gradient, (left, values, jnp.conj(right_adjoint.T))
        )
        selected = jnp.arange(2, dtype=jnp.int32)
        left_response, right_response, _, _ = selected_singular_responses(
            reduced, left, values, right, selected, jnp.asarray(True)
        )
        frame = left_response if left_side else right_response
        factor = target_factor if left_side else source_factor
        physical = jsp.linalg.solve_triangular(factor, frame, lower=False)
        projector = physical @ jnp.conj(frame.T) @ factor
        selected_values = singular_value_response(
            reduced, left, values, right, selected, jnp.asarray(True)
        )
        return projector, selected_values

    _, actual = jax.jvp(result, (source, target), (source_tangent, target_tangent))
    step = 2e-5

    # Independent NumPy whitening and covariance-eigh projector oracle.
    def oracle(sign: float) -> tuple[HostMatrix, HostMatrix]:
        gs = np.asarray(source + sign * step * source_tangent)
        gt = np.asarray(target + sign * step * target_tangent)
        fs, ft = np.linalg.cholesky(gs).conj().T, np.linalg.cholesky(gt).conj().T
        reduced = ft @ np.linalg.solve(fs.T, np.asarray(matrix).T).T
        orthogonal = _numpy_projector(reduced, 2, left_side)
        factor = ft if left_side else fs
        projector = np.linalg.solve(factor, orthogonal @ factor)
        values = np.linalg.svd(reduced, compute_uv=False)[:2]
        return projector, values

    minus, plus = oracle(-1.0), oracle(1.0)
    for tangent, lower, upper in zip(actual, minus, plus, strict=True):
        np.testing.assert_allclose(
            tangent, (upper - lower) / (2 * step), rtol=4e-7, atol=3e-8
        )
    projector_probe = jnp.asarray(_direction(actual[0].shape, True))
    value_probe = jnp.asarray([0.7, -0.2])

    def loss(source_metric: Array, target_metric: Array) -> Array:
        current_projector, current_values = result(source_metric, target_metric)
        return jnp.real(jnp.sum(jnp.conj(projector_probe) * current_projector)) + jnp.sum(
            value_probe * current_values
        )

    source_gradient, target_gradient = jax.grad(loss, argnums=(0, 1))(source, target)
    reverse_direction = jnp.real(
        jnp.sum(source_gradient * source_tangent)
        + jnp.sum(target_gradient * target_tangent)
    )
    expected_direction = np.real(
        np.sum(np.asarray(projector_probe).conj() * (plus[0] - minus[0]) / (2 * step))
    ) + np.sum(np.asarray(value_probe) * (plus[1] - minus[1]) / (2 * step))
    np.testing.assert_allclose(
        reverse_direction, expected_direction, rtol=4e-7, atol=3e-8
    )


def test_phase_ties_and_zero_primal_carrier_are_real_invariants() -> None:
    rows = jnp.asarray([[1j, -1j, 0], [0, 0, 0], [1, 0.5j, -0.2]])
    magnitude, margin = row_phase_evidence(rows)
    np.testing.assert_array_equal(magnitude, jnp.asarray([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(margin, jnp.asarray([0.0, 0.0, 0.5]))
    np.testing.assert_allclose(canonicalize_rows(rows)[0], jnp.asarray([1, -1, 0]))
    matrix = jnp.asarray(_matrix(3, 5, (4.0, 4.0, 1.0), False))
    _, frame, _, core, _, _ = _response(matrix, 2)
    response = make_subspace_response(frame, core)
    np.testing.assert_array_equal(response.frame_correction, jnp.zeros_like(frame))
    np.testing.assert_array_equal(response.covariance_correction, jnp.zeros_like(core))
    with pytest.raises(Exception, match="identically zero"):
        SingularSubspaceResponse(jnp.ones((5, 2)), jnp.zeros((2, 2)))


def test_zero_full_ambient_projectors_have_zero_tangent() -> None:
    matrix = jnp.zeros((3, 3), dtype=jnp.float64)
    direction = jnp.asarray(_direction((3, 3), False))
    for left_side in (False, True):

        def projector(current: Array) -> Array:
            return _projector(current, 3, left_side)

        primal, tangent = jax.jvp(projector, (matrix,), (direction,))
        np.testing.assert_allclose(primal, jnp.eye(3), atol=2e-12)
        np.testing.assert_array_equal(tangent, jnp.zeros_like(tangent))


def test_rejected_boundary_does_not_supply_a_false_supported_derivative() -> None:
    matrix = jnp.diag(jnp.asarray([4.0, 4.0, 1.0]))

    def rejected(current: Array) -> Array:
        left, values, right_adjoint = jnp.linalg.svd(current, full_matrices=False)
        _, right, _, _ = selected_singular_responses(
            current,
            jax.lax.stop_gradient(left),
            jax.lax.stop_gradient(values),
            jax.lax.stop_gradient(right_adjoint.T),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray(False),
        )
        return right @ right.T

    with pytest.raises(Exception, match="not admitted"):
        jax.jvp(rejected, (matrix,), (jnp.ones_like(matrix),))


@pytest.mark.parametrize("complex_data", [False, True], ids=["real", "complex"])
def test_shared_eigen_cluster_and_density_response_preserves_internal_repeats(
    complex_data: bool,
) -> None:
    values = jnp.asarray([2.0, 2.0, 5.0, 5.0])
    selected = jnp.asarray([True, True, False, False])
    direction = _direction((4, 4), complex_data)
    matrix = np.diag(np.asarray(values)).astype(direction.dtype)
    metric = np.diag(np.asarray([1.1, 1.7, 0.9, 2.0])).astype(direction.dtype)
    metric_direction = (direction + direction.conj().T) / 7
    projector = jnp.diag(selected.astype(jnp.float64)).astype(
        jnp.asarray(direction).dtype
    )
    projector_derivative = isolated_projector_divided_difference(
        values, jnp.asarray(direction), selected
    )
    density = density_from_projector(projector, jnp.asarray(metric))
    density_derivative = density_tangent(
        projector,
        projector_derivative,
        density,
        jnp.asarray(metric),
        jnp.asarray(metric_direction),
    )

    def oracle(sign: float) -> tuple[HostMatrix, HostMatrix]:
        step = 2e-5
        spectrum, vectors = np.linalg.eig(matrix + sign * step * direction)
        mask = np.real(spectrum) < 3.5
        current_projector = np.linalg.solve(vectors.T, (vectors * mask[None, :]).T).T
        current_metric = metric + sign * step * metric_direction
        current_density = np.linalg.solve(current_metric.T, current_projector.T).T
        return current_projector, current_density

    minus, plus = oracle(-1.0), oracle(1.0)
    for actual, lower, upper in zip(
        (projector_derivative, density_derivative), minus, plus, strict=True
    ):
        np.testing.assert_allclose(actual, (upper - lower) / 4e-5, rtol=3e-7, atol=3e-8)
