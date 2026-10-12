"""Consumer-visible first derivatives of Hermitian matrix functions and maps."""

from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

from phydrax.linalg import (
    hermitian_exp,
    hermitian_inverse_sqrt,
    hermitian_log,
    hermitian_spectral_map,
    hermitian_sqrt,
    HermitianFunctionResult,
    HermitianPrecisionPolicy,
    HermitianSpectralMapResult,
)
from phydrax.nonlinear import LocalRootPlan, VectorLocalRootPlan


pytestmark = pytest.mark.strict_jax
jax.config.update("jax_enable_x64", True)


class MatrixFunction(Protocol):
    def __call__(
        self,
        matrix: Array,
        /,
        *,
        tolerance: float = 1e-10,
        precision: HermitianPrecisionPolicy | None = None,
    ) -> HermitianFunctionResult: ...


type Invariants = dict[str, Array]
type PrincipalMap = Callable[[Array, Array], tuple[Array, Invariants]]

FUNCTIONS = (
    pytest.param(hermitian_log, 1.0, 1.0, id="log"),
    pytest.param(hermitian_exp, 0.0, 1.0, id="exp"),
    pytest.param(hermitian_sqrt, 1.0, 0.5, id="sqrt"),
    pytest.param(hermitian_inverse_sqrt, 1.0, -0.5, id="inverse-sqrt"),
)
SPECTRA = (
    pytest.param((1.0, 1.0, 1.0), id="identity"),
    pytest.param((1.2, 1.2, 2.1), id="two-equal"),
    pytest.param((1.7, 1.7, 1.7), id="three-equal"),
    pytest.param((1.0, 1.0 + 1e-9, 2.0), id="relative-gap-1e-9"),
    pytest.param((0.8, 1.4, 2.2), id="rotated-distinct"),
)


def _rotation(dtype: DTypeLike) -> Array:
    return jnp.asarray(((0.6, -0.8, 0.0), (0.8, 0.6, 0.0), (0.0, 0.0, 1.0)), dtype=dtype)


def _matrix(eigenvalues: tuple[float, ...], dtype: DTypeLike) -> Array:
    rotation = _rotation(dtype)
    return (rotation * jnp.asarray(eigenvalues, dtype=dtype)[None, :]) @ rotation.T


def _direction(dtype: DTypeLike) -> Array:
    return jnp.asarray(((0.3, 0.2, -0.1), (0.4, -0.2, 0.5), (0.2, 0.1, 0.4)), dtype=dtype)


@pytest.mark.parametrize("function,center,factor", FUNCTIONS)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64), ids=("float32", "float64"))
def test_confluent_closed_forms(
    function: MatrixFunction, center: float, factor: float, dtype: DTypeLike
) -> None:
    matrix = center * jnp.eye(3, dtype=dtype)
    direction = _direction(dtype)

    def value(argument: Array) -> Array:
        return function(argument).value

    _, tangent = jax.jit(lambda a, e: jax.jvp(value, (a,), (e,)))(matrix, direction)
    expected = factor * (direction + direction.T) / 2
    np.testing.assert_allclose(tangent, expected, rtol=2e-6, atol=2e-6)
    assert tangent.dtype == matrix.dtype


@pytest.mark.parametrize("function,center,factor", FUNCTIONS)
@pytest.mark.parametrize("eigenvalues", SPECTRA)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64), ids=("float32", "float64"))
def test_forward_reverse_and_batched(
    function: MatrixFunction,
    center: float,
    factor: float,
    eigenvalues: tuple[float, ...],
    dtype: DTypeLike,
) -> None:
    del center, factor
    matrix = _matrix(eigenvalues, dtype)

    def value(argument: Array) -> Array:
        return function(argument).value

    forward = jax.jit(jax.jacfwd(value))(matrix)
    reverse = jax.jit(jax.jacrev(value))(matrix)
    assert bool(jnp.all(jnp.isfinite(forward)))
    np.testing.assert_allclose(forward, reverse, rtol=5e-5, atol=5e-6)
    batch = jnp.stack((matrix, matrix * 1.1)).reshape((1, 2, 3, 3))
    _, tangent = jax.jvp(value, (batch,), (jnp.ones_like(batch),))
    expected = jax.vmap(lambda a: jax.jvp(value, (a,), (jnp.ones_like(a),))[1])(batch[0])
    np.testing.assert_allclose(tangent[0], expected, rtol=5e-5, atol=5e-6)
    gradient = jax.jit(jax.grad(lambda a: jnp.sum(value(a))))(matrix)
    assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.parametrize("function,center,factor", FUNCTIONS)
def test_distinct_central_difference(
    function: MatrixFunction, center: float, factor: float
) -> None:
    del center, factor
    matrix = _matrix((0.8, 1.4, 2.2), jnp.float64)
    direction = _direction(jnp.float64)

    def value(argument: Array) -> Array:
        return function(argument).value

    _, tangent = jax.jvp(value, (matrix,), (direction,))
    step = 1e-5
    finite_difference = (
        value(matrix + step * direction) - value(matrix - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("function,center,factor", FUNCTIONS)
@pytest.mark.parametrize("eigenvalues", SPECTRA)
def test_complex_hermitian_first_derivative(
    function: MatrixFunction, center: float, factor: float, eigenvalues: tuple[float, ...]
) -> None:
    del center, factor
    phases = jnp.asarray((1.0, 1j, -1.0), dtype=jnp.complex128)
    matrix = (
        phases[:, None] * _matrix(eigenvalues, jnp.complex128) * phases.conj()[None, :]
    )
    direction = _direction(jnp.complex128) + 0.2j * jnp.eye(3, dtype=jnp.complex128)
    direction = (direction + direction.conj().T) / 2

    def value(argument: Array) -> Array:
        return function(argument).value

    _, tangent = jax.jit(lambda a, e: jax.jvp(value, (a,), (e,)))(matrix, direction)
    step = 1e-5
    finite_difference = (
        value(matrix + step * direction) - value(matrix - step * direction)
    ) / (2 * step)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-8, atol=2e-9)

    # Real coordinates exercise reverse AD without imposing holomorphic semantics.
    def coordinates(x: Array) -> Array:
        return jnp.real(value(matrix + x * direction))

    np.testing.assert_allclose(
        jax.jacfwd(coordinates)(0.0), jax.jacrev(coordinates)(0.0), rtol=2e-10, atol=2e-10
    )


def test_derivative_valid_and_stopped_diagnostics() -> None:
    singular = jnp.diag(jnp.asarray((0.0, 1.0, 2.0), dtype=jnp.float64))
    square_root = hermitian_sqrt(singular)
    assert bool(square_root.valid)
    assert not bool(square_root.derivative_valid)
    for function in (hermitian_log, hermitian_inverse_sqrt):
        result = function(singular)
        assert not bool(result.valid)
        assert not bool(result.derivative_valid)
    assert bool(hermitian_exp(singular).derivative_valid)
    matrix = jnp.eye(3, dtype=jnp.float64)
    for function in (
        hermitian_log,
        hermitian_exp,
        hermitian_sqrt,
        hermitian_inverse_sqrt,
    ):
        assert bool(function(matrix).derivative_valid)
        for diagnostic in (
            "eigenvalues",
            "eigenvectors",
            "condition_number",
            "minimum_gap",
        ):

            def evidence(
                a: Array,
                function: MatrixFunction = function,
                diagnostic: str = diagnostic,
            ) -> Array:
                spectrum = function(a).spectrum
                match diagnostic:
                    case "eigenvalues":
                        return spectrum.eigenvalues
                    case "eigenvectors":
                        return spectrum.eigenvectors
                    case "condition_number":
                        return spectrum.condition_number
                    case _:
                        return spectrum.minimum_gap

            _, tangent = jax.jvp(evidence, (matrix,), (jnp.ones_like(matrix),))
            np.testing.assert_array_equal(tangent, jnp.zeros_like(tangent))


def _cube_map(eigenvalues: Array, parameters: Array) -> tuple[Array, Invariants]:
    principal = jnp.stack((parameters[0] * eigenvalues**3, parameters[1] * eigenvalues))
    return principal, {
        "energy": jnp.sum(eigenvalues**2) * parameters[0],
        "branch": jnp.asarray(1, dtype=jnp.int32),
        "active": jnp.asarray(True, dtype=jnp.bool_),
    }


@pytest.mark.parametrize("eigenvalues", SPECTRA)
def test_spectral_map_confluent_cube_and_parameter_gradients(
    eigenvalues: tuple[float, ...],
) -> None:
    matrix = _matrix(eigenvalues, jnp.float64)
    parameters = jnp.asarray((1.3, 0.7), dtype=jnp.float64)
    direction = (_direction(jnp.float64) + _direction(jnp.float64).T) / 2

    def values(a: Array, p: Array) -> Array:
        return hermitian_spectral_map(a, _cube_map, p, function_id="cube-test").values

    result = hermitian_spectral_map(
        matrix, _cube_map, parameters, function_id="cube-test"
    )
    np.testing.assert_allclose(
        result.values[0],
        parameters[0] * (matrix @ matrix @ matrix),
        rtol=2e-12,
        atol=2e-12,
    )
    _, tangent = jax.jvp(
        values, (matrix, parameters), (direction, jnp.zeros_like(parameters))
    )
    expected = parameters[0] * (
        direction @ matrix @ matrix
        + matrix @ direction @ matrix
        + matrix @ matrix @ direction
    )
    np.testing.assert_allclose(tangent[0], expected, rtol=2e-8, atol=2e-8)
    for argument in (0, 1):
        np.testing.assert_allclose(
            jax.jit(jax.jacfwd(values, argnums=argument))(matrix, parameters),
            jax.jit(jax.jacrev(values, argnums=argument))(matrix, parameters),
            rtol=2e-8,
            atol=2e-8,
        )
    gradient = jax.grad(lambda p: jnp.sum(values(matrix, p)))(parameters)
    np.testing.assert_allclose(
        gradient,
        jnp.stack((jnp.sum(matrix @ matrix @ matrix), jnp.sum(matrix))),
        rtol=2e-12,
        atol=2e-12,
    )
    _, invariant_tangent = jax.jvp(
        lambda a: (
            hermitian_spectral_map(
                a, _cube_map, parameters, function_id="cube-test"
            ).invariants
        ),
        (matrix,),
        (direction,),
    )
    assert invariant_tangent["branch"].dtype == jax.dtypes.float0
    assert invariant_tangent["active"].dtype == jax.dtypes.float0
    np.testing.assert_allclose(
        invariant_tangent["energy"],
        2 * parameters[0] * jnp.trace(matrix @ direction),
        rtol=2e-12,
        atol=2e-12,
    )


_SCALAR_ROOT = LocalRootPlan(tolerance=1e-12, plan_id="spectral-scalar-test")
_VECTOR_ROOT = VectorLocalRootPlan(2, tolerance=1e-12, plan_id="spectral-vector-test")


def _scalar_root_map(eigenvalues: Array, parameters: Array) -> tuple[Array, Invariants]:
    target = jnp.mean(eigenvalues) + parameters[0]

    def residual(root: Array) -> Array:
        return root + root**3 - target

    root = _SCALAR_ROOT.solve(residual, jnp.asarray(0.5, dtype=eigenvalues.dtype))
    active = target > parameters[1]
    scale = jnp.where(active, root, target)
    principal = jnp.stack((scale * eigenvalues, eigenvalues**3 + scale))
    return principal, {
        "root": scale,
        "branch": active.astype(jnp.int32),
        "active": active,
    }


def _vector_root_map(eigenvalues: Array, parameters: Array) -> tuple[Array, Invariants]:
    targets = jnp.stack(
        (jnp.mean(eigenvalues) + parameters[0], jnp.mean(eigenvalues**2) + parameters[1])
    )

    def residual(root: Array) -> Array:
        return jnp.stack(
            (
                root[0] + root[0] ** 3 + 0.1 * root[1] - targets[0],
                root[1] + root[1] ** 3 + 0.1 * root[0] - targets[1],
            )
        )

    root = _VECTOR_ROOT.solve(residual, jnp.full((2,), 0.5, dtype=eigenvalues.dtype))
    active = targets[0] > parameters[1]
    scale = jax.lax.cond(active, lambda _: root[0], lambda _: targets[0], None)
    return jnp.stack((scale * eigenvalues, root[1] * eigenvalues**2)), {
        "root": jnp.sum(root),
        "branch": active.astype(jnp.int32),
        "active": active,
    }


@pytest.mark.parametrize(
    "eigenvalue_map",
    (_cube_map, _scalar_root_map, _vector_root_map),
    ids=("cube", "scalar-root", "vector-root"),
)
def test_principal_map_permutation_contract(eigenvalue_map: PrincipalMap) -> None:
    eigenvalues = jnp.asarray((0.8, 1.4, 2.2), dtype=jnp.float64)
    parameters = jnp.asarray((0.2, 0.3), dtype=jnp.float64)
    principal, invariants = eigenvalue_map(eigenvalues, parameters)
    permutation = jnp.asarray((2, 0, 1), dtype=jnp.int32)
    permuted, permuted_invariants = eigenvalue_map(eigenvalues[permutation], parameters)
    np.testing.assert_allclose(
        permuted, principal[:, permutation], rtol=2e-12, atol=2e-12
    )
    for key in invariants:
        np.testing.assert_allclose(
            permuted_invariants[key], invariants[key], rtol=2e-12, atol=2e-12
        )


@pytest.mark.parametrize(
    "eigenvalue_map",
    (_scalar_root_map, _vector_root_map),
    ids=("scalar-root", "vector-root"),
)
@pytest.mark.parametrize("active", (False, True), ids=("inactive", "active"))
@pytest.mark.parametrize("eigenvalues", SPECTRA)
def test_local_roots_compose_with_spectral_ad(
    eigenvalue_map: PrincipalMap, active: bool, eigenvalues: tuple[float, ...]
) -> None:
    matrix = _matrix(eigenvalues, jnp.float64)
    parameters = jnp.asarray((0.2, 0.3 if active else 4.0), dtype=jnp.float64)

    def values(a: Array, p: Array) -> Array:
        result = hermitian_spectral_map(
            a, eigenvalue_map, p, function_id="root-composition-test"
        )
        return jnp.concatenate(
            (result.values.reshape(-1), jnp.reshape(result.invariants["root"], (1,)))
        )

    result = hermitian_spectral_map(
        matrix, eigenvalue_map, parameters, function_id="root-composition-test"
    )
    assert result.values.shape == (2, 3, 3)
    assert int(result.invariants["branch"]) == int(active)
    assert bool(result.valid) and bool(result.derivative_valid)
    for argument in (0, 1):
        forward = jax.jit(jax.jacfwd(values, argnums=argument))(matrix, parameters)
        reverse = jax.jit(jax.jacrev(values, argnums=argument))(matrix, parameters)
        assert bool(jnp.all(jnp.isfinite(forward)))
        np.testing.assert_allclose(forward, reverse, rtol=2e-8, atol=2e-8)
    step = 1e-5
    for index in range(2):
        direction = jnp.eye(2, dtype=jnp.float64)[index]
        finite_difference = (
            values(matrix, parameters + step * direction)
            - values(matrix, parameters - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(
            jax.jacfwd(values, argnums=1)(matrix, parameters)[:, index],
            finite_difference,
            rtol=2e-7,
            atol=2e-8,
        )
    direction = (_direction(jnp.float64) + _direction(jnp.float64).T) / 2
    _, tangent = jax.jvp(
        values, (matrix, parameters), (direction, jnp.zeros_like(parameters))
    )
    finite_difference = (
        values(matrix + step * direction, parameters)
        - values(matrix - step * direction, parameters)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-7, atol=2e-8)
    for argument in (0, 1):
        gradient = jax.jit(
            jax.grad(lambda a, p: jnp.sum(values(a, p)), argnums=argument)
        )(matrix, parameters)
        assert bool(jnp.all(jnp.isfinite(gradient)))
    batch = jnp.stack((matrix, matrix * 1.1))
    mapped = jax.jit(jax.vmap(values, in_axes=(0, None)))(batch, parameters)
    batched = hermitian_spectral_map(
        batch, eigenvalue_map, parameters, function_id="root-composition-test"
    )
    np.testing.assert_allclose(
        mapped[:, :-1].reshape((2, 2, 3, 3)), batched.values, rtol=2e-12, atol=2e-12
    )
    _, invariants = jax.jvp(
        lambda p: (
            hermitian_spectral_map(
                matrix, eigenvalue_map, p, function_id="root-composition-test"
            ).invariants
        ),
        (parameters,),
        (jnp.ones_like(parameters),),
    )
    assert invariants["branch"].dtype == jax.dtypes.float0
    assert invariants["active"].dtype == jax.dtypes.float0


def test_complex_spectral_map_and_external_parameter_vmap() -> None:
    matrix = jnp.asarray(((1.4, 0.2j), (-0.2j, 1.4)), dtype=jnp.complex128)
    parameters = jnp.asarray((1.3, 0.7), dtype=jnp.float64)

    def values(a: Array, p: Array) -> Array:
        return hermitian_spectral_map(a, _cube_map, p, function_id="complex-cube").values

    direction = jnp.asarray(((0.2, 0.1j), (-0.1j, -0.3)), dtype=jnp.complex128)
    _, tangent = jax.jvp(
        values, (matrix, parameters), (direction, jnp.zeros_like(parameters))
    )
    expected = parameters[0].astype(matrix.dtype) * (
        direction @ matrix @ matrix
        + matrix @ direction @ matrix
        + matrix @ matrix @ direction
    )
    np.testing.assert_allclose(tangent[0], expected, rtol=2e-12, atol=2e-12)
    batch_parameters = jnp.stack((parameters, parameters * 2))
    mapped = jax.jit(jax.vmap(values, in_axes=(None, 0)))(matrix, batch_parameters)
    np.testing.assert_allclose(mapped[1], 2 * mapped[0], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("function,center,factor", FUNCTIONS)
def test_precision_policy_applies_to_primal_and_tangent(
    function: MatrixFunction, center: float, factor: float
) -> None:
    matrix = center * jnp.eye(3, dtype=jnp.float32)
    direction = _direction(jnp.float32)
    precision = HermitianPrecisionPolicy(
        compute_dtype=jnp.float64,
        factorization_dtype=jnp.float64,
        accumulation_dtype=jnp.float64,
        decision_dtype=jnp.float64,
        output_dtype=jnp.float32,
    )

    def value(a: Array) -> Array:
        return function(a, precision=precision).value

    primal, tangent = jax.jit(lambda a, e: jax.jvp(value, (a,), (e,)))(matrix, direction)
    assert primal.dtype == jnp.float32
    assert tangent.dtype == jnp.float32
    assert (
        function(matrix, precision=precision).spectrum.eigenvectors.dtype == jnp.float64
    )
    np.testing.assert_allclose(
        tangent, factor * (direction + direction.T) / 2, rtol=2e-6, atol=2e-6
    )


def test_spectral_map_batched_invariants_and_stopped_diagnostics() -> None:
    matrix = jnp.stack(
        (jnp.eye(3, dtype=jnp.float64), 2 * jnp.eye(3, dtype=jnp.float64))
    ).reshape((1, 2, 3, 3))
    parameters = jnp.asarray((1.3, 0.7), dtype=jnp.float64)
    result = hermitian_spectral_map(
        matrix, _cube_map, parameters, function_id="batch-evidence"
    )
    assert result.values.shape == (1, 2, 2, 3, 3)
    assert result.invariants["energy"].shape == (1, 2)
    assert result.invariants["branch"].dtype == jnp.int32
    assert result.invariants["active"].dtype == jnp.bool_
    assert result.valid.shape == (1, 2)

    def diagnostics(a: Array) -> tuple[Array, Array]:
        spectrum = hermitian_spectral_map(
            a, _cube_map, parameters, function_id="batch-evidence"
        ).spectrum
        return spectrum.eigenvalues, spectrum.eigenvectors

    _, tangents = jax.jvp(diagnostics, (matrix,), (jnp.ones_like(matrix),))
    for tangent in tangents:
        np.testing.assert_array_equal(tangent, jnp.zeros_like(tangent))


@pytest.mark.parametrize("argument", (0, 1), ids=("matrix", "parameters"))
def test_local_root_spectral_map_reverse_through_time_steps(argument: int) -> None:
    matrix = jnp.eye(3, dtype=jnp.float64)
    parameters = jnp.asarray((0.1, 0.3), dtype=jnp.float64)

    def rollout(a: Array, p: Array) -> Array:
        def step(state: Array, _: None) -> tuple[Array, None]:
            result = hermitian_spectral_map(
                state, _vector_root_map, p, function_id="root-time-stepping"
            )
            return result.values[0], None

        final_state, _ = jax.lax.scan(step, a, None, length=3)
        return jnp.sum(final_state)

    forward = jax.jit(jax.jacfwd(rollout, argnums=argument))(matrix, parameters)
    reverse = jax.jit(jax.grad(rollout, argnums=argument))(matrix, parameters)
    assert bool(jnp.all(jnp.isfinite(reverse)))
    np.testing.assert_allclose(reverse, forward, rtol=2e-8, atol=2e-8)


def test_complete_spectral_results_support_jit_and_vmap() -> None:
    matrix = jnp.eye(3, dtype=jnp.float64)
    parameters = jnp.asarray((1.3, 0.7), dtype=jnp.float64)
    for function in (
        hermitian_log,
        hermitian_exp,
        hermitian_sqrt,
        hermitian_inverse_sqrt,
    ):
        result = jax.jit(jax.vmap(function))(jnp.stack((matrix, matrix)))
        assert result.value.shape == (2, 3, 3)
        assert bool(jnp.all(result.derivative_valid))
        assert result.spectrum.precision_evidence.domain == "hermitian-spectral"

    def evaluate(a: Array) -> HermitianSpectralMapResult[Invariants]:
        return hermitian_spectral_map(
            a, _cube_map, parameters, function_id="complete-result"
        )

    mapped = jax.jit(jax.vmap(evaluate))(jnp.stack((matrix, matrix)))
    assert mapped.values.shape == (2, 2, 3, 3)
    assert mapped.invariants["branch"].dtype == jnp.int32
    assert bool(jnp.all(mapped.derivative_valid))
    assert mapped.spectrum.precision_evidence.domain == "hermitian-spectral"


def test_rank_deficient_gram_square_root_value_and_evidence() -> None:
    generator = np.random.default_rng(319)
    factors = generator.normal(size=(32, 3, 2))
    gram = jnp.asarray(factors @ factors.transpose(0, 2, 1), dtype=jnp.float64)
    result = hermitian_sqrt(gram)
    assert bool(jnp.all(result.valid))
    assert not bool(jnp.any(result.derivative_valid))
    assert bool(jnp.any(result.spectrum.minimum_eigenvalue < 0))
    np.testing.assert_allclose(result.value @ result.value, gram, rtol=1e-12, atol=1e-12)
    rejected = hermitian_sqrt(jnp.diag(jnp.asarray((-1e-4, 1.0, 2.0))))
    assert not bool(rejected.valid)
    assert not bool(rejected.derivative_valid)
