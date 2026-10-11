#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from math import comb

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import AxisDomain, FourierBasisPlan, TensorSpectralPlan
from phydrax.discretization.spectral import (
    FourierDeRhamComplex,
    PeriodicLerayProjector,
    SphericalDeRhamComplex,
    SphericalSpectralPlan,
)
from phydrax.discretization.spectral._distributed import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
)
from phydrax.linalg import HarmonicSubspace, HodgeDecompositionPolicy


def _complex(count: int = 5, dimension: int = 2) -> FourierDeRhamComplex:
    space = TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(dimension)),
        axis_names=tuple(f"x{axis}" for axis in range(dimension)),
    ).prepare(tuple(AxisDomain.periodic(0.0, 2.0 * np.pi) for _ in range(dimension)))
    return FourierDeRhamComplex(space, nyquist_policy="zero-self-conjugate")


@pytest.mark.parametrize("dimension", [2, 3])
def test_fourier_exterior_derivative_has_positive_ik_and_is_nilpotent(
    dimension: int,
) -> None:
    realization = _complex(5, dimension)
    modal = np.zeros(realization.space.modal_shape + (1,), dtype=np.complex128)
    modal[(1,) + (0,) * (dimension - 1) + (0,)] = 2.0 - 0.3j
    scalar = realization.from_modal(0, modal)
    gradient = eqx.filter_jit(realization.exterior_derivative)(0, scalar)
    expected = np.zeros(realization.space.modal_shape + (dimension,), dtype=np.complex128)
    expected[(1,) + (0,) * (dimension - 1) + (0,)] = 1j * (2.0 - 0.3j)
    np.testing.assert_allclose(realization.to_modal(1, gradient), expected, atol=1e-13)
    np.testing.assert_allclose(
        realization.exterior_derivative(1, gradient), 0.0, atol=1e-13
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_fourier_all_degree_parseval_laplacian_and_harmonics(degree: int) -> None:
    realization = _complex(3, 3)
    space = realization.hilbert_complex().space(degree)
    values = (jnp.sin(jnp.arange(space.size, dtype=jnp.float64)) + 0.2j).astype(
        jnp.complex128
    )
    numbers = np.fft.fftfreq(3) * 3
    wave = np.stack(
        np.meshgrid(numbers, numbers, numbers, indexing="ij"), axis=-1
    ).reshape((-1, 3))
    squared = np.repeat(np.sum(wave**2, axis=-1), comb(3, degree))
    np.testing.assert_allclose(
        realization.hodge_laplacian(degree, values), squared * values, atol=2e-13
    )
    np.testing.assert_allclose(
        space.inner(values, values), np.vdot(values, values), atol=2e-13
    )
    basis = realization.harmonic_basis(degree)
    np.testing.assert_allclose(
        basis.conj().T @ basis, np.eye(comb(3, degree)), atol=1e-13
    )
    assert np.count_nonzero(squared == 0.0) == realization.betti_numbers[degree]
    star = realization.hodge_star(degree, values)
    np.testing.assert_allclose(
        realization.inverse_hodge_star(degree, star), values, atol=1e-13
    )
    np.testing.assert_allclose(
        realization.metric_star(3 - degree, realization.metric_star(degree, values)),
        (-1) ** (degree * (3 - degree)) * values,
        atol=1e-13,
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_fourier_complex_hilbert_duality_and_hirani_dual_derivative(degree: int) -> None:
    realization = _complex()
    complex_ = realization.hilbert_complex()
    source, target = complex_.space(degree - 1), complex_.space(degree)
    left = (jnp.arange(source.size, dtype=jnp.float64) * (0.03 + 0.1j)).astype(
        jnp.complex128
    )
    right = jnp.exp(0.13j * jnp.arange(target.size, dtype=jnp.float64)).astype(
        jnp.complex128
    )
    derivative = realization.exterior_derivative(degree - 1, left)
    adjoint = realization.codifferential(degree, right)
    np.testing.assert_allclose(
        target.inner(derivative, right), source.inner(left, adjoint), atol=1e-11
    )
    dual = realization.dual_exterior_derivative(
        degree, realization.hodge_star(degree, right)
    )
    np.testing.assert_allclose(
        (-1) ** degree * realization.inverse_hodge_star(degree - 1, dual),
        adjoint,
        atol=1e-13,
    )


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_fourier_decomposition_reconstructs_orthogonal_sectors(degree: int) -> None:
    realization = _complex()
    space = realization.hilbert_complex().space(degree)
    value = jnp.exp(0.21j * jnp.arange(space.size, dtype=jnp.float64)).astype(
        jnp.complex128
    )
    result = eqx.filter_jit(realization.hodge_decomposition)(degree, value)
    assert bool(result.valid)
    np.testing.assert_allclose(
        result.exact + result.coexact + result.harmonic, value, atol=2e-13
    )
    np.testing.assert_allclose(space.inner(result.exact, result.coexact), 0.0, atol=2e-12)
    np.testing.assert_allclose(
        realization.hodge_laplacian(degree, result.harmonic), 0.0, atol=1e-13
    )
    if degree > 0:
        assert result.exact_potential is not None
        np.testing.assert_allclose(
            realization.exterior_derivative(degree - 1, result.exact_potential),
            result.exact,
            atol=2e-13,
        )


def test_fourier_nyquist_admission_matches_leray_and_recovers_pressure() -> None:
    realization = _complex(6)
    modal = jnp.ones(realization.space.modal_shape + (2,), dtype=jnp.complex128)
    owner = PeriodicLerayProjector(realization.space)
    admitted = realization.from_modal(1, modal)
    np.testing.assert_allclose(
        realization.to_modal(1, admitted), owner.zero_forbidden_modes(modal), atol=0.0
    )
    decomposition = realization.hodge_decomposition(1, admitted)
    np.testing.assert_allclose(
        realization.to_modal(1, decomposition.coexact + decomposition.harmonic),
        owner.project(modal),
        atol=2e-13,
    )
    pressure = owner.pressure_from_unconstrained_rhs(modal)
    gradient = realization.exterior_derivative(
        0, realization.from_modal(0, pressure[..., None])
    )
    np.testing.assert_allclose(gradient, decomposition.exact, atol=2e-13)
    assert realization.cell_counts[0] == 25
    with pytest.raises(ValueError, match="relative"):
        realization.hilbert_complex(boundary="relative")


def test_fourier_exterior_derivative_preserves_distributed_modal_parity() -> None:
    realization = _complex(6, 3)
    distributed = DistributedSpectralExecutionPlan.from_discretization(
        SpectralMeshTopology.one_device(),
        realization.space,
        admitted_payload_shapes=((),),
    )
    assert distributed.owner_id == realization.space.prepared_id
    assert distributed.precision.policy_id == realization.space.plan.precision.policy_id
    assert distributed.precision.coefficient_dtype == "complex128"
    assert distributed.numerical_id != distributed.execution_id
    assert distributed.plan_id not in (
        distributed.numerical_id,
        distributed.execution_id,
    )
    modal = (
        jnp.exp(0.11j * jnp.arange(216, dtype=jnp.float64))
        .reshape((6, 6, 6))
        .astype(jnp.complex128)
    )
    scalar = realization.from_modal(0, modal[..., None])
    admitted = realization.to_modal(0, scalar)[..., 0]
    gradient = realization.to_modal(1, realization.exterior_derivative(0, scalar))
    owner_gradient = jnp.stack(
        tuple(distributed.modal_derivative(admitted, axis) for axis in range(3)), axis=-1
    )
    np.testing.assert_allclose(gradient, owner_gradient, atol=2e-13)
    np.testing.assert_allclose(
        distributed.to_physical(admitted),
        realization.space.reconstruct(admitted, real_output=False),
        atol=2e-12,
    )


def test_fourier_one_form_curl_has_zero_exterior_derivative_in_three_dimensions() -> None:
    realization = _complex(3, 3)
    space = realization.hilbert_complex().space(1)
    value = jnp.exp(0.17j * jnp.arange(space.size, dtype=jnp.float64)).astype(
        jnp.complex128
    )
    curl = realization.exterior_derivative(1, value)
    assert np.linalg.norm(np.asarray(curl)) > 1.0
    np.testing.assert_allclose(realization.exterior_derivative(2, curl), 0.0, atol=2e-13)


def _harmonic_artifact(
    realization: FourierDeRhamComplex | SphericalDeRhamComplex,
    degree: int,
    basis: Array,
    *,
    complex_id: str | None = None,
) -> HarmonicSubspace:
    return HarmonicSubspace(
        realization.hilbert_complex().complex_id if complex_id is None else complex_id,
        degree,
        basis,
        jnp.zeros((basis.shape[1],), dtype=jnp.float64),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(True),
        jnp.asarray(True),
    )


@pytest.mark.parametrize("spherical", [False, True])
def test_analytic_decomposition_admits_and_uses_supplied_harmonic_frames(
    spherical: bool,
) -> None:
    realization = (
        SphericalDeRhamComplex(
            SphericalSpectralPlan(3, sampling="gl").prepare(radius=1.7)
        )
        if spherical
        else _complex(3)
    )
    degree = 2
    space = realization.hilbert_complex().space(degree)
    value = jnp.cos(jnp.arange(space.size, dtype=jnp.float64)).astype(
        space.structure().dtype
    )
    harmonic = _harmonic_artifact(
        realization, degree, -realization.harmonic_basis(degree)
    )
    lower = _harmonic_artifact(
        realization, degree - 1, realization.harmonic_basis(degree - 1)
    )
    result = eqx.filter_jit(realization.hodge_decomposition)(
        degree, value, harmonic=harmonic, lower_harmonic=lower
    )
    assert bool(result.valid)
    assert result.exact_potential is not None
    np.testing.assert_allclose(
        result.exact + result.coexact + result.harmonic, value, atol=1e-12
    )
    np.testing.assert_allclose(
        realization.exterior_derivative(degree - 1, result.exact_potential),
        result.exact,
        atol=1e-12,
    )
    expected = harmonic.basis @ (
        harmonic.basis.conj().T @ realization.hodge_star(degree, value)
    )
    np.testing.assert_allclose(result.harmonic, expected, atol=1e-12)


def test_analytic_decomposition_refuses_harmonic_provenance_and_false_kernel() -> None:
    realization = _complex(3)
    space = realization.hilbert_complex().space(1)
    value = jnp.ones((space.size,), dtype=space.structure().dtype)
    basis = realization.harmonic_basis(1)
    foreign = _harmonic_artifact(realization, 1, basis, complex_id="foreign-complex")
    with pytest.raises(ValueError, match="different complex"):
        realization.hodge_decomposition(1, value, harmonic=foreign)
    contaminated = basis.at[2, 0].set(1.0)
    invalid = _harmonic_artifact(realization, 1, contaminated)
    with pytest.raises(eqx.EquinoxRuntimeError, match="orthonormal spectral kernel"):
        eqx.filter_jit(realization.hodge_decomposition)(
            1, value, harmonic=invalid
        ).valid.block_until_ready()


@pytest.mark.parametrize("spherical", [False, True])
def test_analytic_decomposition_explicitly_refuses_iterative_policy(
    spherical: bool,
) -> None:
    realization = (
        SphericalDeRhamComplex(SphericalSpectralPlan(3, sampling="gl").prepare())
        if spherical
        else _complex(3)
    )
    space = realization.hilbert_complex().space(0)
    value = jnp.zeros((space.size,), dtype=space.structure().dtype)
    with pytest.raises(ValueError, match="no iterative solve policy"):
        realization.hodge_decomposition(0, value, policy=HodgeDecompositionPolicy())
