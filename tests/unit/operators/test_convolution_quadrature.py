#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from phydrax.linalg import DenseLinearOperator, LinearSystem, prepare
from phydrax.operators.integral._convolution_quadrature import (
    bdf_symbol,
    convolution_quadrature_fft,
    convolution_quadrature_ifft,
    ConvolutionQuadratureContourPolicy,
    prepare_convolution_quadrature_contour,
)
from phydrax.solver._convolution_quadrature import (
    apply_convolution_quadrature,
    ConvolutionQuadratureDeclaration,
    ConvolutionQuadratureStatus,
    prepare_convolution_quadrature,
)


def _declaration(dimension: int, *, provider: str = "phydrax.linalg"):
    return ConvolutionQuadratureDeclaration(
        dimension,
        family_id="test-dynamic-transfer",
        pde="caller-supplied scalar or finite-dimensional Laplace-domain model",
        geometry="fixed caller-supplied discrete geometry",
        formulation="BDF convolution quadrature of prepared transfer solves",
        provider=provider,
        precision="complex128 node algebra under jax.enable_x64",
        non_goals=("continuum certification", "physics-kernel construction"),
    )


def _system_factory(
    matrix: Callable[[jax.Array], jax.Array],
    *,
    singular_first_forward: bool = False,
):
    calls = {"forward": 0, "transpose": 0, "adjoint": 0}

    def factory(parameter, action):
        base = jnp.asarray(matrix(parameter))
        call_index = calls[action]
        calls[action] += 1
        if singular_first_forward and action == "forward" and call_index == 0:
            base = jnp.zeros_like(base)
        if action == "transpose":
            base = jnp.swapaxes(base, -1, -2)
        elif action == "adjoint":
            base = jnp.conj(jnp.swapaxes(base, -1, -2))
        operator = DenseLinearOperator(base)
        return prepare(LinearSystem(operator))

    return factory, calls


def _scalar_factory(rate: float, *, singular_first_forward: bool = False):
    return _system_factory(
        lambda parameter: jnp.reshape(parameter + rate, (1, 1)),
        singular_first_forward=singular_first_forward,
    )


def test_bdf_symbols_and_balanced_radius_policy_are_explicit():
    zeta = jnp.asarray([0.0 + 0.0j, 0.2 - 0.3j])
    assert jnp.allclose(bdf_symbol(zeta, "bdf1"), 1.0 - zeta)
    assert jnp.allclose(
        bdf_symbol(zeta, "bdf2"),
        1.5 - 2.0 * zeta + 0.5 * zeta * zeta,
    )

    policy = ConvolutionQuadratureContourPolicy(tolerance=1.0e-10)
    radius = policy.resolve(32)
    assert radius**64 == pytest.approx(1.0e-10)
    explicit = ConvolutionQuadratureContourPolicy(radius=0.82, tolerance=1.0e-8)
    assert explicit.resolve(128) == 0.82


def test_contour_fft_round_trip_retains_every_history_sample():
    with jax.enable_x64():
        contour = prepare_convolution_quadrature_contour(
            0.125,
            5,
            method="bdf2",
            fft_length=16,
            policy=ConvolutionQuadratureContourPolicy(tolerance=1.0e-14),
        )
        history = jnp.asarray(
            [
                [0.2 + 0.1j, -0.4j],
                [0.7 - 0.3j, 0.1 + 0.2j],
                [-0.2 + 0.8j, 0.9],
                [0.0, -0.6 + 0.1j],
                [1.3 - 0.4j, -0.2 - 0.3j],
            ],
            dtype=jnp.complex128,
        )
        transformed = convolution_quadrature_fft(history, contour)
        recovered = convolution_quadrature_ifft(transformed, contour)

    assert transformed.shape == (16, 2)
    assert jnp.allclose(recovered, history, rtol=2.0e-13, atol=2.0e-13)
    assert not contour.conjugate_symmetric


def test_scalar_transfer_matches_direct_cq_weight_oracle():
    with jax.enable_x64():
        factory, _ = _scalar_factory(0.7)
        prepared = prepare_convolution_quadrature(
            factory,
            0.1,
            6,
            _declaration(1),
            method="bdf2",
            fft_length=16,
            contour_policy=ConvolutionQuadratureContourPolicy(tolerance=1.0e-14),
        )
        history = jnp.asarray([[1.0], [-0.2], [0.5], [0.0], [0.8], [-0.1]])
        result = prepared.apply(history)
        transfer_samples = 1.0 / (prepared.contour.parameters + 0.7)
        coefficients = jnp.fft.ifft(transfer_samples)
        indices = jnp.arange(history.shape[0])
        weights = coefficients[: history.shape[0]] * prepared.contour.radius ** (-indices)
        expected = jnp.stack(
            tuple(
                jnp.sum(weights[: index + 1, None] * history[index::-1], axis=0)
                for index in range(history.shape[0])
            )
        )

    assert bool(result.successful)
    assert jnp.allclose(result.value, expected, rtol=2.0e-12, atol=2.0e-12)
    assert len(result.node_results) == prepared.contour.fft_length


def _terminal_smooth_forcing_error(method: str, steps: int) -> float:
    step_size = 1.0 / steps
    history_length = steps + 1
    factory, _ = _scalar_factory(1.0)
    prepared = prepare_convolution_quadrature(
        factory,
        step_size,
        history_length,
        _declaration(1),
        method=method,
        fft_length=2 * history_length,
        contour_policy=ConvolutionQuadratureContourPolicy(tolerance=1.0e-14),
        conjugate_symmetric=True,
    )
    times = jnp.arange(history_length, dtype=jnp.float64) * step_size
    history = jnp.square(times)[:, None]
    terminal = prepared.apply(history).value[-1, 0]
    exact = 1.0 - 2.0 / jnp.e
    return float(jnp.abs(terminal - exact))


def test_bdf1_and_bdf2_show_their_expected_convergence_orders():
    with jax.enable_x64():
        bdf1_coarse = _terminal_smooth_forcing_error("bdf1", 24)
        bdf1_fine = _terminal_smooth_forcing_error("bdf1", 48)
        bdf2_coarse = _terminal_smooth_forcing_error("bdf2", 24)
        bdf2_fine = _terminal_smooth_forcing_error("bdf2", 48)

    assert bdf1_coarse / bdf1_fine > 1.7
    assert bdf2_coarse / bdf2_fine > 3.2
    assert bdf2_fine < bdf1_fine


def test_conjugacy_reduction_matches_full_contour_and_rejects_complex_history():
    with jax.enable_x64():
        full_factory, full_calls = _scalar_factory(0.4)
        reduced_factory, reduced_calls = _scalar_factory(0.4)
        common = dict(
            method="bdf2",
            fft_length=16,
            contour_policy=ConvolutionQuadratureContourPolicy(tolerance=1.0e-14),
        )
        full = prepare_convolution_quadrature(
            full_factory,
            0.08,
            7,
            _declaration(1),
            conjugate_symmetric=False,
            **common,
        )
        reduced = prepare_convolution_quadrature(
            reduced_factory,
            0.08,
            7,
            _declaration(1),
            conjugate_symmetric=True,
            **common,
        )
        history = jnp.linspace(-0.3, 1.1, 7, dtype=jnp.float64)[:, None]
        full_result = full.apply(history)
        reduced_result = reduced.apply(history)

    assert jnp.allclose(reduced_result.value, full_result.value.real, atol=2.0e-12)
    assert full_calls == {"forward": 16, "transpose": 16, "adjoint": 16}
    assert reduced_calls == {"forward": 9, "transpose": 9, "adjoint": 0}
    assert reduced.resource_evidence.solved_node_count_per_action == 9
    with pytest.raises(TypeError, match="real history"):
        reduced.apply(history.astype(jnp.complex128))


def test_causal_reconstruction_cannot_see_future_samples_or_truncate_old_ones():
    with jax.enable_x64():
        factory, _ = _scalar_factory(0.25)
        prepared = prepare_convolution_quadrature(
            factory,
            0.05,
            8,
            _declaration(1),
            fft_length=16,
            conjugate_symmetric=True,
        )
        early = jnp.asarray([1.0, -0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0])[:, None]
        changed_future = early.at[4:, 0].set(jnp.asarray([4.0, -3.0, 2.0, 1.0]))
        first = prepared.apply(early)
        second = prepared.apply(changed_future)

    assert jnp.allclose(first.value[:4], second.value[:4], rtol=0.0, atol=2.0e-12)
    assert jnp.abs(first.value[-1, 0]) > 0.0
    assert not prepared.resource_evidence.history_truncated
    assert not first.error_evidence.history_truncated


def test_any_failed_node_invalidates_the_complete_history_and_retains_results():
    with jax.enable_x64():
        factory, _ = _scalar_factory(0.5, singular_first_forward=True)
        prepared = prepare_convolution_quadrature(
            factory,
            0.1,
            5,
            _declaration(1),
            fft_length=16,
        )
        result = prepared.apply(jnp.ones((5, 1), dtype=jnp.float64))

    assert int(result.status) == int(ConvolutionQuadratureStatus.NODE_SOLVE_FAILED)
    assert not bool(result.successful)
    assert jnp.all(result.value == 0.0)
    assert len(result.node_results) == prepared.contour.fft_length
    assert jnp.any(result.error_evidence.node_statuses != 0)
    assert not bool(result.error_evidence.node_solves_successful)


def test_total_history_transpose_and_adjoint_are_exact_for_batched_rhs_axes():
    with jax.enable_x64():
        base = jnp.asarray([[0.8 + 0.2j, -0.3 + 0.1j], [0.4j, 1.1 - 0.2j]])
        factory, _ = _system_factory(
            lambda parameter: parameter * jnp.eye(2, dtype=jnp.complex128) + base
        )
        prepared = prepare_convolution_quadrature(
            factory,
            0.07,
            5,
            _declaration(2),
            method="bdf2",
            fft_length=16,
            contour_policy=ConvolutionQuadratureContourPolicy(tolerance=1.0e-14),
        )
        real = jnp.arange(30, dtype=jnp.float64).reshape(5, 2, 3) / 17.0
        imag = jnp.flip(real, axis=0) / 11.0
        x = real + 1.0j * imag
        y = (0.3 - 0.2j) * jnp.flip(x, axis=1) + 0.1
        forward = prepared.apply(x)
        transposed = prepared.transpose(y)
        adjointed = prepared.adjoint(y)
        bilinear_left = jnp.sum(forward.value * y)
        bilinear_right = jnp.sum(x * transposed.value)
        hermitian_left = jnp.vdot(forward.value, y)
        hermitian_right = jnp.vdot(x, adjointed.value)

    assert forward.value.shape == (5, 2, 3)
    assert transposed.value.shape == x.shape
    assert adjointed.value.shape == x.shape
    assert forward.right_hand_side_count == 3
    assert jnp.allclose(bilinear_left, bilinear_right, rtol=3.0e-11, atol=3.0e-11)
    assert jnp.allclose(hermitian_left, hermitian_right, rtol=3.0e-11, atol=3.0e-11)
    assert adjointed.parameter_indices == tuple((-index) % 16 for index in range(16))


def test_resource_error_and_scientific_scope_evidence_are_retained():
    with jax.enable_x64():
        factory, _ = _scalar_factory(0.9)
        prepared = prepare_convolution_quadrature(
            factory,
            0.1,
            4,
            _declaration(1, provider="phydrax.linalg checked solve"),
            fft_length=8,
            conjugate_symmetric=True,
        )
        result = apply_convolution_quadrature(
            prepared,
            jnp.ones((4, 1), dtype=jnp.float64),
        )

    resources = result.resource_evidence
    assert resources.contour_node_count == 8
    assert resources.solved_node_count_per_action == 5
    assert resources.retained_prepared_solve_count == 10
    assert resources.retained_array_bytes > 0
    assert resources.controller_workspace_upper_bound_bytes_per_rhs > 0
    assert resources.node_right_hand_sides_per_external_rhs == 4
    assert not resources.provider_opaque_allocations_included
    assert result.declaration.dimension == 1
    assert result.declaration.pde.startswith("caller-supplied")
    assert result.declaration.geometry == "fixed caller-supplied discrete geometry"
    assert "continuum certification" in result.declaration.non_goals
    assert not result.error_evidence.continuum_certified
    assert result.error_evidence.error_scope.startswith("checked discrete node solves")
    assert not prepared.continuum_certified
    assert prepared.error_scope.startswith("contour policy target")
    assert result.contour.method == "bdf2"
    assert len(result.node_providers) == 5
    assert set(result.node_precision_dtypes) == {"complex128"}
