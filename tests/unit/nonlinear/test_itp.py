#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native ITP arithmetic parity, bracket admission, and terminal evidence."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.geometry.implicit._discovery import isolate_roots
from phydrax.nonlinear import (
    ITP,
    NonlinearStatus,
    NonlinearTermination,
    ScalarRootProblem,
)
from phydrax.units import METER
from tests._support.itp_reference import reference_edge_roots


pytestmark = pytest.mark.strict_jax
_EPSILON = np.finfo(np.float64).eps
_RESIDUAL_TOLERANCE = 16 * _EPSILON
# For x^3 - 2 on [1, 2], f' >= 3. The residual allowance contributes
# tolerance/3 root error, and the independent binary64 cbrt reference adds 2 eps.
_ROOT_ERROR = _RESIDUAL_TOLERANCE / 3 + 2 * _EPSILON


def _cubic(state: Array, args: None) -> Array:
    del args
    return state**3 - 2.0


@pytest.mark.parametrize(
    "tolerance",
    (1e-10, 1e-12, 0.0),
    ids=("residual-loose", "residual-tight", "bracket-only"),
)
def test_native_edge_roots_preserve_pre_cutover_binary64_bits(tolerance: float) -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="itp-parity"
    ).compile(coordinate_unit=METER)
    lower = jnp.asarray(
        ((0.1, 0.2, 0.3), (0.0, 0.1, 0.0), (0.25, -0.125, 0.0)), dtype=jnp.float64
    )
    upper = jnp.asarray(
        ((1.2, 0.7, 0.5), (0.0, 1.3, 0.0), (1.0, -0.5, 0.0)), dtype=jnp.float64
    )
    lower_values = geometry.kernel.boundary_field(geometry.state, lower)
    upper_values = geometry.kernel.boundary_field(geometry.state, upper)
    allowance = jnp.asarray(tolerance, dtype=jnp.float64)
    expected_points, expected_residual = reference_edge_roots(
        geometry.kernel,
        geometry.state,
        lower,
        upper,
        lower_values,
        upper_values,
        allowance,
    )
    isolation = isolate_roots(
        geometry.kernel,
        geometry.state,
        lower,
        upper,
        lower_values,
        upper_values,
        allowance,
    )
    points, residual = isolation.points, isolation.residuals
    np.testing.assert_array_equal(
        np.asarray(points).view(np.uint64), np.asarray(expected_points).view(np.uint64)
    )
    np.testing.assert_array_equal(
        np.asarray(residual).view(np.uint64),
        np.asarray(expected_residual).view(np.uint64),
    )
    expected_success = np.asarray(expected_residual) <= tolerance
    np.testing.assert_array_equal(
        np.asarray(isolation.native_result.successful), expected_success
    )
    assert isolation.native_result.nonlinear_result.provenance.method_id == "itp"
    assert (
        int(isolation.native_result.nonlinear_result.diagnostics.residual_evaluations)
        >= 3
    )


def test_scalar_itp_certifies_a_closed_form_root_and_bracket() -> None:
    problem = ScalarRootProblem(_cubic, bracket=(jnp.asarray(1.0), jnp.asarray(2.0)))
    result = ITP().solve(
        problem,
        termination=NonlinearTermination(
            absolute_residual=float(_RESIDUAL_TOLERANCE), relative_residual=0.0
        ),
    )
    assert int(result.status) == NonlinearStatus.SUCCESS
    assert bool(result.bracket_valid)
    assert float(result.lower) <= np.cbrt(2.0) <= float(result.upper)
    assert float(result.root) == pytest.approx(
        np.cbrt(2.0), abs=float(_ROOT_ERROR), rel=0.0
    )
    assert abs(float(result.value)) <= _RESIDUAL_TOLERANCE
    assert (
        int(result.nonlinear_result.diagnostics.residual_evaluations)
        == int(result.nonlinear_result.diagnostics.iterations) + 3
    )


def test_scalar_itp_accepts_a_root_at_a_bracket_endpoint() -> None:
    problem = ScalarRootProblem(
        lambda state, args: state + 1.0, bracket=(jnp.asarray(-1.0), jnp.asarray(1.0))
    )
    result = ITP().solve(
        problem,
        termination=NonlinearTermination(absolute_residual=0.0, relative_residual=0.0),
    )
    assert int(result.status) == NonlinearStatus.SUCCESS
    assert float(result.root) == -1.0
    assert float(result.value) == 0.0
    assert int(result.nonlinear_result.diagnostics.iterations) == 0


def test_scalar_itp_reports_a_non_bracketing_function() -> None:
    problem = ScalarRootProblem(
        lambda state, args: state**2 + 1.0, bracket=(jnp.asarray(-1.0), jnp.asarray(1.0))
    )
    result = ITP().solve(
        problem,
        termination=NonlinearTermination(absolute_residual=float(_RESIDUAL_TOLERANCE)),
    )
    assert int(result.status) == NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE
    assert not bool(result.successful)
    assert not bool(result.bracket_valid)
    assert int(result.nonlinear_result.diagnostics.domain_failures) == 1


def test_scalar_itp_retains_step_exhaustion_and_residual_evidence() -> None:
    problem = ScalarRootProblem(_cubic, bracket=(jnp.asarray(0.0), jnp.asarray(2.0)))
    result = ITP().solve(
        problem,
        termination=NonlinearTermination(
            absolute_residual=0.0, relative_residual=0.0, maximum_steps=1
        ),
    )
    assert int(result.status) == NonlinearStatus.MAXIMUM_STEPS_REACHED
    assert not bool(result.successful)
    assert bool(result.bracket_valid)
    assert int(result.nonlinear_result.diagnostics.iterations) == 1
    assert int(result.nonlinear_result.diagnostics.residual_evaluations) == 4
    assert float(result.value) != 0.0


def test_batched_itp_reports_nonfinite_callback_output() -> None:
    method = ITP()

    def nonfinite(parameter: Array) -> Array:
        return jnp.full_like(parameter, jnp.nan)

    result = eqx.filter_jit(method.solve_batched)(
        nonfinite,
        jnp.asarray((-1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(_RESIDUAL_TOLERANCE),
        jnp.asarray((0.0,)),
    )
    assert int(result.status[0]) == NonlinearStatus.NONFINITE_EVALUATION
    assert not bool(result.successful[0])
    assert int(result.nonlinear_result.diagnostics.nonfinite_trials[0]) == 1


def test_edge_root_publication_refuses_native_failure_with_finite_coordinates() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="itp-refusal"
    ).compile(coordinate_unit=METER)
    lower = jnp.asarray(((0.1, 0.0, 0.0),), dtype=jnp.float64)
    upper = jnp.asarray(((1.0, 0.0, 0.0),), dtype=jnp.float64)
    invalid_endpoint_values = jnp.ones((1,), dtype=jnp.float64)
    isolation = isolate_roots(
        geometry.kernel,
        geometry.state,
        lower,
        upper,
        invalid_endpoint_values,
        invalid_endpoint_values,
        jnp.asarray(_RESIDUAL_TOLERANCE, dtype=jnp.float64),
    )
    assert bool(jnp.all(jnp.isfinite(isolation.points)))
    assert (
        int(isolation.native_result.status[0])
        == NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE
    )
    with pytest.raises(ValueError, match="RootIsolationResult.native_result.status"):
        isolation.require_converged()
