#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent scalar scientific references for bulk and point-particle drag."""

from __future__ import annotations

from typing import assert_never

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.optimize import root_scalar

from phydrax.equations import (
    InterphaseDragCorrelation,
    InterphaseDragEvaluation,
    InterphaseDragLawPlan,
)


pytestmark = pytest.mark.strict_jax

_DIAMETER = 0.002
_VISCOSITY = 0.001
_DENSITY = 1000.0
_VOLUME = np.pi * _DIAMETER**3 / 6.0


def _evaluate(
    law: InterphaseDragLawPlan,
    porosity: float = 0.9,
    reynolds: float = 0.1,
    *,
    volume: float = _VOLUME,
) -> InterphaseDragEvaluation:
    speed = reynolds * _VISCOSITY / (_DENSITY * _DIAMETER)
    return law.evaluate(
        *(
            jnp.asarray(value, dtype=jnp.float64)
            for value in (
                porosity,
                speed,
                _DIAMETER,
                _VISCOSITY,
                _DENSITY,
                volume,
            )
        )
    )


def test_stokes_preserves_point_coefficient_for_non_spherical_volume() -> None:
    law = InterphaseDragLawPlan("stokes")
    result = _evaluate(law, 0.75, volume=7.0 * _VOLUME)
    radius = _DIAMETER / 2.0
    point_reference = 6.0 * np.pi * _VISCOSITY * radius
    bulk_reference = 18.0 * _VISCOSITY * (1.0 - 0.75) / _DIAMETER**2
    assert result.successful
    np.testing.assert_allclose(
        result.particle_drag_coefficient, point_reference, rtol=1e-13
    )
    np.testing.assert_allclose(result.beta, bulk_reference, rtol=1e-13)
    assert not np.isclose(
        result.particle_drag_coefficient,
        7.0 * _VOLUME * bulk_reference / (1.0 - 0.75),
        rtol=1e-3,
        atol=0.0,
    )


def test_stokes_zero_reynolds_has_exact_regular_derivative() -> None:
    law = InterphaseDragLawPlan("stokes")
    inputs = jnp.asarray(
        (0.75, 0.0, _DIAMETER, _VISCOSITY, _DENSITY, _VOLUME),
        dtype=jnp.float64,
    )

    def coefficients(values: Array) -> Array:
        result = law.evaluate(*values)
        return jnp.stack((result.beta, result.particle_drag_coefficient))

    result = law.evaluate(*inputs)
    assert result.successful
    assert result.derivative_valid
    assert result.reynolds_number == 0.0
    porosity_direction = jnp.asarray((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=jnp.float64)
    _, derivative = jax.jvp(coefficients, (inputs,), (porosity_direction,))
    np.testing.assert_allclose(
        derivative,
        np.asarray((-18.0 * _VISCOSITY / _DIAMETER**2, 0.0), dtype=np.float64),
        rtol=1e-13,
        atol=0.0,
    )
    speed_direction = jnp.asarray((0.0, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=jnp.float64)
    _, derivative = jax.jvp(coefficients, (inputs,), (speed_direction,))
    np.testing.assert_array_equal(derivative, np.zeros((2,), dtype=np.float64))


@pytest.mark.parametrize("reynolds", (1e-3, 1e-6, 1e-9), ids=("small", "tiny", "limit"))
def test_schiller_naumann_approaches_stokes_without_rate_floor(reynolds: float) -> None:
    result = _evaluate(InterphaseDragLawPlan("schiller_naumann"), 0.9, reynolds)
    stokes = 18.0 * _VISCOSITY * 0.1 / _DIAMETER**2
    correction = 0.15 * reynolds**0.687
    assert result.successful
    np.testing.assert_allclose(result.beta / stokes - 1.0, correction, rtol=3e-8)
    np.testing.assert_allclose(result.reynolds_number, reynolds, rtol=1e-13)


@pytest.mark.parametrize("porosity", (0.05, 0.35, 0.65), ids=("dense", "packed", "loose"))
@pytest.mark.parametrize("superficial_speed", (1e-6, 0.01), ids=("darcy", "inertial"))
def test_ergun_packed_column_pressure_gradient(
    porosity: float, superficial_speed: float
) -> None:
    """Ergun (1952), CEP 48:89–94, superficial-velocity pressure drop.

    The fluid-phase balance is epsilon * |grad p| = beta * intrinsic slip;
    superficial velocity is epsilon * intrinsic slip, not intrinsic slip.
    """
    solid = 1.0 - porosity
    intrinsic_speed = superficial_speed / porosity
    reynolds = _DENSITY * _DIAMETER * intrinsic_speed / _VISCOSITY
    result = _evaluate(InterphaseDragLawPlan("ergun"), porosity, reynolds)
    permeability = porosity**3 * _DIAMETER**2 / (150.0 * solid**2)
    reference_gradient = (
        _VISCOSITY * superficial_speed / permeability
        + 1.75 * _DENSITY * solid * superficial_speed** 2 / (porosity**3 * _DIAMETER)
    )
    assert result.successful
    assert result.admissible
    np.testing.assert_allclose(
        result.beta * intrinsic_speed / porosity, reference_gradient, rtol=2e-13
    )
    np.testing.assert_allclose(
        result.particle_drag_coefficient, _VOLUME * result.beta / solid, rtol=2e-13
    )


def test_schiller_naumann_terminal_velocity_matches_independent_host_root() -> None:
    """Buoyant weight balances SN drag; host root never calls the device law."""
    diameter = 0.001
    viscosity = 0.001
    density = 1000.0
    particle_density = 2500.0
    volume = np.pi * diameter**3 / 6.0
    buoyant_weight = (particle_density - density) * volume * 9.81

    def host_force_residual(speed: float) -> float:
        reynolds = density * diameter * speed / viscosity
        point_coefficient = (
            3.0 * np.pi * viscosity * diameter * (1.0 + 0.15 * reynolds**0.687)
        )
        return float(point_coefficient * speed - buoyant_weight)

    root = root_scalar(
        host_force_residual, bracket=(0.0, 0.5), method="brentq", xtol=1e-14
    )
    assert root.converged
    terminal_speed = root.root
    assert 1.0 < density * diameter * terminal_speed / viscosity < 1000.0
    law = InterphaseDragLawPlan("schiller_naumann")
    result = law.evaluate(
        *(
            jnp.asarray(value, dtype=jnp.float64)
            for value in (1.0, terminal_speed, diameter, viscosity, density, volume)
        )
    )
    assert result.successful
    assert result.derivative_valid
    np.testing.assert_allclose(
        result.particle_drag_coefficient * terminal_speed, buoyant_weight, rtol=2e-12
    )
    assert result.beta == 0.0


@pytest.mark.parametrize(
    ("index", "invalid"),
    (
        pytest.param(0, 0.0, id="zero-porosity"),
        pytest.param(0, -0.1, id="negative-porosity"),
        pytest.param(0, 1.1, id="excess-porosity"),
        pytest.param(0, np.nan, id="nan-porosity"),
        pytest.param(0, np.inf, id="infinite-porosity"),
        pytest.param(1, -0.01, id="negative-speed"),
        pytest.param(1, np.nan, id="nan-speed"),
        pytest.param(1, np.inf, id="infinite-speed"),
        pytest.param(2, 0.0, id="zero-diameter"),
        pytest.param(2, -0.001, id="negative-diameter"),
        pytest.param(2, np.nan, id="nan-diameter"),
        pytest.param(2, np.inf, id="infinite-diameter"),
        pytest.param(3, 0.0, id="zero-viscosity"),
        pytest.param(3, -0.001, id="negative-viscosity"),
        pytest.param(3, np.nan, id="nan-viscosity"),
        pytest.param(3, np.inf, id="infinite-viscosity"),
        pytest.param(4, 0.0, id="zero-density"),
        pytest.param(4, -1.0, id="negative-density"),
        pytest.param(4, np.nan, id="nan-density"),
        pytest.param(4, np.inf, id="infinite-density"),
        pytest.param(5, 0.0, id="zero-volume"),
        pytest.param(5, -1e-9, id="negative-volume"),
        pytest.param(5, np.nan, id="nan-volume"),
        pytest.param(5, np.inf, id="infinite-volume"),
    ),
)
def test_invalid_physical_inputs_report_failed_evidence(
    index: int, invalid: float
) -> None:
    inputs = np.asarray(
        (0.9, 0.01, _DIAMETER, _VISCOSITY, _DENSITY, _VOLUME), dtype=np.float64
    )
    inputs[index] = invalid
    result = InterphaseDragLawPlan("gidaspow").evaluate(
        *(jnp.asarray(value, dtype=jnp.float64) for value in inputs)
    )
    assert not result.admissible
    assert not result.successful
    assert not result.derivative_valid


def test_invalid_selector_is_refused() -> None:
    with pytest.raises(ValueError, match="correlation"):
        InterphaseDragLawPlan("invented")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "maximum", (0.0, -1.0, np.nan, np.inf), ids=("zero", "negative", "nan", "infinite")
)
def test_invalid_reynolds_limit_is_refused(maximum: float) -> None:
    with pytest.raises(ValueError, match="maximum_reynolds"):
        InterphaseDragLawPlan("stokes", maximum_reynolds=maximum)


def test_law_identity_owns_selected_correlation_and_admission_limit() -> None:
    first = InterphaseDragLawPlan("stokes", maximum_reynolds=0.1)
    equivalent = InterphaseDragLawPlan("stokes", maximum_reynolds=0.1)
    tightened = InterphaseDragLawPlan("stokes", maximum_reynolds=0.05)
    different = InterphaseDragLawPlan("schiller_naumann", maximum_reynolds=0.1)
    assert first.law_id == equivalent.law_id
    assert first.law_id != tightened.law_id
    assert first.law_id != different.law_id
    assert _evaluate(first).law_id == first.law_id


_SMOOTH_CASES = (
    pytest.param("stokes", 0.7, 0.5, id="stokes"),
    pytest.param("schiller_naumann", 0.9, 25.0, id="schiller-naumann"),
    pytest.param("di_felice", 0.65, 25.0, id="di-felice"),
    pytest.param("ergun", 0.4, 25.0, id="ergun"),
    pytest.param("wen_yu", 0.9, 25.0, id="wen-yu"),
    pytest.param("gidaspow", 0.6, 25.0, id="gidaspow-dense"),
    pytest.param("gidaspow", 0.9, 25.0, id="gidaspow-dilute"),
    pytest.param("huilin_gidaspow", 0.8, 25.0, id="huilin-gidaspow"),
    pytest.param("beetstra", 0.65, 25.0, id="beetstra"),
)


@pytest.mark.parametrize(("correlation", "porosity", "reynolds"), _SMOOTH_CASES)
def test_smooth_branch_jvp_matches_directional_finite_difference(
    correlation: InterphaseDragCorrelation, porosity: float, reynolds: float
) -> None:
    law = InterphaseDragLawPlan(correlation)
    speed = reynolds * _VISCOSITY / (_DENSITY * _DIAMETER)
    inputs = jnp.asarray(
        (porosity, speed, _DIAMETER, _VISCOSITY, _DENSITY, _VOLUME),
        dtype=jnp.float64,
    )
    direction = inputs * jnp.asarray(
        (0.03, 0.12, -0.04, 0.08, -0.02, 0.1), dtype=jnp.float64
    )

    def coefficients(values: Array) -> Array:
        result = law.evaluate(*values)
        return jnp.stack((result.beta, result.particle_drag_coefficient))

    result = law.evaluate(*inputs)
    assert result.successful
    assert result.derivative_valid
    _, tangent = jax.jvp(coefficients, (inputs,), (direction,))
    step = 1e-5
    finite_difference = (
        coefficients(inputs + step * direction) - coefficients(inputs - step * direction)
    ) / (2.0 * step)
    assert jnp.all(jnp.isfinite(tangent))
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-7, atol=1e-12)


@pytest.mark.parametrize(("correlation", "porosity", "reynolds"), _SMOOTH_CASES)
def test_smooth_branch_forward_and_reverse_jacobians_agree(
    correlation: InterphaseDragCorrelation, porosity: float, reynolds: float
) -> None:
    law = InterphaseDragLawPlan(correlation)
    speed = reynolds * _VISCOSITY / (_DENSITY * _DIAMETER)
    inputs = jnp.asarray(
        (porosity, speed, _DIAMETER, _VISCOSITY, _DENSITY, _VOLUME),
        dtype=jnp.float64,
    )

    def coefficients(values: Array) -> Array:
        result = law.evaluate(*values)
        return jnp.stack((result.beta, result.particle_drag_coefficient))

    forward = jax.jacfwd(coefficients)(inputs)
    reverse = jax.jacrev(coefficients)(inputs)
    assert jnp.all(jnp.isfinite(forward))
    assert jnp.all(jnp.isfinite(reverse))
    np.testing.assert_allclose(forward, reverse, rtol=3e-12, atol=1e-10)


@pytest.mark.parametrize(
    ("correlation", "porosity", "reynolds"),
    (
        pytest.param("stokes", 0.9, 1.0, id="stokes"),
        pytest.param("schiller_naumann", 0.9, 1000.0, id="schiller-naumann"),
        pytest.param("beetstra", 0.5, 2000.0, id="beetstra-superficial-reynolds"),
    ),
)
def test_published_reynolds_boundary_admission(
    correlation: InterphaseDragCorrelation, porosity: float, reynolds: float
) -> None:
    law = InterphaseDragLawPlan(correlation)
    boundary = _evaluate(law, porosity, reynolds)
    outside = _evaluate(law, porosity, reynolds * (1.0 + 1e-6))
    assert boundary.admissible
    assert boundary.successful
    assert not outside.admissible
    assert not outside.successful
    assert not outside.derivative_valid
    np.testing.assert_allclose(outside.reynolds_number, reynolds * (1.0 + 1e-6))


@pytest.mark.parametrize(
    ("correlation", "porosity", "scale"),
    (
        pytest.param("stokes", 0.9, 1.0, id="stokes"),
        pytest.param("schiller_naumann", 0.9, 1.0, id="schiller-naumann"),
        pytest.param("ergun", 0.5, 1.0, id="ergun"),
        pytest.param("beetstra", 0.5, 0.5, id="beetstra-superficial-reynolds"),
    ),
)
def test_explicit_reynolds_limit_tightens_admission(
    correlation: InterphaseDragCorrelation, porosity: float, scale: float
) -> None:
    law = InterphaseDragLawPlan(correlation, maximum_reynolds=0.1)
    inside = _evaluate(law, porosity, 0.09 / scale)
    outside = _evaluate(law, porosity, 0.11 / scale)
    assert inside.successful
    assert not outside.admissible
    assert not outside.successful
    assert not outside.derivative_valid


@pytest.mark.parametrize(
    ("correlation", "maximum"),
    (
        pytest.param("stokes", 1.01, id="stokes"),
        pytest.param("schiller_naumann", 1000.01, id="schiller-naumann"),
        pytest.param("beetstra", 1000.01, id="beetstra"),
    ),
)
def test_user_limit_cannot_widen_published_domain(
    correlation: InterphaseDragCorrelation, maximum: float
) -> None:
    with pytest.raises(ValueError, match="maximum_reynolds"):
        InterphaseDragLawPlan(correlation, maximum_reynolds=maximum)


@pytest.mark.parametrize(
    "porosity", (0.349, 0.901, 1.0), ids=("too-dense", "too-dilute", "absent-solid")
)
def test_beetstra_outside_published_voidage_range_is_inadmissible(
    porosity: float,
) -> None:
    result = _evaluate(InterphaseDragLawPlan("beetstra"), porosity, 20.0)
    assert not result.admissible
    assert not result.successful
    assert not result.derivative_valid


@pytest.mark.parametrize("porosity", (0.7, 0.8), ids=("dense", "threshold"))
def test_wen_yu_dense_regime_is_inadmissible(porosity: float) -> None:
    result = _evaluate(InterphaseDragLawPlan("wen_yu"), porosity, 20.0)
    assert not result.admissible
    assert not result.successful
    assert not result.derivative_valid


def _beetstra_reference(porosity: float, reynolds: float) -> tuple[float, float]:
    """Beetstra et al. (2007), DOI 10.1002/aic.11065, Eqs. (1), (2), (17).

    The paper's Reynolds number and Stokes normalization use superficial speed,
    unlike the API's reported intrinsic-slip Reynolds number.
    """
    solid = 1.0 - porosity
    superficial_reynolds = porosity * reynolds
    creeping = 10.0 * solid / porosity**2 + porosity**2 * (1.0 + 1.5 * np.sqrt(solid))
    inertial = 0.0
    if superficial_reynolds > 0.0:
        inertial = (
            0.413
            * superficial_reynolds
            / (24.0 * porosity**2)
            * (
                1.0 / porosity
                + 3.0 * solid * porosity
                + 8.4 * superficial_reynolds**-0.343
            )
            / (
                1.0
                + 10.0 ** (3.0 * solid)
                * superficial_reynolds ** (-(1.0 + 4.0 * solid) / 2.0)
            )
        )
    particle = 3.0 * np.pi * _VISCOSITY * _DIAMETER * porosity * (creeping + inertial)
    beta = solid * particle / _VOLUME
    return float(beta), float(particle)


@pytest.mark.parametrize(
    ("porosity", "reynolds"),
    (
        pytest.param(0.35, 0.0, id="dense-creeping"),
        pytest.param(0.5, 20.0, id="packed-superficial-re10"),
        pytest.param(0.9, 100.0 / 0.9, id="dilute-superficial-re100"),
        pytest.param(0.65, 1000.0 / 0.65, id="maximum-superficial-re1000"),
    ),
)
def test_beetstra_matches_superficial_reynolds_scalar_reference(
    porosity: float, reynolds: float
) -> None:
    result = _evaluate(InterphaseDragLawPlan("beetstra"), porosity, reynolds)
    reference = _beetstra_reference(porosity, reynolds)
    assert result.successful
    assert result.admissible
    np.testing.assert_allclose(
        (result.beta, result.particle_drag_coefficient), reference, rtol=3e-13
    )
    np.testing.assert_allclose(result.reynolds_number, reynolds, rtol=2e-13)


def test_out_of_domain_reynolds_is_evidence_not_clipped_drag() -> None:
    reynolds = 1001.0
    result = _evaluate(InterphaseDragLawPlan("schiller_naumann"), 0.9, reynolds)
    reference = 18.0 * _VISCOSITY * 0.1 / _DIAMETER**2 * (1.0 + 0.15 * reynolds**0.687)
    boundary = _evaluate(InterphaseDragLawPlan("schiller_naumann"), 0.9, 1000.0)
    assert not result.admissible
    assert not result.successful
    np.testing.assert_allclose(result.beta, reference, rtol=3e-13)
    assert result.beta > boundary.beta


def test_gidaspow_hard_switch_exposes_jump_and_signed_branch_margin() -> None:
    law = InterphaseDragLawPlan("gidaspow")
    step = 1e-8
    left = _evaluate(law, 0.8 - step, 25.0)
    switch = _evaluate(law, 0.8, 25.0)
    right = _evaluate(law, 0.8 + step, 25.0)
    assert left.successful
    assert switch.successful
    assert right.successful
    assert left.branch_code == 0
    assert switch.branch_code == 0
    assert right.branch_code == 1
    np.testing.assert_allclose(left.branch_margin, -step, rtol=1e-8)
    assert switch.branch_margin == 0.0
    np.testing.assert_allclose(right.branch_margin, step, rtol=1e-8)
    assert left.derivative_valid
    assert not switch.derivative_valid
    assert right.derivative_valid
    assert not np.isclose(left.beta, right.beta, rtol=1e-3, atol=0.0)
    solid = 0.2
    speed = 25.0 * _VISCOSITY / (_DENSITY * _DIAMETER)
    dense_reference = (
        150.0 * _VISCOSITY * solid**2 / (0.8 * _DIAMETER**2)
        + 1.75 * _DENSITY * solid * speed / _DIAMETER
    )
    np.testing.assert_allclose(switch.beta, dense_reference, rtol=3e-13)


@pytest.mark.parametrize(
    "porosity", (0.75, 0.8, 0.85), ids=("dense-side", "center", "dilute-side")
)
def test_huilin_gidaspow_value_and_first_derivative_are_continuous(
    porosity: float,
) -> None:
    """The published arctangent stitch has no finite-width endpoints."""
    law = InterphaseDragLawPlan("huilin_gidaspow")
    reynolds = 25.0
    speed = reynolds * _VISCOSITY / (_DENSITY * _DIAMETER)

    def beta(epsilon: Array) -> Array:
        result = law.evaluate(
            epsilon,
            *(
                jnp.asarray(value, dtype=jnp.float64)
                for value in (speed, _DIAMETER, _VISCOSITY, _DENSITY, _VOLUME)
            ),
        )
        return result.beta

    epsilon = jnp.asarray(porosity, dtype=jnp.float64)
    step = 1e-7
    middle = _evaluate(law, porosity, reynolds)
    left = _evaluate(law, porosity - step, reynolds)
    right = _evaluate(law, porosity + step, reynolds)
    assert middle.successful
    assert middle.derivative_valid
    assert left.derivative_valid
    assert right.derivative_valid
    assert middle.branch_code == 2
    derivative = jax.grad(beta)(epsilon)
    left_derivative = jax.grad(beta)(epsilon - step)
    right_derivative = jax.grad(beta)(epsilon + step)
    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(left.beta, right.beta, rtol=2e-5)
    np.testing.assert_allclose(left_derivative, right_derivative, rtol=2e-4)
    np.testing.assert_allclose(
        (right.beta - left.beta) / (2.0 * step), derivative, rtol=3e-7
    )


@pytest.mark.parametrize(("correlation", "porosity", "reynolds"), _SMOOTH_CASES)
@pytest.mark.parametrize("precision", ("float32", "float64"), ids=("float32", "float64"))
def test_batched_dtype_contract_preserves_all_evidence_shapes(
    correlation: InterphaseDragCorrelation,
    porosity: float,
    reynolds: float,
    precision: str,
) -> None:
    dtype = jnp.float32 if precision == "float32" else jnp.float64
    speed = reynolds * _VISCOSITY / (_DENSITY * _DIAMETER)
    inputs = tuple(
        jnp.asarray((value, value), dtype=dtype)
        for value in (porosity, speed, _DIAMETER, _VISCOSITY, _DENSITY, _VOLUME)
    )
    law = InterphaseDragLawPlan(correlation)
    result = jax.jit(law.evaluate)(*inputs)
    assert result.successful
    assert result.beta.shape == (2,)
    assert result.beta.dtype == dtype
    assert result.particle_drag_coefficient.shape == (2,)
    assert result.particle_drag_coefficient.dtype == dtype
    assert result.reynolds_number.shape == (2,)
    assert result.reynolds_number.dtype == dtype
    assert result.branch_margin.shape == (2,)
    assert result.branch_margin.dtype == dtype
    assert result.branch_code.shape == (2,)
    assert jnp.issubdtype(result.branch_code.dtype, jnp.integer)
    assert result.derivative_valid.shape == (2,)
    assert result.derivative_valid.dtype == jnp.bool_
    assert result.admissible.shape == (2,)
    assert result.admissible.dtype == jnp.bool_
    assert jnp.all(result.admissible)
    assert jnp.all(result.derivative_valid)
    assert result.law_id == law.law_id
    np.testing.assert_allclose(result.reynolds_number, (reynolds, reynolds), rtol=3e-7)


@pytest.mark.parametrize(
    ("correlation", "porosity"),
    (
        pytest.param("stokes", 0.7, id="stokes"),
        pytest.param("schiller_naumann", 0.9, id="schiller-naumann"),
        pytest.param("di_felice", 0.65, id="di-felice"),
        pytest.param("ergun", 0.4, id="ergun"),
        pytest.param("wen_yu", 0.9, id="wen-yu"),
        pytest.param("gidaspow", 0.6, id="gidaspow-dense"),
        pytest.param("gidaspow", 0.9, id="gidaspow-dilute"),
        pytest.param("huilin_gidaspow", 0.8, id="huilin-gidaspow"),
        pytest.param("beetstra", 0.65, id="beetstra"),
    ),
)
def test_exact_zero_reynolds_has_finite_nonzero_coefficients(
    correlation: InterphaseDragCorrelation, porosity: float
) -> None:
    result = _evaluate(InterphaseDragLawPlan(correlation), porosity, 0.0)
    assert result.successful
    assert result.admissible
    assert result.reynolds_number == 0.0
    assert jnp.isfinite(result.beta)
    assert jnp.isfinite(result.particle_drag_coefficient)
    assert result.beta > 0.0
    assert result.particle_drag_coefficient > 0.0


@pytest.mark.parametrize(
    ("correlation", "reynolds"),
    (
        pytest.param("stokes", 0.1, id="stokes"),
        pytest.param("schiller_naumann", 25.0, id="schiller-naumann"),
    ),
)
def test_dilute_limit_keeps_isolated_particle_drag_when_bulk_drag_vanishes(
    correlation: InterphaseDragCorrelation, reynolds: float
) -> None:
    law = InterphaseDragLawPlan(correlation)
    near = _evaluate(law, 1.0 - 1e-6, reynolds)
    absent = _evaluate(law, 1.0, reynolds)
    assert near.successful
    assert absent.successful
    assert absent.beta == 0.0
    assert absent.particle_drag_coefficient > 0.0
    np.testing.assert_allclose(
        near.particle_drag_coefficient, absent.particle_drag_coefficient, rtol=1e-13
    )


@pytest.mark.parametrize(
    "correlation",
    ("stokes", "schiller_naumann", "di_felice"),
    ids=("stokes", "schiller-naumann", "di-felice"),
)
def test_isolated_particle_coefficient_does_not_depend_on_supplied_volume(
    correlation: InterphaseDragCorrelation,
) -> None:
    law = InterphaseDragLawPlan(correlation)
    first = _evaluate(law, 0.9, 0.1)
    second = _evaluate(law, 0.9, 0.1, volume=3.0 * _VOLUME)
    assert first.successful
    assert second.successful
    np.testing.assert_allclose(first.beta, second.beta, rtol=1e-13)
    np.testing.assert_allclose(
        first.particle_drag_coefficient, second.particle_drag_coefficient, rtol=1e-13
    )


@pytest.mark.parametrize(
    ("correlation", "porosity"),
    (
        pytest.param("ergun", 0.4, id="ergun"),
        pytest.param("wen_yu", 0.9, id="wen-yu"),
        pytest.param("gidaspow", 0.6, id="gidaspow"),
        pytest.param("huilin_gidaspow", 0.8, id="huilin-gidaspow"),
        pytest.param("beetstra", 0.65, id="beetstra"),
    ),
)
def test_bulk_law_particle_coefficient_scales_with_supplied_volume(
    correlation: InterphaseDragCorrelation, porosity: float
) -> None:
    law = InterphaseDragLawPlan(correlation)
    first = _evaluate(law, porosity, 0.1)
    second = _evaluate(law, porosity, 0.1, volume=3.0 * _VOLUME)
    assert first.successful
    assert second.successful
    np.testing.assert_allclose(first.beta, second.beta, rtol=1e-13)
    np.testing.assert_allclose(
        second.particle_drag_coefficient,
        3.0 * first.particle_drag_coefficient,
        rtol=1e-13,
    )


def _scalar_reference(
    correlation: InterphaseDragCorrelation,
    porosity: float,
    reynolds: float,
    volume: float,
) -> tuple[float, float]:
    """Host-only scalar oracle, independent of every production JAX helper.

    Ergun (1952), CEP 48:89–94; Schiller–Naumann (1933), VDI 77;
    Di Felice (1994), DOI 10.1016/0301-9322(94)90011-6, same-volumetric-flux
    convention in the original abstract; Beetstra (2007), Eqs. (1), (2), (17),
    DOI 10.1002/aic.11065. Wen–Yu superficial Reynolds convention is described
    by Beetstra's Eq. (4), with the canonical Gidaspow 2.65 rounding.

    Original full-text verification of the Di Felice Dallavalle constants and
    Huilin–Gidaspow stitching constants remains unavailable. These fixtures pin
    the accepted published-model convention, not a claim of primary-source
    verification. Secondary provenance for Di Felice is NETL's discussion
    https://mfix.netl.doe.gov/forum/t/di-felice-drag-correlation/4585;
    and for H-G is the Fluent Theory Guide:
    https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_th/flu_th_sec_eulermp_theory_exchange.html.
    """
    solid = 1.0 - porosity
    speed = reynolds * _VISCOSITY / (_DENSITY * _DIAMETER)
    creeping_beta = 18.0 * _VISCOSITY * solid / _DIAMETER**2
    creeping_particle = 3.0 * np.pi * _VISCOSITY * _DIAMETER
    superficial_reynolds = porosity * reynolds
    ergun = (
        150.0 * _VISCOSITY * solid**2 / (porosity * _DIAMETER**2)
        + 1.75 * _DENSITY * solid * speed / _DIAMETER
    )
    wen_yu_resistance = (
        1.0 + 0.15 * superficial_reynolds**0.687
        if superficial_reynolds < 1000.0
        else 0.44 * superficial_reynolds / 24.0
    )
    wen_yu = creeping_beta * wen_yu_resistance * porosity**-2.65
    match correlation:
        case "stokes":
            return float(creeping_beta), float(creeping_particle)
        case "schiller_naumann":
            correction = 1.0 + 0.15 * reynolds**0.687
            return float(creeping_beta * correction), float(
                creeping_particle * correction
            )
        case "di_felice":
            chi = 3.7
            if superficial_reynolds > 0.0:
                chi -= 0.65 * np.exp(-0.5 * (1.5 - np.log10(superficial_reynolds)) ** 2)
            resistance = (0.63 * np.sqrt(superficial_reynolds) + 4.8) ** 2 / 24.0
            correction = resistance * porosity ** (1.0 - chi)
            return float(creeping_beta * correction), float(
                creeping_particle * correction
            )
        case "ergun":
            beta = ergun
        case "wen_yu":
            beta = wen_yu
        case "gidaspow":
            beta = ergun if porosity <= 0.8 else wen_yu
        case "huilin_gidaspow":
            dense_weight = 0.5 + np.arctan(150.0 * 1.75 * (solid - 0.2)) / np.pi
            beta = dense_weight * ergun + (1.0 - dense_weight) * wen_yu
        case "beetstra":
            beta, _ = _beetstra_reference(porosity, reynolds)
        case _:
            assert_never(correlation)
    return float(beta), float(volume * beta / solid)


_REFERENCE_CASES = (
    pytest.param("stokes", 0.65, 0.0, id="stokes-zero"),
    pytest.param("stokes", 0.9, 0.5, id="stokes-finite"),
    pytest.param("schiller_naumann", 0.9, 0.0, id="schiller-naumann-zero"),
    pytest.param("schiller_naumann", 0.9, 10.0, id="schiller-naumann-low"),
    pytest.param("schiller_naumann", 0.9, 750.0, id="schiller-naumann-high"),
    pytest.param("di_felice", 0.45, 0.0, id="di-felice-zero"),
    pytest.param("di_felice", 0.45, 0.1, id="di-felice-dense"),
    pytest.param("di_felice", 0.85, 100.0, id="di-felice-dilute"),
    pytest.param("ergun", 0.35, 0.0, id="ergun-zero"),
    pytest.param("ergun", 0.35, 10.0, id="ergun-dense"),
    pytest.param("ergun", 0.65, 300.0, id="ergun-inertial"),
    pytest.param("wen_yu", 0.9, 0.0, id="wen-yu-zero"),
    pytest.param("wen_yu", 0.85, 20.0, id="wen-yu-low"),
    pytest.param("wen_yu", 0.95, 1500.0, id="wen-yu-high"),
    pytest.param("gidaspow", 0.6, 20.0, id="gidaspow-dense"),
    pytest.param("gidaspow", 0.8, 20.0, id="gidaspow-threshold"),
    pytest.param("gidaspow", 0.9, 20.0, id="gidaspow-dilute"),
    pytest.param("huilin_gidaspow", 0.6, 20.0, id="huilin-gidaspow-dense"),
    pytest.param("huilin_gidaspow", 0.8, 0.0, id="huilin-gidaspow-zero-stitch"),
    pytest.param("huilin_gidaspow", 0.8, 20.0, id="huilin-gidaspow-center"),
    pytest.param("huilin_gidaspow", 0.9, 20.0, id="huilin-gidaspow-dilute"),
    pytest.param("beetstra", 0.65, 0.0, id="beetstra-zero"),
    pytest.param("beetstra", 0.5, 20.0, id="beetstra-dense"),
    pytest.param("beetstra", 0.85, 750.0, id="beetstra-dilute"),
)


@pytest.mark.parametrize(("correlation", "porosity", "reynolds"), _REFERENCE_CASES)
def test_all_correlations_match_independent_scalar_numpy_equations(
    correlation: InterphaseDragCorrelation, porosity: float, reynolds: float
) -> None:
    volume = 1.7 * _VOLUME
    result = _evaluate(
        InterphaseDragLawPlan(correlation), porosity, reynolds, volume=volume
    )
    reference = _scalar_reference(correlation, porosity, reynolds, volume)
    assert result.successful
    assert result.admissible
    np.testing.assert_allclose(
        (result.beta, result.particle_drag_coefficient), reference, rtol=3e-13
    )
    np.testing.assert_allclose(result.reynolds_number, reynolds, rtol=2e-13, atol=0.0)


@pytest.mark.parametrize(
    ("correlation", "porosity"),
    (
        pytest.param("schiller_naumann", 0.9, id="schiller-naumann"),
        pytest.param("di_felice", 0.65, id="di-felice"),
        pytest.param("wen_yu", 0.9, id="wen-yu"),
        pytest.param("gidaspow", 0.9, id="gidaspow-dilute"),
        pytest.param("huilin_gidaspow", 0.8, id="huilin-gidaspow"),
        pytest.param("beetstra", 0.65, id="beetstra"),
    ),
)
def test_fractional_power_zero_reynolds_does_not_claim_regular_derivative(
    correlation: InterphaseDragCorrelation, porosity: float
) -> None:
    result = _evaluate(InterphaseDragLawPlan(correlation), porosity, 0.0)
    assert result.successful
    assert result.admissible
    assert not result.derivative_valid


@pytest.mark.parametrize(
    "reynolds", (900.0, 1200.0), ids=("below-cd-switch", "above-cd-switch")
)
def test_wen_yu_cd_switch_uses_superficial_reynolds(reynolds: float) -> None:
    porosity = 0.9
    result = _evaluate(InterphaseDragLawPlan("wen_yu"), porosity, reynolds)
    reference = _scalar_reference("wen_yu", porosity, reynolds, _VOLUME)
    assert result.successful
    assert result.derivative_valid
    assert result.branch_code == (0 if porosity * reynolds < 1000.0 else 1)
    np.testing.assert_allclose(
        (result.beta, result.particle_drag_coefficient), reference, rtol=3e-13
    )


@pytest.mark.parametrize("porosity", (-0.1, 1.1), ids=("negative", "excess"))
def test_ergun_invalid_column_voidage_is_not_repaired(porosity: float) -> None:
    reynolds = 25.0
    intrinsic_speed = reynolds * _VISCOSITY / (_DENSITY * _DIAMETER)
    solid = 1.0 - porosity
    raw_beta = (
        150.0 * _VISCOSITY * solid**2 / (porosity * _DIAMETER**2)
        + 1.75 * _DENSITY * solid * intrinsic_speed / _DIAMETER
    )
    result = _evaluate(InterphaseDragLawPlan("ergun"), porosity, reynolds)
    assert not result.admissible
    assert not result.successful
    assert not result.derivative_valid
    np.testing.assert_allclose(result.beta, raw_beta, rtol=3e-13)
    np.testing.assert_allclose(
        result.beta * intrinsic_speed / porosity,
        raw_beta * intrinsic_speed / porosity,
        rtol=3e-13,
    )


@pytest.mark.parametrize(
    ("correlation", "porosity", "admitted"),
    (
        pytest.param("ergun", 0.5, False, id="ergun"),
        pytest.param("gidaspow", 0.5, False, id="gidaspow-dense"),
        pytest.param("gidaspow", 0.9, True, id="gidaspow-dilute"),
        pytest.param("huilin_gidaspow", 0.5, False, id="huilin-dense"),
        pytest.param("huilin_gidaspow", 0.9, False, id="huilin-dilute"),
    ),
)
def test_reynolds_cap_covers_each_evaluated_branch(
    correlation: InterphaseDragCorrelation, porosity: float, admitted: bool
) -> None:
    # Re=200, epsilon Re=100 in the reported dense counterexample.
    cap = 150.0 if porosity == 0.5 else 190.0
    result = _evaluate(
        InterphaseDragLawPlan(correlation, maximum_reynolds=cap), porosity, 200.0
    )
    assert bool(result.successful) is admitted
    assert bool(result.admissible) is admitted
    assert bool(result.derivative_valid) is admitted
