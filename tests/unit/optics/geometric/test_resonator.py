#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp

from phydrax.optics.geometric._paraxial import DifferentialRayMap
from phydrax.optics.geometric._resonator import (
    analyze_paraxial_resonator,
    ParaxialResonatorPlan,
    ParaxialResonatorStatus,
    prepare_paraxial_resonator,
)


_COORDINATES = "(u,v,nθu,nθv)"


def _map(
    jacobian: Any,
    *,
    input_reference: Any = None,
    output_reference: Any = None,
    input_frame: Any = "loop",
    output_frame: Any = "loop",
    source: Any = "resonator",
    coordinate_convention: Any = _COORDINATES,
    valid: Any = True,
) -> Any:
    zero = jnp.zeros((4,))
    return DifferentialRayMap(
        zero if input_reference is None else jnp.asarray(input_reference),
        zero if output_reference is None else jnp.asarray(output_reference),
        jnp.asarray(jacobian),
        jnp.asarray(1.0),
        jnp.asarray(True),
        jnp.asarray(valid),
        jnp.asarray(0, dtype=jnp.int32),
        input_frame_id=input_frame,
        output_frame_id=output_frame,
        source_prepared_id=source,
        coordinate_convention=coordinate_convention,
    )


def _stable_map(phases: Any = (0.37, 0.71), curvatures: Any = (1.4, 0.65)) -> Any:
    phase = jnp.asarray(phases)
    curvature = jnp.asarray(curvatures)
    cosine = jnp.diag(jnp.cos(phase))
    sine = jnp.diag(jnp.sin(phase))
    inverse_curvature = jnp.diag(1.0 / curvature)
    curvature_matrix = jnp.diag(curvature)
    return jnp.block(
        [
            [cosine, sine @ inverse_curvature],
            [-curvature_matrix @ sine, cosine],
        ]
    )


def _canonical_rotation(angle: Any) -> Any:
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    transverse = jnp.asarray(((cosine, -sine), (sine, cosine)))
    zero = jnp.zeros((2, 2))
    return jnp.block([[transverse, zero], [zero, transverse]])


def test_resonator_scenario_1() -> None:
    first_jacobian = _stable_map(phases=(0.19, 0.31), curvatures=(1.2, 0.8))
    second_jacobian = _stable_map(phases=(0.23, 0.17), curvatures=(1.2, 0.8))
    round_trip = second_jacobian @ first_jacobian
    fixed_point = jnp.asarray((0.12, -0.08, 0.025, -0.04))
    first_offset = jnp.asarray((0.01, -0.02, 0.005, 0.003))
    second_offset = (jnp.eye(4) - round_trip) @ fixed_point - (
        second_jacobian @ first_offset
    )
    maps = (
        _map(
            first_jacobian,
            output_reference=first_offset,
            input_frame="entrance",
            output_frame="middle",
        ),
        _map(
            second_jacobian,
            output_reference=second_offset,
            input_frame="middle",
            output_frame="entrance",
        ),
    )

    result = prepare_paraxial_resonator(ParaxialResonatorPlan(maps)).execute()

    assert int(result.status) == int(ParaxialResonatorStatus.SUCCESS_STABLE)
    assert bool(result.successful)
    assert bool(result.stable)
    assert jnp.allclose(result.round_trip_jacobian, round_trip, atol=2e-12)
    assert jnp.allclose(
        result.round_trip_offset,
        second_jacobian @ first_offset + second_offset,
        atol=2e-12,
    )
    assert jnp.allclose(result.closed_orbit, fixed_point, atol=2e-11)
    assert float(result.evidence.closed_orbit_residual) < 1e-11
    uncoupled = _stable_map()
    rotation = _canonical_rotation(0.43)
    coupled = rotation @ uncoupled @ rotation.T
    fixed_point = jnp.asarray((0.04, -0.07, 0.015, 0.025))
    ray_map = _map(
        coupled,
        input_reference=fixed_point,
        output_reference=fixed_point,
    )
    prepared = prepare_paraxial_resonator(ParaxialResonatorPlan((ray_map,)))

    result = jax.jit(analyze_paraxial_resonator)(prepared)
    mode = result.mode
    symplectic_form = jnp.block(
        [
            [jnp.zeros((2, 2)), jnp.eye(2)],
            [-jnp.eye(2), jnp.zeros((2, 2))],
        ]
    )
    transported = coupled @ mode.lagrangian_state
    invariant = -1j * (
        jnp.conj(mode.lagrangian_state.T) @ symplectic_form @ mode.lagrangian_state
    )

    assert int(result.status) == int(ParaxialResonatorStatus.SUCCESS_STABLE)
    assert bool(mode.valid)
    assert jnp.allclose(
        transported,
        mode.lagrangian_state @ mode.round_trip_action,
        atol=3e-9,
    )
    assert jnp.allclose(
        mode.lagrangian_state.T @ symplectic_form @ mode.lagrangian_state,
        jnp.zeros((2, 2)),
        atol=3e-9,
    )
    assert jnp.allclose(invariant, jnp.eye(2), atol=3e-9)
    assert jnp.all(jnp.linalg.eigvalsh(jnp.imag(mode.curvature)) > 0.0)
    assert float(mode.invariance_residual) < 3e-9
    unstable_jacobian = jnp.diag(jnp.asarray((1.4, 1.2, 1.0 / 1.4, 1.0 / 1.2)))
    unstable = prepare_paraxial_resonator(
        ParaxialResonatorPlan((_map(unstable_jacobian),))
    ).execute()
    marginal = prepare_paraxial_resonator(
        ParaxialResonatorPlan((_map(-jnp.eye(4)),))
    ).execute()

    assert int(unstable.status) == int(ParaxialResonatorStatus.SUCCESS_UNSTABLE)
    assert bool(unstable.successful)
    assert not bool(unstable.stable)
    assert not bool(unstable.mode.valid)
    assert jnp.allclose(unstable.mode.lagrangian_state, 0.0)
    assert int(marginal.status) == int(ParaxialResonatorStatus.MARGINAL)
    assert not bool(marginal.successful)
    assert not bool(marginal.mode.valid)


def test_resonator_scenario_2() -> None:
    singular = prepare_paraxial_resonator(
        ParaxialResonatorPlan((_map(jnp.eye(4)),))
    ).execute()
    nonsymplectic = prepare_paraxial_resonator(
        ParaxialResonatorPlan((_map(2.0 * jnp.eye(4)),))
    ).execute()

    assert int(singular.status) == int(ParaxialResonatorStatus.SINGULAR_CLOSED_ORBIT)
    assert int(nonsymplectic.status) == int(ParaxialResonatorStatus.NONSYMPLECTIC)
    assert not bool(singular.successful)
    assert not bool(nonsymplectic.successful)
    jacobian = _stable_map().at[0, 0].set(jnp.nan)

    result = prepare_paraxial_resonator(
        ParaxialResonatorPlan((_map(jacobian),))
    ).execute()

    assert int(result.status) == int(ParaxialResonatorStatus.NONFINITE)
    assert not bool(result.successful)
    assert not bool(result.mode.valid)
    for maps in (
        (
            _map(jnp.eye(4), input_frame="a", output_frame="b"),
            _map(jnp.eye(4), input_frame="c", output_frame="a"),
        ),
        (_map(jnp.eye(4), coordinate_convention="(u,v,θu,θv)"),),
        (
            _map(jnp.eye(4), input_frame="a", output_frame="b", source="first"),
            _map(jnp.eye(4), input_frame="b", output_frame="a", source="second"),
        ),
        (_map(jnp.eye(4), valid=False),),
    ):
        result = prepare_paraxial_resonator(ParaxialResonatorPlan(maps)).execute()

        assert int(result.status) == int(ParaxialResonatorStatus.INCOMPATIBLE_MAPS)
        assert not bool(result.successful)
        assert not bool(result.mode.valid)
