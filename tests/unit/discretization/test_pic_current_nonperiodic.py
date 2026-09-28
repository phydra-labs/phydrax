#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Charge-conserving current on nonperiodic axes with boundary exits.

References are independent of the deposition: exit points are the analytic
first crossing of the straight path with the box faces, and the domain integral
of the physical current is ``Σ q Δx / Δt`` over the deposited paths.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic


def _bridge(count: int, periodic: tuple[bool, bool, bool]) -> Any:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count, periodic=value) for value in periodic),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
    return D.StructuredCochainBridge(grid)


def _current(bridge: Any, order: PIC.PICShapeOrder, capacity: int) -> Any:
    particles = D.ParticleSetPlan(
        jnp.arange(capacity), jnp.ones((capacity,)), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(jnp.ones((capacity,)), "open").prepare(particles)
    transfer = PIC.PICParticleCochainTransferPlan(bridge, shape_order=order).prepare(
        charged
    )
    return PIC.ChargeConservingCurrentPlan(transfer)


_START = np.asarray(
    [
        [0.50, 0.50, 0.50],
        [0.97, 0.20, 0.30],
        [0.10, 0.02, 0.90],
        [0.30, 1.00, 0.40],
        [0.60, 0.60, 0.99],
        [0.00, 0.40, 0.70],
    ]
)
_STEP = np.asarray(
    [
        [0.10, -0.05, 0.03],
        [0.06, 0.01, 0.00],
        [0.00, -0.05, 0.02],
        [0.05, 0.00, 0.00],
        [0.02, 0.00, 0.05],
        [0.08, -0.02, 0.01],
    ]
)
_CHARGE = np.asarray([1.0, -2.0, 1.5, 0.7, 1.1, -0.4])


def _exit_points(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = end - start
    fraction = np.ones_like(start)
    above, below = end > 1.0, end < 0.0
    fraction[above] = (1.0 - start[above]) / delta[above]
    fraction[below] = -start[below] / delta[below]
    parameter = np.min(fraction, axis=1)
    return start + parameter[:, None] * delta, np.any(above | below, axis=1)


def test_open_box_current_clips_exits_and_conserves_charge() -> None:
    bridge = _bridge(6, (False, False, False))
    current = _current(bridge, 1, _START.shape[0])
    step = 0.01
    end = _START + _STEP
    result = jax.jit(
        lambda left, right: current.deposit(left, right, step, macrocharge=_CHARGE)
    )(_START, end)
    head, leaves = _exit_points(_START, end)

    assert bool(result.successful)
    np.testing.assert_array_equal(np.asarray(result.boundary_exit), leaves)
    np.testing.assert_allclose(np.asarray(result.deposited_end), head, atol=1e-15)
    # Exits leave their charge on the crossed face: the box keeps every charge.
    np.testing.assert_allclose(
        float(jnp.sum(result.end_charge.content)), np.sum(_CHARGE), rtol=1e-14
    )
    rate = float(
        jnp.max(jnp.abs(result.end_charge.cochain - result.start_charge.cochain))
    )
    assert float(result.maximum_continuity_defect) <= 1e-13 * rate / step
    # Face-parallel edge fields of a nonperiodic axis carry half-weight dual faces.
    weights = [np.ones(shape) for shape in bridge.orientation_shapes[1]]
    for axis, weight in enumerate(weights):
        for other in range(3):
            if other == axis:
                continue
            index: list[slice | int] = [slice(None)] * 3
            index[other] = 0
            weight[tuple(index)] *= 0.5
            index[other] = -1
            weight[tuple(index)] *= 0.5
    integrals = np.asarray(
        [
            np.sum(weight * np.asarray(value)) / 6.0**2
            for weight, value in zip(
                weights, bridge.unpack(1, result.current), strict=True
            )
        ]
    )
    np.testing.assert_allclose(
        integrals, np.sum(_CHARGE[:, None] * (head - _START), axis=0) / step, atol=1e-10
    )


def test_periodic_axes_keep_unclipped_paths() -> None:
    periodic = _current(_bridge(6, (True, True, True)), 1, _START.shape[0])
    end = _START + _STEP
    result = periodic.deposit(_START, end, 0.01, macrocharge=_CHARGE)
    assert bool(result.successful)
    assert not bool(jnp.any(result.boundary_exit))
    np.testing.assert_array_equal(np.asarray(result.deposited_end), end)


def test_mixed_axes_clip_only_nonperiodic_faces() -> None:
    current = _current(_bridge(6, (True, False, True)), 1, 2)
    start = np.asarray([[0.98, 0.5, 0.5], [0.5, 0.97, 0.5]])
    end = start + np.asarray([[0.05, 0.0, 0.0], [0.0, 0.06, 0.0]])
    result = current.deposit(start, end, 0.01, macrocharge=np.ones(2))
    np.testing.assert_array_equal(np.asarray(result.boundary_exit), [False, True])
    np.testing.assert_allclose(
        np.asarray(result.deposited_end), [end[0], [0.5, 1.0, 0.5]], atol=1e-15
    )
    assert float(result.maximum_continuity_defect) <= 1e-9


@pytest.mark.parametrize("order", [2, 3], ids=["quadratic", "cubic"])
def test_spline_current_conserves_charge_away_from_open_faces(
    order: PIC.PICShapeOrder,
) -> None:
    current = _current(_bridge(8, (False, False, False)), order, 3)
    start = np.asarray([[0.40, 0.45, 0.50], [0.55, 0.52, 0.47], [0.48, 0.60, 0.41]])
    end = start + np.asarray([[0.1, -0.04, 0.02], [-0.03, 0.08, 0.05], [0.0, 0.0, 0.1]])
    result = current.deposit(start, end, 0.01, macrocharge=np.asarray([1.0, -1.0, 2.0]))
    assert bool(result.successful)
    rate = float(
        jnp.max(jnp.abs(result.end_charge.cochain - result.start_charge.cochain))
    )
    assert float(result.maximum_continuity_defect) <= 1e-13 * rate / 0.01


def test_spline_stencil_beyond_an_open_face_is_refused() -> None:
    current = _current(_bridge(8, (False, False, False)), 2, 1)
    start = np.asarray([[0.02, 0.5, 0.5]])
    with pytest.raises(Exception, match="support, or boundary"):
        jax.block_until_ready(
            current.deposit(start, start + 0.01, 0.01, macrocharge=np.ones(1)).current
        )
