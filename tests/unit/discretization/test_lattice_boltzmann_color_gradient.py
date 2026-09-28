#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.lattice_boltzmann import (
    BGKCollisionPlan,
    ColorGradientLBMMethod,
    ColorGradientLBMRuntimeParameters,
    D2Q9,
    GuoForcingPlan,
    LatticeBoltzmannBoundaryPlan,
    LatticeBoltzmannMethodPlan,
    LatticeBoltzmannPlan,
    NearContactRepulsionPlan,
)
from phydrax.equations import (
    ColorGradientLatticeBoltzmannProblem,
    compile_color_gradient_lattice_boltzmann_problem,
)
from phydrax.interfacial_transport import InterfaceTensionMatrix


_BINARY_TENSION = np.asarray([[0.0, 0.01], [0.01, 0.0]])


def _compiled(
    components: Sequence[str],
    shape: tuple[int, int],
    near_contact: NearContactRepulsionPlan | None,
) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(n, periodic=True) for n in shape),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (float(shape[0]), float(shape[1])))))
    discretization = LatticeBoltzmannPlan(grid, D2Q9()).prepare()
    method = ColorGradientLBMMethod(
        LatticeBoltzmannMethodPlan(BGKCollisionPlan(), forcing=GuoForcingPlan()),
        components,
        near_contact=near_contact,
        maximum_capillary_number=10.0,
    )
    return compile_color_gradient_lattice_boltzmann_problem(
        ColorGradientLatticeBoltzmannProblem("n-color", 2),
        discretization,
        method,
        LatticeBoltzmannBoundaryPlan(),
        time_step=1.0,
    )


def _coordinates(compiled: Any) -> tuple[np.ndarray, np.ndarray]:
    grid = compiled.discretization.grid
    points = np.asarray(grid.points)
    return points[:, 0].reshape(grid.shape), points[:, 1].reshape(grid.shape)


def _disk(
    x: np.ndarray, y: np.ndarray, center: tuple[float, float], radius: float
) -> np.ndarray:
    distance = np.hypot(x - center[0], y - center[1])
    return 0.5 * (1.0 + np.tanh((radius - distance) / 2.0))


def _two_drops(compiled: Any, gap: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = _coordinates(compiled)
    middle = 0.5 * compiled.discretization.grid.shape[0]
    offset = radius + 0.5 * gap
    left = _disk(x, y, (middle - offset, 0.5 * y.max() + 0.5), radius)
    right = _disk(x, y, (middle + offset, 0.5 * y.max() + 0.5), radius)
    return left, right


def _step(compiled: Any) -> Any:
    return jax.jit(
        lambda state, parameters, size: compiled.dynamics.step_detailed(
            0, 0.0, state, size, parameters
        )
    )


def test_ternary_step_conserves_component_masses_and_total_momentum() -> None:
    components = ("a", "b", "c")
    compiled = _compiled(
        components, (32, 32), NearContactRepulsionPlan((("a", "b"),), interaction_range=3)
    )
    x, y = _coordinates(compiled)
    a = _disk(x, y, (11.0, 16.0), 6.0)
    b = (1.0 - a) * _disk(x, y, (20.0, 17.5), 5.0)
    densities = np.stack((a, b, 1.0 - a - b))
    tension = InterfaceTensionMatrix(
        components,
        np.asarray([[0.0, 0.004, 0.006], [0.004, 0.0, 0.005], [0.006, 0.005, 0.0]]),
    )
    parameters = ColorGradientLBMRuntimeParameters(
        1.0 / 6.0, tension, near_contact_strength=np.asarray([0.002])
    )
    state = compiled.initialize_state(densities, jnp.asarray((0.02, -0.01)), parameters)
    initial = compiled.dynamics.scalar_diagnostics(0, 0.0, state, parameters)

    @jax.jit
    def rollout(state: Any) -> tuple[Any, Any]:
        def body(current: Any, index: Any) -> tuple[Any, Any]:
            result = compiled.dynamics.step_detailed(index, 0.0, current, 1.0, parameters)
            return result.accepted_state, (result.successful, result.near_contact)

        return jax.lax.scan(body, state, jnp.arange(25))

    final_state, (successful, near_contact) = rollout(state)
    final = compiled.dynamics.scalar_diagnostics(0, 0.0, final_state, parameters)

    assert bool(jnp.all(successful))
    assert int(jnp.min(near_contact.active_pair_count)) > 0
    assert float(jnp.max(near_contact.net_force_residual)) <= 1e-13
    assert float(final.capillary_net_force_residual) <= 1e-12
    np.testing.assert_allclose(
        final.component_masses, initial.component_masses, rtol=1e-13
    )
    np.testing.assert_allclose(initial.total_momentum, (20.48, -10.24), rtol=1e-12)
    np.testing.assert_allclose(
        final.total_momentum, initial.total_momentum, rtol=0.0, atol=1e-11
    )


def test_near_contact_repulsion_separates_facing_drops_and_ledgers_relative_work() -> (
    None
):
    compiled = _compiled(
        ("water", "oil"),
        (48, 32),
        NearContactRepulsionPlan((("oil", "oil"),), interaction_range=4),
    )
    tension = InterfaceTensionMatrix(("water", "oil"), _BINARY_TENSION)
    parameters = ColorGradientLBMRuntimeParameters(
        1.0 / 6.0, tension, near_contact_strength=np.asarray([0.01])
    )
    step = _step(compiled)
    left, right = _two_drops(compiled, 4.0, 8.0)
    densities = np.stack((1.0 - left - right, left + right))
    approach = np.zeros((*left.shape, 2))
    approach[..., 0] = 0.01 * (left - right)
    frame = np.asarray((0.01, 0.005))

    approaching = compiled.initialize_state(densities, approach, parameters)
    receding = compiled.initialize_state(densities, -approach, parameters)
    translated = compiled.initialize_state(densities, approach + frame, parameters)
    result = step(approaching, parameters, 1.0)
    reverse = step(receding, parameters, 1.0)
    moving_frame = step(translated, parameters, 1.0)

    evidence = result.near_contact
    assert bool(result.successful)
    assert int(evidence.active_pair_count) > 0
    assert float(evidence.net_force_residual) <= 1e-13
    force = np.asarray(
        compiled.macroscopic_state(approaching, parameters).near_contact_force
    )
    x, _ = _coordinates(compiled)
    left_push = np.sum(force[..., 0][x < 24.0])
    right_push = np.sum(force[..., 0][x > 24.0])
    assert left_push < 0.0 < right_push
    np.testing.assert_allclose(left_push, -right_push, rtol=1e-12)
    # Approaching films are decelerated (negative work); receding films are pushed.
    assert float(evidence.power) < 0.0 < float(reverse.near_contact.power)
    np.testing.assert_allclose(reverse.near_contact.power, -evidence.power, rtol=1e-10)
    # A pairwise antisymmetric force does no work on uniform translation.
    np.testing.assert_allclose(
        moving_frame.near_contact.power,
        evidence.power,
        rtol=0.0,
        atol=1e-12 * abs(float(evidence.power)),
    )
    np.testing.assert_allclose(
        result.accepted_state.near_contact_work, evidence.work, rtol=1e-15
    )
    rejected = step(result.accepted_state, parameters, 2.0)
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(
        rejected.accepted_state.near_contact_work, result.accepted_state.near_contact_work
    )

    single = compiled.initialize_state(
        np.stack((1.0 - left, left)), jnp.zeros((2,)), parameters
    )
    isolated = step(single, parameters, 1.0)
    assert int(isolated.near_contact.active_pair_count) == 0


def test_component_identity_and_repulsion_shape_are_refused() -> None:
    compiled = _compiled(
        ("water", "oil"),
        (16, 16),
        NearContactRepulsionPlan((("oil", "oil"),), interaction_range=2),
    )
    densities = np.stack((np.full((16, 16), 0.5), np.full((16, 16), 0.5)))
    swapped = InterfaceTensionMatrix(("oil", "water"), _BINARY_TENSION)
    with pytest.raises(ValueError, match="label ids"):
        compiled.initialize_state(
            densities,
            jnp.zeros((2,)),
            ColorGradientLBMRuntimeParameters(
                1.0 / 6.0, swapped, near_contact_strength=np.asarray([0.01])
            ),
        )
    tension = InterfaceTensionMatrix(("water", "oil"), _BINARY_TENSION)
    with pytest.raises(ValueError, match="repelling pair"):
        compiled.initialize_state(
            densities,
            jnp.zeros((2,)),
            ColorGradientLBMRuntimeParameters(1.0 / 6.0, tension),
        )
    with pytest.raises(ValueError, match="unknown components"):
        ColorGradientLBMMethod(
            LatticeBoltzmannMethodPlan(BGKCollisionPlan(), forcing=GuoForcingPlan()),
            ("water", "oil"),
            near_contact=NearContactRepulsionPlan((("oil", "air"),)),
        )
    with pytest.raises(ValueError, match="symmetric"):
        ColorGradientLBMMethod(
            LatticeBoltzmannMethodPlan(BGKCollisionPlan(), forcing=GuoForcingPlan()),
            ("water", "oil", "air"),
            recoloring_strength=[[0.0, 0.7, 0.7], [0.6, 0.0, 0.7], [0.7, 0.7, 0.0]],
        )
