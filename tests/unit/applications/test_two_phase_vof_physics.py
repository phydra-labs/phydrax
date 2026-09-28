#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


two_phase_api = phx.applications.two_phase_flow


def _grid(
    shape: tuple[int, ...],
    upper: tuple[float, ...],
    periodic: tuple[bool, ...],
) -> Any:
    names = ("x", "y", "z")[: len(shape)]
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=flag)
            for count, flag in zip(shape, periodic, strict=True)
        ),
        axis_names=names,
    ).prepare(jnp.asarray(((0.0,) * len(shape), upper)))
    return phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()


def _fraction(discretization: Any, inside: Any, samples: int = 12) -> Any:
    """Sub-sampled cell fraction of the region ``inside(points) -> bool``."""

    centers = np.asarray(discretization.cell_centers)
    widths = [
        float(np.asarray(axis.interval_widths)[0])
        for axis in discretization.grid.structured_axes
    ]
    offsets = (np.arange(samples) + 0.5) / samples - 0.5
    grids = np.meshgrid(*([offsets] * centers.shape[-1]), indexing="ij")
    local = np.stack(grids, axis=-1).reshape((-1, centers.shape[-1])) * np.asarray(widths)
    points = centers[..., None, :] + local
    return jnp.asarray(np.mean(inside(points), axis=-1))


def _prepared(discretization: Any, **options: Any) -> Any:
    material = two_phase_api.TwoPhaseMaterialPlan(
        liquid_density=options.pop("liquid_density", 1.0),
        gas_density=options.pop("gas_density", 1.0),
        liquid_viscosity=options.pop("liquid_viscosity", 0.0),
        gas_viscosity=options.pop("gas_viscosity", 0.0),
        surface_tension=options.pop("surface_tension", 0.0),
    )
    return two_phase_api.IncompressibleTwoPhaseVOFPlan(
        discretization, material, tolerance=1.0e-10, maximum_iterations=4000, **options
    ).prepare()


def _run(
    two_phase: Any,
    alpha: Any,
    steps: int,
    step_size: float,
    velocity: Any = None,
    record: Any = None,
) -> tuple[Any, list[Any]]:
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(two_phase.initial_state(alpha, velocity))
    step = eqx.filter_jit(method.step)
    history = []
    for index in range(steps):
        result = step(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * step_size),
            continuation,
            jnp.asarray(step_size),
            None,
        )
        assert bool(result.successful), f"step {index} refused"
        continuation = result.accepted_state
        if record is not None:
            history.append(record(continuation))
    return continuation, history


def test_static_drop_carries_laplace_jump_with_converging_parasitic_current() -> None:
    errors = []
    speeds = []
    for count in (16, 32):
        discretization = _grid((count, count), (1.0, 1.0), (True, True))
        two_phase = _prepared(discretization, liquid_density=1000.0, surface_tension=1.0)
        alpha = _fraction(
            discretization,
            lambda points: np.sum((points - 0.5) ** 2, axis=-1) < 0.25**2,
        )
        step = 0.5 * np.sqrt(500.5 / count**3 / (2.0 * np.pi))
        continuation, _ = _run(two_phase, alpha, 4, step)
        fraction = two_phase.alpha(continuation.state)
        pressure = continuation.pressure
        jump = jnp.mean(pressure[fraction == 1.0]) - jnp.mean(pressure[fraction == 0.0])
        errors.append(abs(float(jump) - 4.0) / 4.0)
        speeds.append(float(continuation.evidence.parasitic_velocity))
    assert errors[1] < errors[0]
    assert errors[1] < 3.0e-2
    assert speeds[1] < speeds[0]


def _drop_mode(two_phase: Any, discretization: Any) -> Any:
    x = discretization.cell_centers[..., 0] - 0.5
    y = discretization.cell_centers[..., 1] - 0.5

    def mode(continuation: Any) -> float:
        fraction = two_phase.alpha(continuation.state)
        return float(jnp.sum(fraction * (x**2 - y**2)) / jnp.sum(fraction))

    return mode


def test_oscillating_drop_restores_with_frequency_converging_in_time() -> None:
    radius, amplitude = 0.25, 0.05
    period = 2.0 * np.pi / np.sqrt(6.0 / (2.0 * radius**3))
    discretization = _grid((24, 24), (1.0, 1.0), (True, True))
    two_phase = _prepared(discretization, surface_tension=1.0)

    def inside(points: Any) -> Any:
        offset = points - 0.5
        angle = np.arctan2(offset[..., 1], offset[..., 0])
        return np.hypot(offset[..., 0], offset[..., 1]) < radius * (
            1.0 + amplitude * np.cos(2.0 * angle)
        )

    alpha = _fraction(discretization, inside, samples=16)
    record = _drop_mode(two_phase, discretization)
    half_periods = []
    for divisions in (320, 640):
        step = period / divisions
        _, history = _run(two_phase, alpha, int(0.65 * divisions), step, record=record)
        values = np.asarray(history)
        index = int(np.argmin(values))
        assert 0 < index < values.size - 1
        left, middle, right = values[index - 1 : index + 2]
        shift = 0.5 * (left - right) / (left - 2.0 * middle + right)
        half_periods.append((index + 1 + shift) * step)
        assert values[index] < -0.7 * values[0]
    assert abs(half_periods[1] - half_periods[0]) < 5.0e-3 * half_periods[1]
    assert abs(half_periods[1] / (0.5 * period) - 1.0) < 0.15


def test_capillary_wave_phase_converges_to_inviscid_dispersion() -> None:
    amplitude = 0.01
    wavenumber = 2.0 * np.pi
    omega = np.sqrt(wavenumber**3 / (2.0 / np.tanh(wavenumber * 0.5)))
    period = 2.0 * np.pi / omega
    discretization = _grid((16, 16), (1.0, 1.0), (True, False))
    two_phase = _prepared(discretization, surface_tension=1.0)
    alpha = _fraction(
        discretization,
        lambda points: (
            points[..., 1] < 0.5 + amplitude * np.cos(wavenumber * points[..., 0])
        ),
        samples=24,
    )
    x = discretization.cell_centers[..., 0]

    def mode(continuation: Any) -> float:
        height = jnp.sum(two_phase.alpha(continuation.state), axis=1) / 16.0
        return float(2.0 * jnp.mean(height * jnp.cos(wavenumber * x[:, 0])))

    crossings = []
    amplitudes = []
    for divisions in (200, 400):
        step = period / divisions
        _, history = _run(two_phase, alpha, int(0.6 * divisions), step, record=mode)
        values = np.asarray(history)
        sign = np.flatnonzero((values[:-1] > 0.0) & (values[1:] <= 0.0))[0]
        fraction = values[sign] / (values[sign] - values[sign + 1])
        crossings.append((sign + 1 + fraction) * step)
        amplitudes.append(float(np.min(values)))
    assert abs(crossings[1] - crossings[0]) < 1.0e-2 * crossings[1]
    assert abs(crossings[1] / (0.25 * period) - 1.0) < 0.15
    assert abs(amplitudes[1] - amplitudes[0]) < 0.05 * abs(amplitudes[1])
    assert -amplitudes[1] > 0.6 * amplitude


def test_flat_hydrostatic_interface_stays_at_rest_with_absolute_pressure() -> None:
    discretization = _grid((8, 16), (1.0, 2.0), (True, False))
    two_phase = _prepared(
        discretization,
        liquid_density=1000.0,
        gravity=(0.0, -9.81),
        hydrostatic_reference=(0.5, 1.0),
        reference_pressure=1.0e5,
    )
    y = np.asarray(discretization.cell_centers[..., 1])
    alpha = jnp.asarray(np.clip((0.5625 - (y - 0.0625)) / 0.125, 0.0, 1.0))
    continuation, _ = _run(two_phase, alpha, 20, 1.0e-3)

    assert float(continuation.evidence.parasitic_velocity) < 1.0e-8
    view = two_phase.view(continuation.state, continuation.pressure)
    column = np.asarray(view.absolute_pressure[0])
    level = 0.5625
    expected = np.where(
        y[0] > level,
        1.0e5 + 9.81 * (1.0 - y[0]),
        1.0e5 + 9.81 * (1.0 - level) + 1000.0 * 9.81 * (level - y[0]),
    )
    pure = np.isin(np.asarray(view.alpha[0]), (0.0, 1.0))
    np.testing.assert_allclose(column[pure], expected[pure], rtol=1.0e-12)


def _bubble_rise(gravity: float, count: int, steps: int, step_size: float) -> Any:
    discretization = _grid((count, count), (1.0, 1.0), (False, False))
    two_phase = _prepared(
        discretization,
        liquid_density=1000.0,
        gravity=(0.0, gravity),
        hydrostatic_reference=(0.5, 0.5),
    )
    alpha = _fraction(
        discretization,
        lambda points: np.sum((points - 0.5) ** 2, axis=-1) >= 0.25**2,
    )
    continuation, _ = _run(two_phase, alpha, steps, step_size)
    velocity = np.asarray(two_phase.velocity(continuation.state)[1])
    gas = 1.0 - np.asarray(two_phase.alpha(continuation.state))
    cell_velocity = 0.5 * (velocity[:, :-1] + velocity[:, 1:])
    return float(np.sum(gas * cell_velocity) / np.sum(gas)), continuation.ledger


def test_gravity_direction_sets_bubble_acceleration_and_ledger_converges() -> None:
    up, coarse = _bubble_rise(-9.81, 16, 16, 5.0e-4)
    down, _ = _bubble_rise(9.81, 16, 16, 5.0e-4)
    _, fine = _bubble_rise(-9.81, 32, 16, 5.0e-4)

    assert up > 0.0
    np.testing.assert_allclose(down, -up, rtol=1.0e-8)
    for ledger in (coarse, fine):
        work = float(ledger.work_energy_residual)
        assert abs(work) < 1.0e-2 * float(ledger.kinetic_energy_change)

    def relative_defect(ledger: Any) -> float:
        return abs(float(ledger.gravitational_energy_residual)) / abs(
            float(ledger.gravitational_energy_change)
        )

    assert relative_defect(fine) < relative_defect(coarse)
    assert relative_defect(fine) < 0.15


def test_two_layer_couette_profile_and_nonnegative_dissipation() -> None:
    errors = []
    for count in (8, 16):
        discretization = _grid((4, count), (1.0, 1.0), (True, False))
        walls = (
            phx.discretization.MACBoundarySide("y", "lower", "no-slip"),
            phx.discretization.MACBoundarySide(
                "y",
                "upper",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(jnp.asarray([1.0, 0.0])),
            ),
        )
        two_phase = _prepared(
            discretization,
            liquid_viscosity=1.0,
            gas_viscosity=0.25,
            wall_sides=walls,
        )
        y = np.asarray(discretization.cell_centers[..., 1])
        alpha = jnp.asarray((y < 0.5).astype(np.float64))
        step = 0.4 / count
        continuation, history = _run(
            two_phase,
            alpha,
            int(3.0 / step),
            step,
            record=lambda state: float(state.ledger.viscous_dissipation),
        )
        assert np.all(np.diff(np.asarray(history)) >= 0.0)
        velocity = np.asarray(two_phase.velocity(continuation.state)[0])[0]
        stress = 1.0 / (0.5 / 1.0 + 0.5 / 0.25)
        exact = np.where(
            y[0] < 0.5, stress * y[0], stress * 0.5 + stress * (y[0] - 0.5) / 0.25
        )
        errors.append(float(np.max(np.abs(velocity - exact))))
    assert errors[1] < 0.6 * errors[0]
    assert errors[1] < 3.0e-2


def test_inviscid_tensionless_plan_disables_viscous_and_interfacial_stages() -> None:
    discretization = _grid((8, 8), (1.0, 1.0), (True, True))
    two_phase = _prepared(discretization)
    assert two_phase.viscosity is None
    assert two_phase.curvature_plan is None
    alpha = jnp.zeros((8, 8)).at[2:6, 2:6].set(1.0)
    continuation, _ = _run(two_phase, alpha, 2, 1.0e-3)
    assert float(continuation.ledger.viscous_dissipation) == 0.0
    assert float(continuation.ledger.capillary_work) == 0.0
    assert float(continuation.evidence.capillary_pressure_jump) == 0.0


def _translation_error(shape: tuple[int, ...], inside: Any, samples: int) -> Any:
    count = shape[0]
    discretization = _grid(shape, (1.0,) * len(shape), (True,) * len(shape))
    two_phase = _prepared(discretization)
    alpha = _fraction(discretization, inside, samples=samples)
    # One full period along every periodic axis returns the body to its start.
    shift = (1.0,) * len(shape)
    velocity = tuple(
        jnp.full(layout.shape, shift[axis])
        for axis, layout in enumerate(discretization.face_layouts)
    )
    step = 0.25 / count
    minimum = [1.0]
    maximum = [0.0]

    def record(continuation: Any) -> None:
        fraction = two_phase.alpha(continuation.state)
        minimum[0] = min(minimum[0], float(jnp.min(fraction)))
        maximum[0] = max(maximum[0], float(jnp.max(fraction)))

    continuation, _ = _run(
        two_phase, alpha, int(round(1.0 / step)), step, velocity, record
    )
    final = two_phase.alpha(continuation.state)
    return (
        float(jnp.mean(jnp.abs(final - alpha))),
        abs(float(jnp.sum(final - alpha))) / float(jnp.sum(alpha)),
        minimum[0],
        maximum[0],
    )


def test_geometric_transport_translates_circle_slot_and_sphere_boundedly() -> None:
    tolerance = 1024.0 * np.finfo(np.float64).eps

    def circle(points: Any) -> Any:
        return np.sum((points - 0.5) ** 2, axis=-1) < 0.25**2

    def slotted(points: Any) -> Any:
        slot = (np.abs(points[..., 0] - 0.5) < 0.06) & (points[..., 1] < 0.6)
        return circle(points) & ~slot

    coarse = _translation_error((16, 16), circle, 12)
    fine = _translation_error((32, 32), circle, 12)
    slot = _translation_error((32, 32), slotted, 12)
    sphere = _translation_error(
        (10, 10, 10), lambda points: np.sum((points - 0.5) ** 2, axis=-1) < 0.3**2, 6
    )
    for error, volume, minimum, maximum in (coarse, fine, slot, sphere):
        assert volume < 1.0e-12
        assert minimum >= -tolerance
        assert maximum <= 1.0 + tolerance
    assert fine[0] < coarse[0] / 2.0
    assert slot[0] < 2.0e-2


def test_solid_body_rotation_keeps_alpha_bounded_and_volume_conserved() -> None:
    discretization = _grid((32, 32), (1.0, 1.0), (True, True))
    two_phase = _prepared(discretization)
    alpha = _fraction(
        discretization,
        lambda points: np.sum((points - (0.5, 0.72)) ** 2, axis=-1) < 0.15**2,
    )
    velocity = (
        -(discretization.face_centers[0][..., 1] - 0.5),
        discretization.face_centers[1][..., 0] - 0.5,
    )
    tolerance = 1024.0 * np.finfo(np.float64).eps

    def record(continuation: Any) -> tuple[float, float]:
        fraction = two_phase.alpha(continuation.state)
        return float(jnp.min(fraction)), float(jnp.max(fraction))

    continuation, history = _run(two_phase, alpha, 60, 0.02, velocity, record)
    bounds = np.asarray(history)
    assert bounds[:, 0].min() >= -tolerance
    assert bounds[:, 1].max() <= 1.0 + tolerance
    np.testing.assert_allclose(
        jnp.sum(continuation.state.liquid_content),
        jnp.sum(two_phase.initial_state(alpha).liquid_content),
        rtol=1.0e-9,
    )


def test_accepted_continuation_reuses_the_initial_step_signature() -> None:
    discretization = _grid((12, 12), (1.0, 1.0), (True, True))
    two_phase = _prepared(discretization, surface_tension=0.5)
    alpha = _fraction(
        discretization,
        lambda points: np.sum((points - 0.5) ** 2, axis=-1) < 0.3**2,
    )
    method = two_phase_api.IncompressibleTwoPhaseVOFMethod(two_phase)
    initial = method.initial_continuation(two_phase.initial_state(alpha))
    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        initial,
        jnp.asarray(1.0e-3),
        None,
    )
    assert bool(result.successful)

    def signature(tree: Any) -> Any:
        leaves, structure = jax.tree_util.tree_flatten(eqx.filter(tree, eqx.is_array))
        return structure, [(leaf.shape, leaf.dtype) for leaf in leaves]

    assert signature(result.accepted_state) == signature(initial)
