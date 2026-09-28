#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frequency-domain moving charges on compatible cochains.

Independent references: Whitney forms reproduce uniform fields, so pairing the
load with the circulations of a uniform ``u`` must return
``q (u·d̂) ∫ exp(iωs/v) ds``; the Frank–Tamm law
``d²W/(dω dl) = (q² μ ω / 4π)(1 − 1/(β² n²))`` (Jackson 13.48, SI, one-sided);
the 2-D line-charge Cherenkov slab flux ``(λ²/2π) η √(1 − 1/(β²n²))``; the
analytic bound/radiating line-charge field; and the Ginzburg–Frank transition
radiation of a charge entering a perfect conductor,
``d²W/(dω dΩ) = q² β² sin²θ / (4π³ ε₀ c (1 − β² cos²θ)²)``.
Code units ε₀ = μ₀ = c = 1 throughout.
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.linalg import (
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    SolveResourcePolicy,
    SparseLU,
)


mx = phx.solver.maxwell
em = phx.electromagnetics
OMEGA = 2.0 * np.pi


def _bridge(
    counts: tuple[int, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    periodic: tuple[bool, ...],
) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=flag)
            for count, flag in zip(counts, periodic, strict=True)
        ),
        axis_names=tuple("xyz"[: len(counts)]),
    ).prepare(jnp.asarray([lower, upper]))
    return phx.discretization.StructuredCochainBridge(grid)


def _direct_policy(resources: SolveResourcePolicy | None = None) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        SparseLU(provider="scipy-superlu"),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
        resources=resources,
    )


# The 3-D direct solves' symbolic (George–Ng) fill bound is about 1.2 GB, above
# the 512 MiB default factorization budget; SuperLU realizes about half of it.
_THREE_D_RESOURCES = SolveResourcePolicy(factorization_bytes=2**31)


def _uniform_circulations(bridge: Any, field: np.ndarray) -> Any:
    axes = bridge.grid.structured_axes
    return bridge.pack(
        1,
        tuple(
            field[axis]
            * float(np.asarray(axes[axis].interval_widths)[0])
            * np.ones(shape)
            for axis, shape in enumerate(bridge.orientation_shapes[1])
        ),
    )


def _clipped_path(
    origin: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float]:
    entries = (lower - origin) / direction
    exits = (upper - origin) / direction
    return float(np.max(np.minimum(entries, exits))), float(
        np.min(np.maximum(entries, exits))
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_whitney_load_reproduces_uniform_field_work_and_conserves_charge(
    dimension: int,
) -> None:
    counts = (7, 6, 5)[:dimension]
    lower = np.asarray((-0.3, -0.2, -0.4)[:dimension])
    upper = np.asarray((0.8, 0.7, 0.5)[:dimension])
    bridge = _bridge(counts, tuple(lower), tuple(upper), (False,) * dimension)
    layout = mx.MaxwellCochainLayout(bridge, "tez" if dimension == 2 else "full_3d")
    origin = np.asarray((0.11, 0.07, 0.03)[:dimension])
    direction = np.asarray((0.8, 0.45, -0.39)[:dimension])
    direction = direction / np.linalg.norm(direction)
    speed, charge, omega = 0.7, 1.3, 9.0
    source = mx.MaxwellMovingChargePlan(
        bridge, layout, charge=charge, speed=speed, origin=origin, direction=direction
    ).prepare()
    uniform = np.asarray((0.4, -1.1, 0.7)[:dimension])
    start, stop = _clipped_path(origin, direction, lower, upper)
    wavenumber = omega / speed
    expected = (
        charge
        * (uniform @ direction)
        * (np.exp(1j * wavenumber * stop) - np.exp(1j * wavenumber * start))
        / (1j * wavenumber)
    )
    work = jnp.sum(source.edge_load(omega) * _uniform_circulations(bridge, uniform))
    np.testing.assert_allclose(work, expected, rtol=1e-12)
    assert source.open_endpoints == 2
    assert float(source.continuity_defect(omega)) < 1e-12


def test_periodic_path_is_one_closed_pass_at_commensurate_frequencies() -> None:
    speed = 0.8
    length = 2.0 * np.pi * speed / OMEGA
    bridge = _bridge((8, 6), (0.0, -0.3), (length, 0.3), (True, False))
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    source = mx.MaxwellMovingChargePlan(
        bridge, layout, charge=1.0, speed=speed, origin=[0.013, 0.021], direction=[1, 0]
    ).prepare()
    assert source.open_endpoints == 0
    # No endpoint exclusions: continuity holds at every node of the closed pass.
    assert float(source.continuity_defect(2.0 * OMEGA)) < 1e-12
    with pytest.raises(eqx.EquinoxRuntimeError, match="integer multiple of 2π"):
        source.current(1.5 * OMEGA)


def _line_problem(
    permittivity: Any, speed: float, *, formulation: str = "total-field", **options: Any
) -> tuple[Any, Any, Any, float]:
    length = 2.0 * np.pi * speed / OMEGA
    cells, transverse, layers = 32, 144, 16
    step = length / cells
    bridge = _bridge(
        (cells, transverse),
        (0.0, -transverse * step / 2),
        (length, transverse * step / 2),
        (True, False),
    )
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    material = mx.DiagonalMaxwellConstitutivePlan(permittivity=permittivity).prepare(
        bridge.cochain, layout
    )
    source = mx.MaxwellMovingChargePlan(
        bridge,
        layout,
        charge=1.0,
        speed=speed,
        origin=[0.3 * step, 0.5 * step],
        direction=[1.0, 0.0],
    )
    plan = mx.FrequencyMovingChargePlan(
        source,
        material,
        formulation=formulation,  # ty: ignore[invalid-argument-type]
        stretching=mx.MaxwellCPMLPlan((0, layers), target_reflection=1e-6),
        policy=_direct_policy(),
        **options,
    )
    return bridge, plan.prepare().solve(OMEGA), source, step


def _analytic_rows(
    bridge: Any, result: Any, permittivity: float, speed: float, step: float
) -> list[float]:
    cells, transverse = 32, 144
    analytic = em.UniformMotionFieldPlan(
        "line",
        em.UniformMotionMedium([OMEGA], permittivity, 1.0),
        charge=1.0,
        speed=speed,
        origin=[0.3 * step, 0.5 * step],
        direction=[1.0, 0.0],
    )
    along, _ = bridge.unpack(1, result.electric)
    centers = (np.arange(cells) + 0.5) * step
    errors = []
    for row in (transverse // 2 + 8, transverse // 2 - 20):
        height = -transverse * step / 2 + row * step
        points = np.stack((centers, np.full(cells, height)), axis=1)
        reference = analytic.evaluate(jnp.asarray(points)).electric[0, :, 0] * step
        errors.append(
            float(jnp.linalg.norm(along[:, row] - reference) / jnp.linalg.norm(reference))
        )
    return errors


def test_line_charge_cherenkov_matches_analytic_field_and_power() -> None:
    permittivity, speed = 4.0, 0.8
    bridge, result, _, step = _line_problem(permittivity, speed)
    evidence = result.evidence
    assert bool(evidence.converged) and bool(evidence.radiating)
    assert float(evidence.continuity_defect) < 1e-12
    length = 32 * step
    beta_n = speed * np.sqrt(permittivity)
    expected = (
        np.sqrt(1.0 / permittivity) * np.sqrt(1.0 - 1.0 / beta_n**2) / (2.0 * np.pi)
    )
    power = 2.0 / np.pi * float(result.ledger.source_power) / length
    np.testing.assert_allclose(power, expected, rtol=2e-2)
    np.testing.assert_allclose(
        result.ledger.absorbed_power, result.ledger.source_power, rtol=1e-10
    )
    assert max(_analytic_rows(bridge, result, permittivity, speed, step)) < 4e-2


def test_bound_line_field_does_no_work_below_threshold() -> None:
    speed = 0.6
    bridge, result, _, step = _line_problem(1.0, speed)
    evidence = result.evidence
    assert not bool(evidence.radiating)
    assert bool(evidence.bound_field_contained)
    # Bound-field reach γβλ = 2πγv/ω.
    np.testing.assert_allclose(
        evidence.bound_extent, 2.0 * np.pi * speed / np.sqrt(1 - speed**2) / OMEGA
    )
    length = 32 * step
    radiating_scale = 1.0 / (2.0 * np.pi)
    work = 2.0 / np.pi * float(result.ledger.source_power) / length
    assert abs(work) < 1e-4 * radiating_scale
    assert max(_analytic_rows(bridge, result, 1.0, speed, step)) < 2e-2


def test_scattered_field_total_equals_total_field_formulation() -> None:
    # A line charge in vacuum above a glass half-space: the bound field couples
    # into the glass (βn = 1.2) and radiates there. Only the glass is a source
    # in the scattered-field formulation.
    speed = 0.8
    length = 2.0 * np.pi * speed / OMEGA
    step = length / 32
    bridge = _bridge((32, 144), (0.0, -72 * step), (length, 72 * step), (True, False))
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    along, across = bridge.orientation_shapes[1]
    heights = np.concatenate(
        (
            np.broadcast_to(np.arange(along[1]) * step - 72 * step, along).reshape(-1),
            np.broadcast_to(
                (np.arange(across[1]) + 0.5) * step - 72 * step, across
            ).reshape(-1),
        )
    )
    glass = mx.DiagonalMaxwellConstitutivePlan(
        permittivity=np.where(heights < -0.3, 2.25, 1.0)
    ).prepare(bridge.cochain, layout)
    vacuum = mx.DiagonalMaxwellConstitutivePlan().prepare(bridge.cochain, layout)
    source = mx.MaxwellMovingChargePlan(
        bridge,
        layout,
        charge=1.0,
        speed=speed,
        origin=[0.3 * step, 0.5 * step],
        direction=[1, 0],
    )
    stretching = mx.MaxwellCPMLPlan((0, 16), target_reflection=1e-6)
    total = (
        mx.FrequencyMovingChargePlan(
            source, glass, stretching=stretching, policy=_direct_policy()
        )
        .prepare()
        .solve(OMEGA)
    )
    scattered = (
        mx.FrequencyMovingChargePlan(
            source,
            glass,
            formulation="scattered-field",
            background=vacuum,
            stretching=stretching,
            policy=_direct_policy(),
        )
        .prepare()
        .solve(OMEGA)
    )
    assert float(scattered.evidence.source_distance) > 5.0
    assert scattered.scattered_electric is not None
    # Rows away from the path, where both carry the same physical field.
    first, _ = bridge.unpack(1, total.electric)
    second, _ = bridge.unpack(1, scattered.electric)
    rows = np.r_[20:60, 90:120]
    np.testing.assert_allclose(
        np.asarray(second[:, rows]),
        np.asarray(first[:, rows]),
        rtol=0,
        atol=3e-2 * float(jnp.max(jnp.abs(first[:, rows]))),
    )
    # The glass radiates the same power into the absorbing layers.
    np.testing.assert_allclose(
        scattered.ledger.absorbed_power, total.ledger.absorbed_power, rtol=5e-2
    )


def test_point_charge_frank_tamm_power_per_length() -> None:
    permittivity, speed = 2.25, 0.8
    length = 2.0 * np.pi * speed / OMEGA
    cells, transverse, layers = 12, 24, 6
    step = length / cells
    half = transverse * step / 2
    bridge = _bridge(
        (cells, transverse, transverse),
        (0.0, -half, -half),
        (length, half, half),
        (True, False, False),
    )
    layout = mx.MaxwellCochainLayout(bridge, "full_3d")
    material = mx.DiagonalMaxwellConstitutivePlan(permittivity=permittivity).prepare(
        bridge.cochain, layout
    )
    source = mx.MaxwellMovingChargePlan(
        bridge,
        layout,
        charge=1.0,
        speed=speed,
        origin=[0.0, 0.5 * step, 0.5 * step],
        direction=[1.0, 0.0, 0.0],
    )
    result = (
        mx.FrequencyMovingChargePlan(
            source,
            material,
            stretching=mx.MaxwellCPMLPlan((0, layers, layers), target_reflection=1e-6),
            policy=_direct_policy(_THREE_D_RESOURCES),
        )
        .prepare()
        .solve(OMEGA)
    )
    frank_tamm = OMEGA / (4.0 * np.pi) * (1.0 - 1.0 / (speed**2 * permittivity))
    power = 2.0 / np.pi * float(result.ledger.source_power) / length
    np.testing.assert_allclose(power, frank_tamm, rtol=3e-2)
    assert float(result.ledger.relative_residual) < 1e-10
    assert float(result.evidence.continuity_defect) < 1e-12
    assert result.evidence.open_endpoints == 0


def test_transition_radiation_into_a_conductor_matches_ginzburg_frank() -> None:
    # A charge at β = 0.5 enters a thick perfect conductor (all edges at z ≤ 0)
    # through a cell centre. Only the conductor surface radiates in the
    # scattered-field formulation; the absorbed scattered power is the backward
    # transition radiation. At h = λ/10 the solve gives 0.90 of the Ginzburg–Frank
    # energy and 0.956 at h = λ/13.3 (second-order convergence).
    speed, step, side, rise, layers = 0.5, 0.1, 12, 7, 6
    width = side + 2 * layers
    lower = -width * step / 2
    bridge = _bridge(
        (width, width, layers + rise + layers),
        (lower, lower, -layers * step),
        (-lower, -lower, (rise + layers) * step),
        (False, False, False),
    )
    layout = mx.MaxwellCochainLayout(bridge, "full_3d")
    vacuum = mx.DiagonalMaxwellConstitutivePlan().prepare(bridge.cochain, layout)
    below = []
    for axis, shape in enumerate(bridge.orientation_shapes[1]):
        top = np.indices(shape)[2] * step - layers * step + (step if axis == 2 else 0.0)
        below.append((top <= 1e-12).reshape(-1))
    conductor = mx.MaxwellBoundaryPlan("pec", support=np.concatenate(below))
    source = mx.MaxwellMovingChargePlan(
        bridge,
        layout,
        charge=1.0,
        speed=speed,
        origin=[0.5 * step, 0.5 * step, 0.0],
        direction=[0.0, 0.0, -1.0],
    )
    result = (
        mx.FrequencyMovingChargePlan(
            source,
            vacuum,
            formulation="scattered-field",
            background=vacuum,
            stretching=mx.MaxwellCPMLPlan((layers,) * 3, target_reflection=1e-6),
            boundaries=(conductor,),
            policy=_direct_policy(_THREE_D_RESOURCES),
        )
        .prepare()
        .solve(OMEGA)
    )
    angles = np.linspace(0.0, np.pi / 2, 20001)
    pattern = np.sin(angles) ** 3 / (1.0 - speed**2 * np.cos(angles) ** 2) ** 2
    ginzburg_frank = (
        speed**2 / (4.0 * np.pi**3) * 2.0 * np.pi * np.trapezoid(pattern, angles)
    )
    energy = 2.0 / np.pi * float(result.ledger.absorbed_power)
    np.testing.assert_allclose(energy, ginzburg_frank, rtol=0.12)
    evidence = result.evidence
    # The path crosses the conductor surface at a cell centre.
    np.testing.assert_allclose(evidence.source_distance, 0.5)
    assert bool(jnp.all(result.electric[np.concatenate(below)] == 0.0))
    np.testing.assert_allclose(
        evidence.bound_extent, 2.0 * np.pi * speed / np.sqrt(1.0 - speed**2) / OMEGA
    )


def test_krylov_solution_matches_direct_with_convergence_evidence() -> None:
    speed = 0.8
    length = 2.0 * np.pi * speed / OMEGA
    step = length / 8
    bridge = _bridge((8, 40), (0.0, -20 * step), (length, 20 * step), (True, False))
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    material = mx.DiagonalMaxwellConstitutivePlan(permittivity=4.0).prepare(
        bridge.cochain, layout
    )
    source = mx.MaxwellMovingChargePlan(
        bridge,
        layout,
        charge=1.0,
        speed=speed,
        origin=[0.3 * step, 0.5 * step],
        direction=[1, 0],
    )
    stretching = mx.MaxwellCPMLPlan((0, 8), target_reflection=1e-6)
    krylov = (
        mx.FrequencyMovingChargePlan(
            source,
            material,
            stretching=stretching,
            method="krylov",
            tolerance=1e-11,
            restart=300,
            maxiter=6000,
        )
        .prepare()
        .solve(OMEGA)
    )
    direct = (
        mx.FrequencyMovingChargePlan(
            source, material, stretching=stretching, policy=_direct_policy()
        )
        .prepare()
        .solve(OMEGA)
    )
    evidence = krylov.evidence
    assert evidence.method == "krylov"
    assert bool(evidence.converged)
    assert int(evidence.iterations) > 0
    scale = float(jnp.linalg.norm(krylov.source))
    assert float(evidence.residual_norm) < 1e-9 * scale
    np.testing.assert_allclose(
        krylov.electric,
        direct.electric,
        rtol=0,
        atol=1e-8 * float(jnp.max(jnp.abs(direct.electric))),
    )


def test_moving_charge_refusals() -> None:
    bridge = _bridge((6, 6), (0.0, 0.0), (1.0, 1.0), (True, False))
    tmz = mx.MaxwellCochainLayout(bridge, "tmz")
    with pytest.raises(ValueError, match="tez or full_3d"):
        mx.MaxwellMovingChargePlan(
            bridge, tmz, charge=1.0, speed=0.5, origin=[0.1, 0.1], direction=[1, 0]
        )
    layout = mx.MaxwellCochainLayout(bridge, "tez")
    with pytest.raises(ValueError, match="parallel"):
        mx.MaxwellMovingChargePlan(
            bridge, layout, charge=1.0, speed=0.5, origin=[0.1, 0.1], direction=[1, 1]
        ).prepare()
    with pytest.raises(ValueError, match="outside"):
        mx.MaxwellMovingChargePlan(
            bridge, layout, charge=1.0, speed=0.5, origin=[0.1, 1.5], direction=[1, 0]
        ).prepare()
    vacuum = mx.DiagonalMaxwellConstitutivePlan().prepare(bridge.cochain, layout)
    source = mx.MaxwellMovingChargePlan(
        bridge, layout, charge=1.0, speed=0.5, origin=[0.05, 0.55], direction=[1, 0]
    )
    with pytest.raises(TypeError, match="background"):
        mx.FrequencyMovingChargePlan(source, vacuum, formulation="scattered-field")
    with pytest.raises(ValueError, match="Only the scattered-field"):
        mx.FrequencyMovingChargePlan(source, vacuum, background=vacuum)
    lossy = mx.DiagonalMaxwellConstitutivePlan(
        permittivity=np.linspace(1.0, 2.0, layout.electric_count)
    ).prepare(bridge.cochain, layout)
    with pytest.raises(ValueError, match="homogeneous and isotropic"):
        mx.FrequencyMovingChargePlan(
            source,
            vacuum,
            formulation="scattered-field",
            background=lossy,
            policy=_direct_policy(),
        ).prepare().solve(2.0 * np.pi * 0.5)
    # Contrast on the path itself cannot be a scattered-field source.
    touching = mx.DiagonalMaxwellConstitutivePlan(
        permittivity=np.linspace(1.0, 2.0, layout.electric_count)
    ).prepare(bridge.cochain, layout)
    with pytest.raises(ValueError, match="quarter edge"):
        mx.FrequencyMovingChargePlan(
            source,
            touching,
            formulation="scattered-field",
            background=vacuum,
            policy=_direct_policy(),
        ).prepare().solve(2.0 * np.pi * 0.5)
