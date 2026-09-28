from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.finite_volume._capillarity import (
    BalancedCapillaryOperator,
    CurvatureStatus,
    SurfaceTensionPolicy,
)
from phydrax.discretization.finite_volume._cell_polynomial import (
    CellPolynomialReconstructionPlan,
)
from phydrax.discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.finite_volume._unstructured_vof import UnstructuredVOFPlan


def _grid(nx: int = 7, ny: int = 7, oversampling: int = 3, spacing: float = 1.0) -> Any:
    vertices = np.asarray(
        [(spacing * i, spacing * j) for j in range(ny + 1) for i in range(nx + 1)]
    )
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower = j * (nx + 1) + i
            cells.append((lower, lower + 1, lower + nx + 2, lower + nx + 1))
    discretization = UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=np.asarray(cells, dtype=np.int32)
    ).prepare()
    gradient = CellPolynomialReconstructionPlan(1, oversampling=oversampling).prepare(
        discretization
    )
    return discretization, gradient


def _operator_and_plic(kind: str = "circle") -> Any:
    discretization, gradient = _grid(oversampling=8 if kind == "circle" else 3)
    vof = UnstructuredVOFPlan(discretization, gradient)
    centers = np.asarray(discretization.cell_centers)
    if kind == "planar":
        alpha = jnp.asarray(
            np.clip(0.5 + 0.5 * (centers[:, 0] - 3.5), 0.0, 1.0),
            dtype=jnp.float32,
        )
    else:
        radius = 2.1
        distance = np.sqrt((centers[:, 0] - 3.5) ** 2 + (centers[:, 1] - 3.5) ** 2)
        alpha = jnp.asarray(np.clip(0.5 + radius - distance, 0.0, 1.0), dtype=jnp.float32)
    plic = vof.reconstruct(alpha)
    operator = BalancedCapillaryOperator(
        discretization,
        gradient,
        SurfaceTensionPolicy(0.7, 1.0e-6, 0.4, "test-surface"),
        curvature_tolerance=0.5,
    )
    return operator, plic, alpha


def _exact_circle(
    cells: int, radius: float, width: float, phase: str
) -> tuple[Any, Any, Any]:
    """Build a radial interface with exact circle facets and outward normals."""
    discretization, gradient = _grid(cells, cells, spacing=1.0 / cells)
    offset = np.asarray(discretization.cell_centers) - 0.5
    distance = np.hypot(offset[:, 0], offset[:, 1])
    inside = np.clip(0.5 + (radius - distance) / width, 0.0, 1.0)
    alpha = jnp.asarray(inside if phase == "drop" else 1.0 - inside)
    plic = UnstructuredVOFPlan(discretization, gradient).reconstruct(alpha)
    active = np.asarray(plic.interface_active)[:, None]
    radial = offset / distance[:, None]
    outward = radial if phase == "drop" else -radial
    plic = eqx.tree_at(
        lambda value: (value.normals, value.interface_centers),
        plic,
        (
            jnp.asarray(np.where(active, outward, np.asarray(plic.normals))),
            jnp.asarray(
                np.where(
                    active, 0.5 + radius * radial, np.asarray(plic.interface_centers)
                )
            ),
        ),
    )
    return discretization, gradient, (alpha, plic, radial)


def _rest_pressure_force(discretization: Any, pressure: Any) -> np.ndarray:
    """Return the cell force of arithmetic-mean Riemann face pressure at rest."""
    owner = np.asarray(discretization.owner_cells)
    neighbor = np.asarray(discretization.neighbor_cells)
    area = np.asarray(discretization.area_vectors)
    value = np.asarray(pressure)
    interior = neighbor >= 0
    safe = np.maximum(neighbor, 0)
    face = np.where(interior, 0.5 * (value[owner] + value[safe]), value[owner])
    force = np.zeros((discretization.cell_count, area.shape[1]))
    np.add.at(force, owner, -face[:, None] * area)
    np.add.at(force, safe[interior], face[interior, None] * area[interior])
    return force


def test_capillarity_scenario_1() -> None:
    sigma, radius = 0.7, 0.3
    for phase in ("drop", "bubble"):
        discretization, gradient, (alpha, plic, radial) = _exact_circle(
            16, radius, 1.5 / 16, phase
        )
        operator = BalancedCapillaryOperator(
            discretization,
            gradient,
            SurfaceTensionPolicy(sigma, 1.0e-12, 0.4, "laplace"),
        )
        curvature = 1.0 / radius if phase == "drop" else -1.0 / radius
        evidence = operator.curvature(plic, alpha)
        assert bool(jnp.all(evidence.valid_mask == evidence.interface_active))
        np.testing.assert_allclose(
            evidence.curvature[evidence.valid_mask], curvature, rtol=1.0e-10
        )
        jump = operator.laplace_pressure_jump(plic, alpha)
        np.testing.assert_allclose(jump[evidence.valid_mask], sigma * curvature)

        velocity = jax.random.normal(
            jax.random.key(3), (discretization.cell_count, 2)
        )
        block = operator.face_rate_block(plic, alpha, velocity)
        source = np.asarray(block.cell_momentum_rate(discretization.cell_count))
        pressure = 2.0 + sigma * curvature * np.asarray(alpha)
        residual = source + _rest_pressure_force(discretization, pressure)
        scale = np.max(np.abs(source))
        assert scale > 0.0
        rounding = np.max(pressure) * np.max(
            np.asarray(discretization.face_measures)
        )
        assert np.max(np.abs(residual)) <= (
            64.0 * np.finfo(np.float64).eps * rounding
        )
        assert np.sum(source * radial) < 0.0
        np.testing.assert_allclose(block.net_force, 0.0, atol=1.0e-12 * scale)
        np.testing.assert_allclose(
            block.cell_energy_rate(discretization.cell_count),
            np.sum(np.asarray(velocity) * source, axis=-1),
            rtol=1.0e-12,
            atol=1.0e-12 * scale,
        )

    width = 0.2
    densities = []
    for cells in (16, 32):
        discretization, gradient, (alpha, plic, radial) = _exact_circle(
            cells, radius, width, "drop"
        )
        operator = BalancedCapillaryOperator(
            discretization,
            gradient,
            SurfaceTensionPolicy(sigma, 1.0e-12, 0.4, "refinement"),
        )
        block = operator.face_rate_block(
            plic, alpha, jnp.zeros((discretization.cell_count, 2))
        )
        source = np.asarray(block.cell_momentum_rate(discretization.cell_count))
        density = source / np.asarray(discretization.cell_volumes)[:, None]
        offset = np.asarray(discretization.cell_centers) - 0.5
        core = np.abs(np.hypot(offset[:, 0], offset[:, 1]) - radius) <= 0.03
        densities.append(np.mean(np.sum(density[core] * radial[core], axis=-1)))
    np.testing.assert_allclose(densities, -sigma / (radius * width), rtol=2.0e-2)
    operator, plic, alpha = _operator_and_plic("planar")
    velocity = jnp.ones((alpha.size, 2))
    evidence = operator.curvature(plic, alpha)
    assert jnp.all(
        evidence.status[evidence.interface_active] == int(CurvatureStatus.VALID)
    )
    assert jnp.all(evidence.curvature == 0.0)
    block = operator.face_rate_block(plic, alpha, velocity)
    assert jnp.array_equal(block.face_force, jnp.zeros_like(block.face_force))
    result = eqx.filter_jit(
        lambda values: operator.face_rate_block(plic, values, velocity).face_force
    )(alpha)
    assert jnp.array_equal(result, jnp.zeros_like(result))
    derivative = jax.grad(
        lambda values: jnp.sum(
            operator.face_rate_block(plic, values, velocity).face_force ** 2
        )
    )(alpha)
    assert jnp.all(jnp.isfinite(derivative))
    zero = BalancedCapillaryOperator(
        operator.discretization,
        operator.gradient,
        SurfaceTensionPolicy(0.0, 1.0e-6, 0.4, "zero-surface"),
    )
    assert jnp.array_equal(
        zero.face_rate_block(plic, alpha, velocity).face_force,
        jnp.zeros_like(result),
    )

    operator, plic, alpha = _operator_and_plic("circle")
    evidence = operator.curvature(plic, alpha)
    active_curvature = evidence.curvature[evidence.valid_mask]
    assert active_curvature.size > 0
    assert float(jnp.mean(active_curvature)) > 0.0
    jump = operator.laplace_pressure_jump(plic, alpha)
    assert float(jnp.mean(jump[evidence.valid_mask])) > 0.0
    velocity = jnp.ones((alpha.size, 2))
    block = operator.face_rate_block(plic, alpha, velocity)
    source = block.cell_momentum_rate(alpha.size)
    radial = operator.discretization.cell_centers - 3.5
    assert float(jnp.sum(source * radial)) < 0.0
    np.testing.assert_allclose(
        block.cell_energy_rate(alpha.size), jnp.sum(source, axis=-1), atol=1.0e-12
    )
    expected = 0.4 * jnp.sqrt(1.0 * 0.25**3 / 0.7)
    assert jnp.allclose(operator.capillary_step(0.25, jnp.ones(alpha.shape)), expected)
    for pure_alpha in (0.0, 1.0):
        discretization, gradient = _grid()
        alpha = jnp.full((discretization.cell_count,), pure_alpha, dtype=jnp.float32)
        plic = UnstructuredVOFPlan(discretization, gradient).reconstruct(alpha)
        operator = BalancedCapillaryOperator(
            discretization,
            gradient,
            SurfaceTensionPolicy(0.7, 1.0e-6, 0.4, "pure-phase-surface"),
        )
        density = jnp.ones_like(alpha)
        velocity = jnp.ones((alpha.size, 2))

        assert not bool(jnp.any(plic.interface_active))
        evidence = operator.curvature(plic, alpha)
        assert jnp.all(evidence.status == int(CurvatureStatus.MISSING_INTERFACE))

        eager = operator.face_rate_block(plic, alpha, velocity)
        compiled = eqx.filter_jit(
            lambda fraction: operator.face_rate_block(plic, fraction, velocity)
        )(alpha)
        for block in (eager, compiled):
            assert jnp.array_equal(block.face_force, jnp.zeros_like(block.face_force))
            assert jnp.array_equal(
                block.cell_energy_rate(alpha.size), jnp.zeros_like(alpha)
            )

        eager_limit = operator.capillary_step(
            0.25, density, interface_active=plic.interface_active
        )
        compiled_limit = eqx.filter_jit(
            lambda active: operator.capillary_step(
                0.25, density, interface_active=active
            )
        )(plic.interface_active)
        assert bool(jnp.isinf(eager_limit))
        assert bool(jnp.isinf(compiled_limit))


def test_capillarity_scenario_2() -> None:
    operator, plic, alpha = _operator_and_plic("circle")
    velocity = jnp.ones((alpha.size, 2))
    active_index = int(np.flatnonzero(np.asarray(plic.interface_active))[0])
    one_active = jnp.zeros_like(plic.interface_active).at[active_index].set(True)
    uncertain = eqx.tree_at(
        lambda value: value.interface_active,
        plic,
        one_active,
    )
    evidence = operator.curvature(uncertain, alpha)
    assert bool(jnp.any(evidence.uncertain))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        operator.face_rate_block(uncertain, alpha, velocity)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        np.asarray(
            eqx.filter_jit(
                lambda fraction: (
                    operator.face_rate_block(uncertain, fraction, velocity).face_force
                )
            )(alpha)
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        operator.capillary_step(0.25, -jnp.ones(alpha.shape))
    with pytest.raises(ValueError):
        operator.face_rate_block(plic, alpha, jnp.ones((alpha.size, 3)))
    with pytest.raises(ValueError):
        SurfaceTensionPolicy(-1.0, 1.0, 0.5)
    discretization, gradient = _grid()
    alpha = jnp.where(discretization.cell_centers[:, 0] < 3.0, 1.0, 0.0)
    plic = UnstructuredVOFPlan(discretization, gradient).reconstruct(alpha)
    assert not bool(jnp.any(plic.interface_active))
    velocity = jnp.zeros((alpha.size, 2))
    operator = BalancedCapillaryOperator(
        discretization,
        gradient,
        SurfaceTensionPolicy(0.7, 1.0e-6, 0.4, "unsupported-jump"),
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        operator.face_rate_block(plic, alpha, velocity)
    disabled = BalancedCapillaryOperator(
        discretization,
        gradient,
        SurfaceTensionPolicy(0.0, 1.0e-6, 0.4, "disabled-jump"),
    )
    block = disabled.face_rate_block(plic, alpha, velocity)
    assert jnp.array_equal(block.face_force, jnp.zeros_like(block.face_force))
    first = SurfaceTensionPolicy(1.0, 1.0e-6, 0.5, "a")
    second = SurfaceTensionPolicy(1.1, 1.0e-6, 0.5, "a")
    third = SurfaceTensionPolicy(1.0, 1.0e-6, 0.5, "b")
    assert first.policy_id != second.policy_id
    assert first.policy_id != third.policy_id
    from types import SimpleNamespace

    operator, plic, alpha = _operator_and_plic("circle")
    mismatched = SimpleNamespace(
        normals=plic.normals,
        interface_centers=plic.interface_centers,
        interface_measures=plic.interface_measures,
        interface_active=plic.interface_active,
        geometry_id="different-geometry",
        reconstruction_id=plic.reconstruction_id,
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        operator.face_rate_block(mismatched, alpha, jnp.ones((alpha.size, 2)))
