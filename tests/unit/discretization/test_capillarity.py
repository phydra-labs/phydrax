from __future__ import annotations

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


def _grid(nx: int = 7, ny: int = 7, oversampling: int = 3):
    vertices = np.asarray(
        [(float(i), float(j)) for j in range(ny + 1) for i in range(nx + 1)]
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


def _operator_and_plic(kind: str = "circle"):
    discretization, gradient = _grid(oversampling=8 if kind == "circle" else 3)
    vof = UnstructuredVOFPlan(discretization, gradient)
    centres = np.asarray(discretization.cell_centers)
    if kind == "planar":
        alpha = jnp.asarray(
            np.clip(0.5 + 0.5 * (centres[:, 0] - 3.5), 0.0, 1.0),
            dtype=jnp.float32,
        )
    else:
        radius = 2.1
        distance = np.sqrt((centres[:, 0] - 3.5) ** 2 + (centres[:, 1] - 3.5) ** 2)
        alpha = jnp.asarray(np.clip(0.5 + radius - distance, 0.0, 1.0), dtype=jnp.float32)
    plic = vof.reconstruct(alpha)
    operator = BalancedCapillaryOperator(
        discretization,
        gradient,
        SurfaceTensionPolicy(0.7, 1.0e-6, 0.4, "test-surface"),
    )
    return operator, plic, alpha


def test_planar_zero_force_and_jittable_gradient():
    operator, plic, alpha = _operator_and_plic("planar")
    evidence = operator.curvature(plic, alpha)
    assert jnp.all(
        evidence.status[evidence.interface_active] == int(CurvatureStatus.VALID)
    )
    assert jnp.all(evidence.curvature == 0.0)
    block = operator.face_rate_block(plic, jnp.ones(alpha.shape), alpha)
    assert jnp.array_equal(block.momentum_rate, jnp.zeros_like(block.momentum_rate))
    result = jax.jit(lambda values: operator.momentum_force_rate(plic, values, alpha))(
        jnp.ones(alpha.shape)
    )
    assert jnp.array_equal(result, jnp.zeros_like(result))
    derivative = jax.grad(
        lambda values: jnp.sum(
            operator.momentum_force_rate(plic, jnp.ones(alpha.shape), values) ** 2
        )
    )(alpha)
    assert jnp.all(jnp.isfinite(derivative))
    zero = BalancedCapillaryOperator(
        operator.discretization,
        operator.gradient,
        SurfaceTensionPolicy(0.0, 1.0e-6, 0.4, "zero-surface"),
    )
    assert jnp.array_equal(
        zero.momentum_force_rate(plic, jnp.ones(alpha.shape), alpha),
        jnp.zeros_like(result),
    )


def test_circle_jump_budget_and_capillary_step():
    operator, plic, alpha = _operator_and_plic("circle")
    evidence = operator.curvature(plic, alpha)
    active_curvature = evidence.curvature[evidence.valid_mask]
    assert active_curvature.size > 0
    assert float(jnp.mean(active_curvature)) > 0.0
    jump = operator.laplace_pressure_jump(plic, alpha)
    assert float(jnp.mean(jump[evidence.valid_mask])) > 0.0
    block = operator.face_rate_block(
        plic, jnp.ones(alpha.shape), alpha, jnp.ones((alpha.size, 2))
    )
    assert jnp.allclose(block.momentum_budget(alpha.size), 0.0)
    assert jnp.allclose(block.energy_budget(alpha.size), 0.0)
    expected = 0.4 * jnp.sqrt(1.0 * 0.25**3 / 0.7)
    assert jnp.allclose(operator.capillary_step(0.25, jnp.ones(alpha.shape)), expected)


@pytest.mark.parametrize("pure_alpha", (0.0, 1.0))
def test_pure_phase_is_exact_zero_eager_and_filter_jit(pure_alpha):
    discretization, gradient = _grid()
    alpha = jnp.full((discretization.cell_count,), pure_alpha, dtype=jnp.float32)
    plic = UnstructuredVOFPlan(discretization, gradient).reconstruct(alpha)
    operator = BalancedCapillaryOperator(
        discretization,
        gradient,
        SurfaceTensionPolicy(0.7, 1.0e-6, 0.4, "pure-phase-surface"),
    )
    density = jnp.ones_like(alpha)

    assert not bool(jnp.any(plic.interface_active))
    evidence = operator.curvature(plic, alpha)
    assert jnp.all(evidence.status == int(CurvatureStatus.MISSING_INTERFACE))

    eager = operator.face_rate_block(plic, density, alpha, jnp.ones((alpha.size, 2)))
    compiled = eqx.filter_jit(
        lambda rho, fraction: operator.face_rate_block(
            plic, rho, fraction, jnp.ones((fraction.size, 2))
        )
    )(density, alpha)
    for block in (eager, compiled):
        assert jnp.array_equal(block.momentum_rate, jnp.zeros_like(block.momentum_rate))
        assert jnp.array_equal(
            block.energy_work_rate, jnp.zeros_like(block.energy_work_rate)
        )

    eager_limit = operator.capillary_step(
        0.25, density, interface_active=plic.interface_active
    )
    compiled_limit = eqx.filter_jit(
        lambda active: operator.capillary_step(0.25, density, interface_active=active)
    )(plic.interface_active)
    assert bool(jnp.isinf(eager_limit))
    assert bool(jnp.isinf(compiled_limit))


def test_active_uncertain_and_invalid_inputs_fail_closed():
    operator, plic, alpha = _operator_and_plic("circle")
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
        operator.face_rate_block(uncertain, jnp.ones(alpha.shape), alpha)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        np.asarray(
            eqx.filter_jit(
                lambda density: operator.momentum_force_rate(uncertain, density, alpha)
            )(jnp.ones(alpha.shape))
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        operator.face_rate_block(plic, -jnp.ones(alpha.shape), alpha)
    with pytest.raises(ValueError):
        SurfaceTensionPolicy(-1.0, 1.0, 0.5)


def test_policy_identity_changes_with_policy_fields():
    first = SurfaceTensionPolicy(1.0, 1.0e-6, 0.5, "a")
    second = SurfaceTensionPolicy(1.1, 1.0e-6, 0.5, "a")
    third = SurfaceTensionPolicy(1.0, 1.0e-6, 0.5, "b")
    assert first.policy_id != second.policy_id
    assert first.policy_id != third.policy_id


def test_geometry_identity_mismatch_is_rejected():
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
        operator.face_rate_block(mismatched, jnp.ones(alpha.shape), alpha)
