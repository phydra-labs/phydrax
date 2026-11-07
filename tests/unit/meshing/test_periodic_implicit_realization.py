"""Fixed-topology lattice design realization and its homogenization sensitivities.

The trigonometric table ``F(u) = cos(2 pi u_x)`` has the layered level sets
``x = a(c), 1 - a(c)`` with ``a = arccos(c) / (2 pi)``, so the volume fraction,
the coarea identity ``d phi / dc = int_{F = c} 1 / |grad F| dA`` and the layered
(series/parallel) effective conductivity are known in closed form.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

from phydrax.applications.solid_mechanics import PeriodicCellProblem
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.geometry.analytic._trigonometric_level_set import (
    TrigonometricLevelSet,
    TrigonometricSheetBand,
    TrigonometricTable,
    TrigonometricTerm,
)
from phydrax.geometry.design._schema import DesignState
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    discover_periodic_trigonometric_surface,
)
from phydrax.meshing._motion import MeshMotionDecision
from phydrax.meshing._periodic_implicit_realization import (
    PeriodicImplicitRealizationEventError,
    PeriodicImplicitRealizationPlan,
    PeriodicImplicitRealizationStatus,
    prepare_periodic_implicit_realization,
)
from phydrax.meshing._periodic_implicit_volume import (
    mesh_periodic_trigonometric_region,
    PeriodicImplicitVolume,
)
from tests._support.periodic_simplex_cells import kuhn_cell


_TABLE = TrigonometricTable((TrigonometricTerm(1, (("cos", 1), ("cos", 0), ("cos", 0))),))
_K_NETWORK, _K_COMPLEMENT = 4.0, 1.0


def _layered(
    breakpoints: tuple[float, ...], inside: Callable[[np.ndarray], np.ndarray]
) -> CellMesh:
    return kuhn_cell(
        (breakpoints, (0.0, 0.5, 1.0), (0.0, 0.5, 1.0)),
        lambda centres: np.where(inside(centres[:, 0]), 0, 1),
        names=("region", "other"),
    )


def _network_plan() -> tuple[TrigonometricLevelSet, PeriodicImplicitRealizationPlan]:
    source = TrigonometricLevelSet(
        _TABLE, PeriodicCell(np.eye(3, dtype=np.float64)), level=0.0, region="network"
    )
    mesh = _layered(
        (0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0),
        lambda x: (x > 0.25) & (x < 0.75),
    )
    return source, prepare_periodic_implicit_realization(
        mesh, source, region_blocks=("region",)
    )


def _level_state(plan: PeriodicImplicitRealizationPlan, level: ArrayLike) -> DesignState:
    return plan.base_state.updated({plan.parameter_id("level"): level})


def _layered_conductivity(phi: float) -> tuple[float, float]:
    series = 1.0 / (phi / _K_NETWORK + (1.0 - phi) / _K_COMPLEMENT)
    parallel = phi * _K_NETWORK + (1.0 - phi) * _K_COMPLEMENT
    return series, parallel


def test_network_realization_matches_the_coarea_and_layered_oracles() -> None:
    _, plan = _network_plan()
    problem = PeriodicCellProblem(plan.space, physics="conductivity")
    materials = {"region": _K_NETWORK, "other": _K_COMPLEMENT}
    level = 0.2

    def fraction(value: ArrayLike) -> Array:
        return plan.realize(_level_state(plan, value)).volume_fraction

    def conductivity(value: ArrayLike) -> Array:
        realization = plan.realize(_level_state(plan, value))
        return problem.solve_realization(materials, realization).effective_tensor

    realization = plan.realize(_level_state(plan, level))
    assert bool(realization.accepted)
    assert not bool(realization.refresh_required)
    assert realization.require_sensitivities() is realization
    a = math.acos(level) / (2.0 * math.pi)
    phi = 1.0 - 2.0 * a
    np.testing.assert_allclose(float(realization.volume_fraction), phi, rtol=1e-12)
    # Coarea: two unit planes with |grad F| = 2 pi |sin(2 pi a)|.
    coarea = 2.0 / (2.0 * math.pi * math.sqrt(1.0 - level**2))
    np.testing.assert_allclose(float(jax.grad(fraction)(level)), coarea, rtol=1e-10)

    series, parallel = _layered_conductivity(phi)
    tensor = conductivity(level)
    np.testing.assert_allclose(
        tensor, np.diag((series, parallel, parallel)), rtol=1e-11, atol=1e-12
    )
    derivative = np.asarray(jax.jacfwd(conductivity)(level))
    reverse = np.asarray(jax.grad(lambda value: conductivity(value)[0, 0])(level))
    step = 1.0e-5
    difference = (
        np.asarray(conductivity(level + step)) - np.asarray(conductivity(level - step))
    ) / (2.0 * step)
    np.testing.assert_allclose(derivative, difference, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(reverse, derivative[0, 0], rtol=1e-10)
    d_series = series**2 * (1.0 / _K_NETWORK - 1.0 / _K_COMPLEMENT) * -coarea
    d_parallel = (_K_NETWORK - _K_COMPLEMENT) * coarea
    np.testing.assert_allclose(
        derivative, np.diag((d_series, d_parallel, d_parallel)), rtol=1e-9, atol=1e-11
    )

    assessment = plan.assess(realization)
    assert assessment.decision is MeshMotionDecision.ACCEPT_MOTION


def test_cell_scale_is_an_exact_linear_lattice_map() -> None:
    _, plan = _network_plan()
    problem = PeriodicCellProblem(plan.space, physics="conductivity")
    scale_id = plan.parameter_id("cell_scale")
    materials = {"region": _K_NETWORK, "other": _K_COMPLEMENT}

    def outputs(scale: ArrayLike) -> tuple[Array, Array]:
        realization = plan.realize(plan.base_state.updated({scale_id: scale}))
        result = problem.solve_realization(materials, realization)
        return realization.volume_fraction, result.effective_tensor

    scale = jnp.asarray((1.5, 0.75, 1.25))
    realization = plan.realize(plan.base_state.updated({scale_id: scale}))
    np.testing.assert_allclose(
        realization.lattice_vectors, np.diag(np.asarray(scale)), rtol=0, atol=0
    )
    # Stretching a layered cell along any axis keeps its layers and fractions.
    fraction_jacobian, tensor_jacobian = jax.jacfwd(outputs)(scale)
    np.testing.assert_allclose(fraction_jacobian, 0.0, atol=1e-13)
    np.testing.assert_allclose(tensor_jacobian, 0.0, atol=1e-11)
    series, parallel = _layered_conductivity(0.5)
    np.testing.assert_allclose(
        outputs(scale)[1], np.diag((series, parallel, parallel)), rtol=1e-11, atol=1e-15
    )


def test_sheet_half_width_sensitivity() -> None:
    source = TrigonometricLevelSet(
        _TABLE,
        PeriodicCell(np.eye(3, dtype=np.float64)),
        level=0.0,
        region="sheet",
        sheet=TrigonometricSheetBand(0.5),
    )
    sixth = 1.0 / 6.0
    mesh = _layered(
        (0.0, 1 / 12, sixth, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 5 / 6, 11 / 12, 1.0),
        lambda x: np.abs(np.cos(2.0 * np.pi * x)) < 0.5,
    )
    plan = prepare_periodic_implicit_realization(mesh, source, region_blocks=("region",))
    width_id = plan.parameter_id("half_width")

    def fraction(width: ArrayLike) -> Array:
        return plan.realize(plan.base_state.updated({width_id: width})).volume_fraction

    width = 0.45
    realization = plan.realize(plan.base_state.updated({width_id: width}))
    assert bool(realization.accepted)
    expected = 2.0 * (math.acos(-width) - math.acos(width)) / (2.0 * math.pi)
    np.testing.assert_allclose(float(realization.volume_fraction), expected, rtol=1e-12)
    # Coarea over the four planes |F| = tau.
    coarea = 4.0 / (2.0 * math.pi * math.sqrt(1.0 - width**2))
    np.testing.assert_allclose(float(jax.grad(fraction)(width)), coarea, rtol=1e-10)


def test_leaving_the_fixed_topology_window_is_an_event() -> None:
    _, plan = _network_plan()

    def fraction(value: ArrayLike) -> Array:
        return plan.realize(_level_state(plan, value)).volume_fraction

    # a(0.8) < 1/8: the surface sweeps through the x = 1/8 vertex layer.
    swept = plan.realize(_level_state(plan, 0.8))
    status = int(swept.status)
    assert status & PeriodicImplicitRealizationStatus.SIGN_PATTERN_CHANGED
    assert bool(swept.refresh_required)
    with pytest.raises(
        PeriodicImplicitRealizationEventError, match="SIGN_PATTERN_CHANGED"
    ):
        swept.require_sensitivities()
    assert bool(jnp.isnan(jax.grad(fraction)(0.8)))
    # Past the certified critical-value enclosure of the maximum F = 1.
    past = plan.realize(_level_state(plan, 0.9995))
    assert int(past.status) & PeriodicImplicitRealizationStatus.CRITICAL_GAP_EXIT
    assert float(jnp.min(past.gap_margins)) < 0.0


def test_preparation_refuses_inconsistent_inputs() -> None:
    source, plan = _network_plan()
    shifted = TrigonometricLevelSet(
        _TABLE, PeriodicCell(np.eye(3, dtype=np.float64)), level=0.3, region="network"
    )
    with pytest.raises(ValueError, match="root_tolerance|zero set"):
        prepare_periodic_implicit_realization(
            plan.space.mesh, shifted, region_blocks=("region",)
        )
    other = TrigonometricLevelSet(
        _TABLE, PeriodicCell(2.0 * np.eye(3, dtype=np.float64)), level=0.0
    )
    with pytest.raises(ValueError, match="lattice"):
        prepare_periodic_implicit_realization(plan.space.mesh, other)
    del source


@cache
def _schwarz_p_volume() -> tuple[TrigonometricLevelSet, PeriodicImplicitVolume]:

    source = TrigonometricLevelSet(
        TrigonometricTable.family("schwarz_p"), PeriodicCell(np.eye(3, dtype=np.float64))
    )
    surface = discover_periodic_trigonometric_surface(
        source,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.2, root_tolerance=_DISCOVERY_ROOT_TOLERANCE
        ),
        source_id="schwarz_p-network",
    )
    return source, mesh_periodic_trigonometric_region(surface)


_DISCOVERY_ROOT_TOLERANCE = 1.0e-8


def _schwarz_p_fraction(level: float, samples: int = 4000) -> float:
    """Exact line integral: for fixed (y, z), {cos 2 pi x <= s} has length 1 - acos(s)/pi."""
    t = (np.arange(samples) + 0.5) / samples
    s = level - np.cos(2.0 * np.pi * t)[:, None] - np.cos(2.0 * np.pi * t)[None, :]
    return float(np.mean(1.0 - np.arccos(np.clip(s, -1.0, 1.0)) / np.pi))


def test_schwarz_p_quotient_mesh_homogenizes_and_differentiates() -> None:
    from phydrax.applications.solid_mechanics import isotropic_elastic_stiffness

    source, volume = _schwarz_p_volume()
    assert volume.result.audit.passed
    mesh = volume.mesh
    conduction = PeriodicCellProblem(mesh, physics="conductivity")
    assert conduction.space.component_winding_ranks == (3,)
    result = conduction.solve(1.0)
    assert bool(result.accepted)
    assert float(result.symmetry_defect) <= 1e-12
    assert float(result.energy_defect) <= 1e-12
    phi = float(result.volume_fraction)
    assert phi == pytest.approx(volume.measure, rel=1e-14)
    # Network in void: Hashin-Shtrikman brackets the isotropic part by 0 and
    # 2 phi k / (3 - phi); the quotient mesh is not exactly symmetric, so the
    # cubic defect is a measured discretization consequence.
    value = float(jnp.trace(result.effective_tensor)) / 3.0
    assert 0.0 < value < 2.0 * phi / (3.0 - phi)
    assert float(result.cubic_symmetry_defect) <= 1e-2
    elastic = PeriodicCellProblem(mesh, physics="elasticity").solve(
        isotropic_elastic_stiffness(1.0, 0.3)
    )
    assert bool(elastic.accepted)
    assert float(elastic.cubic_symmetry_defect) <= 1e-2
    tensor = np.asarray(elastic.effective_tensor)
    bulk_modulus, shear_modulus = 1.0 / (3.0 * 0.4), 1.0 / 2.6
    upper = bulk_modulus - (1.0 - phi) / (
        1.0 / bulk_modulus - 3.0 * phi / (3.0 * bulk_modulus + 4.0 * shear_modulus)
    )
    assert 0.0 < (tensor[0, 0] + 2.0 * tensor[0, 1]) / 3.0 < upper

    # The fill's recorded contour root-error bound sets the projection tolerance;
    # the bare mesh's vertices are not roots within the default 1e-8 field tolerance.
    with pytest.raises(ValueError, match="root_tolerance"):
        prepare_periodic_implicit_realization(mesh, source)
    plan = prepare_periodic_implicit_realization(volume, source)
    assert plan.projection.policy.root_tolerance > 1.0e-8
    base = plan.realize(plan.base_state).require_sensitivities()
    assert plan.assess(base).decision is MeshMotionDecision.ACCEPT_MOTION

    def fraction(level: ArrayLike) -> Array:
        return plan.realize(_level_state(plan, level)).volume_fraction

    def conductivity(level: ArrayLike) -> Array:
        realization = plan.realize(_level_state(plan, level))
        return conduction.solve_realization(1.0, realization).effective_tensor

    step = 1.0e-5
    derivative = float(jax.grad(fraction)(0.0))
    difference = (float(fraction(step)) - float(fraction(-step))) / (2.0 * step)
    assert derivative == pytest.approx(difference, rel=1e-7)
    delta = 1.0e-3
    exact = (_schwarz_p_fraction(delta) - _schwarz_p_fraction(-delta)) / (2.0 * delta)
    # Coarea oracle within the O(h^2) contour discretization at d = 0.2.
    assert derivative == pytest.approx(exact, rel=0.05)
    jacobian = np.asarray(jax.jacfwd(conductivity)(0.0))
    finite = (np.asarray(conductivity(step)) - np.asarray(conductivity(-step))) / (
        2.0 * step
    )
    np.testing.assert_allclose(jacobian, finite, rtol=1e-6, atol=1e-9)
    assert np.trace(jacobian) > 0.0


def test_a_fill_of_another_source_is_refused() -> None:
    source, volume = _schwarz_p_volume()
    levels = source.critical_levels
    for other in (
        TrigonometricLevelSet(
            source.table,
            source.cell,
            critical_levels=levels,
            feature_id="another-lattice",
        ),
        TrigonometricLevelSet(
            source.table,
            source.cell,
            region="complement",
            critical_levels=levels,
            feature_id=source.feature_id,
        ),
    ):
        with pytest.raises(ValueError, match="different trigonometric source"):
            prepare_periodic_implicit_realization(volume, other)
