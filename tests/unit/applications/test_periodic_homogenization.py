"""Periodic conductivity and elasticity cell problems against independent oracles."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.solid_mechanics import (
    isotropic_elastic_stiffness,
    PeriodicCellProblem,
    PeriodicHomogenizationCase,
    PeriodicHomogenizationResult,
)
from phydrax.discretization import CellMesh
from phydrax.discretization.fem import PeriodicSimplexSpace
from tests._support.periodic_simplex_cells import kuhn_cell, octahedral_cell


_UNEVEN = (0.0, 0.125, 0.375, 0.5, 0.625, 1.0)
_LAMINATE_X = (0.0, 0.25, 0.375, 0.5, 0.75, 1.0)
_INTERFACE = 0.375


def _laminate_mesh() -> CellMesh:
    return kuhn_cell(
        (_LAMINATE_X, (0.0, 0.5, 1.0), (0.0, 0.25, 1.0)),
        lambda centres: (centres[:, 0] > _INTERFACE).astype(np.int64),
        names=("first", "second"),
    )


def _schwarz_p(points: np.ndarray) -> np.ndarray:
    return np.sum(np.cos(2.0 * np.pi * points), axis=1)


def _schwarz_p_mesh(n: int, *, porous: bool) -> CellMesh:
    def phase(centroids: np.ndarray) -> np.ndarray:
        value = _schwarz_p(centroids)
        assert np.all(value != 0.0)
        # Phase 1 is unmeshed (pore) when only the network block is named.
        return np.where(value < 0.0, 0, 1)

    names = ("network",) if porous else ("network", "complement")
    return octahedral_cell(n, phase, names)


def _accepted(result: PeriodicHomogenizationResult) -> None:
    assert bool(result.accepted)
    assert all(bool(audit.accepted) for audit in result.hill_mandel)
    for audit in result.hill_mandel:
        assert float(audit.power_defect) <= 1.0e-10 * max(
            1.0, abs(float(audit.macroscopic_power))
        )
    assert float(result.symmetry_defect) <= 1.0e-11
    assert float(result.energy_defect) <= 1.0e-11


def test_unit_cases_are_periodic_homogenization_cases() -> None:
    problem = PeriodicCellProblem(kuhn_cell((_UNEVEN,) * 3), physics="elasticity")
    assert all(isinstance(case, PeriodicHomogenizationCase) for case in problem.cases)
    assert [case.case_id for case in problem.cases] == [
        "strain-xx",
        "strain-yy",
        "strain-zz",
        "strain-yz",
        "strain-xz",
        "strain-xy",
    ]
    shear = np.asarray(problem.cases[3].macroscopic_strain)
    np.testing.assert_array_equal(shear, shear.T)
    assert shear[1, 2] == 0.5


def test_homogeneous_cell_recovers_the_material() -> None:
    mesh = kuhn_cell((_UNEVEN, (0.0, 0.25, 1.0), (0.0, 0.5, 0.75, 1.0)))
    space = PeriodicSimplexSpace(mesh)
    assert space.component_winding_ranks == (3,)
    tensor = jnp.asarray(((3.0, 0.5, 0.0), (0.5, 2.0, 0.25), (0.0, 0.25, 1.5)))
    conduction = PeriodicCellProblem(space, physics="conductivity").solve(tensor)
    _accepted(conduction)
    np.testing.assert_allclose(conduction.effective_tensor, tensor, rtol=0, atol=1e-13)
    assert float(jnp.max(jnp.abs(conduction.fluctuation))) <= 1e-13
    stiffness = isotropic_elastic_stiffness(3.0, 0.3)
    elastic = PeriodicCellProblem(space, physics="elasticity").solve(stiffness)
    _accepted(elastic)
    np.testing.assert_allclose(elastic.effective_tensor, stiffness, rtol=0, atol=1e-12)
    np.testing.assert_allclose(float(elastic.volume_fraction), 1.0, rtol=1e-15)


def _layer_fractions() -> tuple[float, float]:
    return _INTERFACE, 1.0 - _INTERFACE


def test_conductivity_laminate_matches_series_and_parallel_means() -> None:
    first, second = 1.0, 10.0
    phi1, phi2 = _layer_fractions()
    result = PeriodicCellProblem(_laminate_mesh(), physics="conductivity").solve(
        {"first": first, "second": second}
    )
    _accepted(result)
    reuss = 1.0 / (phi1 / first + phi2 / second)
    voigt = phi1 * first + phi2 * second
    np.testing.assert_allclose(
        result.effective_tensor, np.diag((reuss, voigt, voigt)), rtol=1e-12, atol=1e-13
    )


def _laminate_stiffness(c1: np.ndarray, c2: np.ndarray, phi1: float) -> np.ndarray:
    """Exact layered stiffness, normal x: in-plane strains and normal tractions continuous."""
    phi2 = 1.0 - phi1
    normal, plane = [0, 4, 5], [1, 2, 3]
    effective = np.zeros((6, 6))
    for column in range(6):
        macro = np.eye(6)[column]
        # Unknowns: the normal strain components of each phase.
        matrix = np.zeros((6, 6))
        rhs = np.zeros(6)
        matrix[:3, :3] = phi1 * np.eye(3)
        matrix[:3, 3:] = phi2 * np.eye(3)
        rhs[:3] = macro[normal]
        matrix[3:, :3] = c1[np.ix_(normal, normal)]
        matrix[3:, 3:] = -c2[np.ix_(normal, normal)]
        rhs[3:] = (c2[np.ix_(normal, plane)] - c1[np.ix_(normal, plane)]) @ macro[plane]
        unknown = np.linalg.solve(matrix, rhs)
        strains = []
        for phase in range(2):
            strain = macro.copy()
            strain[normal] = unknown[3 * phase : 3 * phase + 3]
            strains.append(strain)
        effective[:, column] = phi1 * c1 @ strains[0] + phi2 * c2 @ strains[1]
    return effective


def test_elastic_laminate_matches_the_layered_oracle_inside_voigt_reuss() -> None:
    c1 = np.asarray(isotropic_elastic_stiffness(1.0, 0.25))
    c2 = np.asarray(isotropic_elastic_stiffness(20.0, 0.35))
    phi1, phi2 = _layer_fractions()
    result = PeriodicCellProblem(_laminate_mesh(), physics="elasticity").solve(
        {"first": c1, "second": c2}
    )
    _accepted(result)
    effective = np.asarray(result.effective_tensor)
    np.testing.assert_allclose(
        effective, _laminate_stiffness(c1, c2, phi1), rtol=1e-11, atol=1e-11
    )
    voigt = phi1 * c1 + phi2 * c2
    reuss = np.linalg.inv(phi1 * np.linalg.inv(c1) + phi2 * np.linalg.inv(c2))
    assert np.min(np.linalg.eigvalsh(voigt - effective)) >= -1e-11
    assert np.min(np.linalg.eigvalsh(effective - reuss)) >= -1e-11
    # Strictly inside: the layered response is neither bound.
    assert np.max(np.linalg.eigvalsh(voigt - effective)) > 1e-3
    assert np.max(np.linalg.eigvalsh(effective - reuss)) > 1e-3


def _hashin_shtrikman_conductivity(
    reference: float, other: float, phi_reference: float
) -> float:
    phi_other = 1.0 - phi_reference
    return reference + phi_other / (
        1.0 / (other - reference) + phi_reference / (3.0 * reference)
    )


def _hashin_shtrikman_bulk(
    reference: tuple[float, float], other: float, phi_reference: float
) -> float:
    bulk, shear = reference
    phi_other = 1.0 - phi_reference
    return bulk + phi_other / (
        1.0 / (other - bulk) + 3.0 * phi_reference / (3.0 * bulk + 4.0 * shear)
    )


def _moduli(young: float, poisson: float) -> tuple[float, float]:
    return young / (3.0 * (1.0 - 2.0 * poisson)), young / (2.0 * (1.0 + poisson))


def test_schwarz_p_composite_is_cubic_and_inside_hashin_shtrikman() -> None:
    mesh = _schwarz_p_mesh(4, porous=False)
    space = PeriodicSimplexSpace(mesh)
    assert space.component_winding_ranks == (3,)
    k_network, k_complement = 10.0, 1.0
    conduction = PeriodicCellProblem(space, physics="conductivity").solve(
        {"network": k_network, "complement": k_complement}
    )
    _accepted(conduction)
    # F(u + 1/2) = -F(u) maps the network onto the complement. All 24 n^3 cells
    # have the same measure, so exactly half of them is a volume fraction of 1/2.
    network_cells = int(np.sum(np.asarray(space.block_mask(("network",)))))
    assert 2 * network_cells == space.cell_count
    assert float(conduction.cubic_symmetry_defect) <= 1e-12
    value = float(jnp.trace(conduction.effective_tensor)) / 3.0
    lower = _hashin_shtrikman_conductivity(k_complement, k_network, 0.5)
    upper = _hashin_shtrikman_conductivity(k_network, k_complement, 0.5)
    assert lower < value < upper

    stiff, soft = (10.0, 0.3), (1.0, 0.2)
    elastic = PeriodicCellProblem(space, physics="elasticity").solve(
        {
            "network": isotropic_elastic_stiffness(*stiff),
            "complement": isotropic_elastic_stiffness(*soft),
        }
    )
    _accepted(elastic)
    assert float(elastic.cubic_symmetry_defect) <= 1e-12
    tensor = np.asarray(elastic.effective_tensor)
    bulk = (tensor[0, 0] + 2.0 * tensor[0, 1]) / 3.0
    stiff_moduli, soft_moduli = _moduli(*stiff), _moduli(*soft)
    lower = _hashin_shtrikman_bulk(soft_moduli, stiff_moduli[0], 0.5)
    upper = _hashin_shtrikman_bulk(stiff_moduli, soft_moduli[0], 0.5)
    assert lower < bulk < upper


def test_porous_schwarz_p_network_is_below_the_upper_hashin_shtrikman_bound() -> None:
    space = PeriodicSimplexSpace(_schwarz_p_mesh(4, porous=True))
    assert space.component_winding_ranks == (3,)
    conduction = PeriodicCellProblem(space, physics="conductivity").solve(2.0)
    _accepted(conduction)
    np.testing.assert_allclose(float(conduction.volume_fraction), 0.5, rtol=1e-15)
    assert float(conduction.cubic_symmetry_defect) <= 1e-12
    value = float(jnp.trace(conduction.effective_tensor)) / 3.0
    # Void phase: lower bound 0, upper 2 phi k / (3 - phi).
    assert 0.0 < value < 2.0 * 0.5 * 2.0 / (3.0 - 0.5)

    young, poisson = 5.0, 0.3
    elastic = PeriodicCellProblem(space, physics="elasticity").solve(
        isotropic_elastic_stiffness(young, poisson)
    )
    _accepted(elastic)
    tensor = np.asarray(elastic.effective_tensor)
    bulk = (tensor[0, 0] + 2.0 * tensor[0, 1]) / 3.0
    upper = _hashin_shtrikman_bulk(_moduli(young, poisson), 0.0, 0.5)
    assert 0.0 < bulk < upper


def test_isolated_inclusion_gauges_conduction_and_refuses_elasticity() -> None:
    def island(centres: np.ndarray) -> np.ndarray:
        inside = np.all(np.abs(centres - 0.5) < 0.2, axis=1)
        return np.where(inside, 0, 1)

    mesh = kuhn_cell(((0.0, 0.25, 0.5, 0.75, 1.0),) * 3, island, names=("island",))
    space = PeriodicSimplexSpace(mesh)
    assert space.component_winding_ranks == (0,)
    with pytest.raises(ValueError, match="wind along at least two"):
        PeriodicCellProblem(space, physics="elasticity")
    result = PeriodicCellProblem(space, physics="conductivity").solve(3.0)
    _accepted(result)
    # A non-percolating island carries no macroscopic flux.
    np.testing.assert_allclose(result.effective_tensor, 0.0, atol=1e-13)
    gauge = np.asarray(space.gauge_dofs)
    assert np.all(np.asarray(result.fluctuation)[:, gauge] == 0.0)
