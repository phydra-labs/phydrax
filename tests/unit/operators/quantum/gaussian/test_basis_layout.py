"""CPU numerical contracts; no cuEST installation, loading, or execution required."""

from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax.operators.quantum.gaussian._basis_layout import CuestBasisLayout
from phydrax.operators.quantum.gaussian._shell import cartesian_angular_exponents


def _system(particle_ids: tuple[int, ...] = (9001,)) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        np.ones(len(particle_ids), dtype=np.int32),
        np.zeros((len(particle_ids), 3), dtype=np.float64),
        np.ones(len(particle_ids), dtype=np.float64),
        units.scale,
        particle_ids=np.asarray(particle_ids),
    )
    return phx.atomistic.AtomisticSystemPlan.from_structure(
        structure,
        units,
        molecule_ids=np.zeros(len(particle_ids), dtype=np.int32),
    )


def _basis(shells: tuple[Any, ...], *, particle_ids: tuple[int, ...] = (9001,)) -> Any:
    return phx.operators.quantum.gaussian.GaussianBasisPlan(
        shells, source_id="cuest-layout-numerical-contract"
    ).prepare(_system(particle_ids))


def _shell(center: int, angular: int, *, pure: bool = False) -> Any:
    gaussian = phx.operators.quantum.gaussian
    return gaussian.GaussianShellPlan(
        center,
        angular,
        np.asarray([0.9, 0.23]),
        np.asarray([0.7, -0.15]),
        representation=(
            gaussian.GaussianShellRepresentation.REAL_SPHERICAL
            if pure
            else gaussian.GaussianShellRepresentation.CARTESIAN
        ),
    )


def _moment(power: int, exponent: float) -> float:
    if power % 2:
        return 0.0
    value = np.sqrt(np.pi / exponent)
    for index in range(1, power // 2 + 1):
        value *= (2 * index - 1) / (2 * exponent)
    return float(value)


def _vendor_values(layout: CuestBasisLayout, points: np.ndarray) -> np.ndarray:
    """Independent documented CCA monomials/Racah harmonics through d.

    Source: https://docs.nvidia.com/cuda/cuest/gaussian_basis.html, Tables 9/10.
    No native transform or normalization implementation is used here.
    """
    x, y, z = points.T
    radius2 = np.sum(points * points, axis=1)
    blocks = []
    for _, angular, count, offset in np.asarray(layout.shell_metadata):
        exponents = np.asarray(layout.exponents)[offset : offset + count]
        coefficients = np.asarray(layout.coefficients)[offset : offset + count]
        radial = np.exp(-radius2[:, None] * exponents[None, :]) @ coefficients
        if layout.pure:
            polynomials = (
                np.ones((points.shape[0], 1)),
                np.stack((z, x, y), axis=1),
                np.stack(
                    (
                        (3 * z * z - radius2) / 2,
                        np.sqrt(3) * x * z,
                        np.sqrt(3) * y * z,
                        np.sqrt(3) * (x * x - y * y) / 2,
                        np.sqrt(3) * x * y,
                    ),
                    axis=1,
                ),
            )[angular]
        else:
            polynomials = np.stack(
                tuple(
                    x**a * y**b * z**c for a, b, c in cartesian_angular_exponents(angular)
                ),
                axis=1,
            )
        blocks.append(radial[:, None] * polynomials)
    return np.concatenate(blocks, axis=1)


def test_packing_splits_contractions_masks_and_reorders_stable_centers() -> None:
    gaussian = phx.operators.quantum.gaussian
    basis = _basis(
        (
            gaussian.GaussianShellPlan(
                42,
                0,
                np.asarray([1.2, np.nan, 0.3]),
                np.asarray([[0.7, np.nan, 0.2], [-0.2, np.nan, 0.8]]),
                primitive_mask=np.asarray([True, False, True]),
            ),
            gaussian.GaussianShellPlan(9001, 0, np.asarray([0.4]), np.asarray([1.0])),
            gaussian.GaussianShellPlan(42, 1, np.asarray([0.6]), np.asarray([1.0])),
        ),
        particle_ids=(9001, 42),
    )
    layout = CuestBasisLayout(basis)
    assert layout.pure == 0
    assert layout.shell_count == 4
    assert layout.basis_function_count == 6
    np.testing.assert_array_equal(
        layout.shell_metadata, [[0, 0, 1, 0], [1, 0, 2, 1], [1, 0, 2, 3], [1, 1, 1, 5]]
    )
    np.testing.assert_array_equal(
        layout.shell_native_indices, [[1, 0], [0, 0], [0, 1], [2, 0]]
    )
    np.testing.assert_array_equal(layout.shell_native_offsets, [2, 0, 1, 3])
    np.testing.assert_array_equal(layout.shell_vendor_offsets, [0, 1, 2, 3])
    np.testing.assert_array_equal(layout.active_primitive_indices, [0, 0, 2, 0, 2, 0])
    np.testing.assert_array_equal(layout.center_indices, [0, 1])
    np.testing.assert_array_equal(layout.center_particle_ids, [9001, 42, 42, 42])
    np.testing.assert_array_equal(layout.vendor_to_native, [2, 0, 1, 3, 4, 5])
    np.testing.assert_array_equal(layout.exponents, [0.4, 1.2, 0.3, 1.2, 0.3, 0.6])
    assert layout.shell_metadata.dtype == np.dtype("int32")
    assert layout.exponents.dtype == layout.coefficients.dtype == np.dtype("float64")
    # Verbatim finished coefficients, not raw text-file coefficients and not
    # doubly normalized. Active primitive indices make the split reversible.
    for packed, (shell_index, contraction) in enumerate(
        np.asarray(layout.shell_native_indices)
    ):
        shell = basis.plan.shells[shell_index]
        row = np.flatnonzero(
            (np.asarray(basis.shell_indices) == shell_index)
            & (np.asarray(basis.contraction_indices) == contraction)
        )[0]
        _, _, count, offset = np.asarray(layout.shell_metadata)[packed]
        active = np.asarray(layout.active_primitive_indices)[offset : offset + count]
        np.testing.assert_array_equal(
            active, np.flatnonzero(np.asarray(shell.primitive_mask))
        )
        np.testing.assert_array_equal(
            np.asarray(layout.coefficients)[offset : offset + count],
            np.asarray(basis.normalized_coefficients)[row, active],
        )


@pytest.mark.parametrize("pure", [False, True])
def test_documented_cartesian_and_racah_spherical_values(pure: bool) -> None:
    gaussian = phx.operators.quantum.gaussian
    basis = _basis(tuple(_shell(9001, angular, pure=pure) for angular in range(3)))
    layout = CuestBasisLayout(basis, auxiliary=pure)
    points = np.asarray([[0.2, -0.3, 0.4], [-0.6, 0.1, 0.7], [0.3, 0.5, -0.2]])
    native = np.asarray(gaussian.ao_values(basis, np.zeros((1, 3)), points))
    vendor = _vendor_values(layout, points)
    expected_native = vendor[:, np.asarray(layout.native_to_vendor)] * np.asarray(
        layout.component_scales
    )
    np.testing.assert_allclose(native, expected_native, rtol=3.0e-12, atol=3.0e-13)
    if pure:
        np.testing.assert_array_equal(layout.component_scales, np.ones(9))
    else:
        # d order xx,xy,xz,yy,yz,zz, with only vendor superdiagonals normalized.
        np.testing.assert_allclose(
            np.asarray(layout.component_scales)[4:],
            [1, np.sqrt(3), np.sqrt(3), 1, np.sqrt(3), 1],
            rtol=1.0e-15,
        )


def test_finished_radial_coefficients_normalize_leading_components() -> None:
    layout = CuestBasisLayout(
        _basis(tuple(_shell(9001, angular) for angular in range(5)))
    )
    for _, angular, count, offset in np.asarray(layout.shell_metadata):
        exponents = np.asarray(layout.exponents)[offset : offset + count]
        coefficients = np.asarray(layout.coefficients)[offset : offset + count]
        leading_overlap = 0.0
        mixed_overlap = 0.0
        for i in range(count):
            for j in range(count):
                power = float(exponents[i] + exponents[j])
                weight = coefficients[i] * coefficients[j]
                leading_overlap += (
                    weight * _moment(2 * int(angular), power) * _moment(0, power) ** 2
                )
                if angular == 2:
                    mixed_overlap += weight * _moment(2, power) ** 2 * _moment(0, power)
        np.testing.assert_allclose(leading_overlap, 1.0, rtol=3.0e-13, atol=3.0e-13)
        if angular == 2:
            np.testing.assert_allclose(mixed_overlap, 1 / 3, rtol=3.0e-13)


def test_matrix_density_factor_and_pulay_conversions_preserve_contractions() -> None:
    # Deliberately interleaved centers and nonascending L: vendor groups centers,
    # but does not silently reorder native AOs or invent shells.
    basis = _basis(
        (_shell(42, 0), _shell(9001, 2), _shell(9001, 0), _shell(9001, 1)),
        particle_ids=(9001, 42),
    )
    layout = CuestBasisLayout(basis)
    count = layout.basis_function_count
    rng = np.random.default_rng(51)
    raw = rng.normal(size=(count, count))
    matrix = raw + raw.T
    factors = rng.normal(size=(3, count))
    density = factors.T @ factors
    weighted = density * np.linspace(0.2, 1.1, count)[:, None]
    weighted = (weighted + weighted.T) / 2
    vendor_matrix = np.asarray(layout.matrix_to_vendor(matrix))
    vendor_density = np.asarray(layout.density_to_vendor(density))
    vendor_factors = np.asarray(layout.factors_to_vendor(factors))
    np.testing.assert_allclose(
        layout.matrix_from_vendor(vendor_matrix), matrix, atol=2.0e-14
    )
    np.testing.assert_allclose(
        layout.density_from_vendor(vendor_density), density, atol=2.0e-14
    )
    np.testing.assert_allclose(
        layout.factors_from_vendor(vendor_factors), factors, atol=2.0e-14
    )
    np.testing.assert_allclose(
        vendor_factors.T @ vendor_factors, vendor_density, atol=2.0e-14
    )
    np.testing.assert_allclose(
        np.sum(density * matrix), np.sum(vendor_density * vendor_matrix), atol=2.0e-13
    )
    vendor_weighted = np.asarray(layout.energy_weighted_density_to_vendor(weighted))
    np.testing.assert_allclose(
        np.sum(weighted * matrix), np.sum(vendor_weighted * vendor_matrix), atol=2.0e-13
    )
    batched = np.stack([vendor_matrix, 2 * vendor_matrix])
    np.testing.assert_allclose(
        layout.matrix_from_vendor(batched), np.stack([matrix, 2 * matrix]), atol=2.0e-14
    )
    assert layout.factors_to_vendor(np.zeros((0, count))).shape == (0, count)


def test_layout_arrays_are_dynamic_and_identity_is_deterministic() -> None:
    basis = _basis(tuple(_shell(9001, angular) for angular in range(3)))
    first = CuestBasisLayout(basis)
    second = CuestBasisLayout(basis)
    assert first.layout_id == second.layout_id
    leaves = jax.tree_util.tree_leaves(eqx.filter(first, eqx.is_array))
    assert len(leaves) == 12
    assert all(leaf.ndim <= 2 for leaf in leaves)
    assert all(leaf.shape != (10, 10) for leaf in leaves)
    changed = CuestBasisLayout(
        _basis(tuple(_shell(9001, angular, pure=True) for angular in range(3)))
    )
    assert first.layout_id != changed.layout_id
    gaussian = phx.operators.quantum.gaussian
    changed_exponents = CuestBasisLayout(
        _basis(
            (
                gaussian.GaussianShellPlan(
                    9001, 0, np.asarray([1.1, 0.23]), np.asarray([0.7, -0.15])
                ),
                _shell(9001, 1),
                _shell(9001, 2),
            )
        )
    )
    np.testing.assert_array_equal(first.shell_metadata, changed_exponents.shell_metadata)
    assert first.layout_id != changed_exponents.layout_id
    assert not np.array_equal(first.exponents, changed_exponents.exponents)


def test_centers_remain_actual_system_indices_without_filler_shells() -> None:
    layout = CuestBasisLayout(_basis((_shell(42, 0),), particle_ids=(9001, 42)))
    np.testing.assert_array_equal(layout.center_indices, [1])
    np.testing.assert_array_equal(layout.shell_metadata[:, 0], [1])
    np.testing.assert_array_equal(layout.center_particle_ids, [42])
    assert layout.shell_count == 1


@pytest.mark.parametrize("angular", [1, 2, 4])
def test_angular_holes_are_rejected_per_center(angular: int) -> None:
    basis = _basis((_shell(9001, 0), _shell(42, angular)), particle_ids=(9001, 42))
    with pytest.raises(ValueError, match="contiguous angular momenta.*system center 1"):
        CuestBasisLayout(basis)


def test_mixed_representation_and_cartesian_auxiliary_are_rejected() -> None:
    mixed = _basis((_shell(9001, 0), _shell(9001, 1, pure=True)))
    with pytest.raises(ValueError, match="mixed shell representations"):
        CuestBasisLayout(mixed)
    with pytest.raises(ValueError, match="auxiliary basis must be spherical"):
        CuestBasisLayout(_basis((_shell(9001, 0),)), auxiliary=True)


def test_matrix_and_factor_shape_contracts() -> None:
    layout = CuestBasisLayout(_basis((_shell(9001, 0), _shell(9001, 1))))
    for method in (
        layout.matrix_from_vendor,
        layout.matrix_to_vendor,
        layout.density_to_vendor,
        layout.density_from_vendor,
        layout.energy_weighted_density_to_vendor,
    ):
        with pytest.raises(ValueError, match="trailing shape"):
            method(np.zeros((4, 3)))
    for method in (layout.factors_to_vendor, layout.factors_from_vendor):
        with pytest.raises(ValueError, match="shape \\(rank, nao\\)"):
            method(np.zeros((4,)))
