import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.operators.quantum.gaussian import (
    FactorizedERITensor,
    PreparedDirectJK,
    PreparedGaussianBasis,
)


def _hydrogen_basis(
    source_id: str = "integral-provider-hydrogen",
    *,
    spherical: bool = False,
) -> tuple[PreparedGaussianBasis, Array]:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]], dtype=np.float64),
        np.ones((2,), dtype=np.float64),
        units.scale,
        particle_ids=np.asarray([11, 17], dtype=np.int32),
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    gaussian = phx.operators.quantum.gaussian
    if spherical:
        plan = gaussian.GaussianBasisPlan(
            tuple(
                gaussian.GaussianShellPlan(
                    center,
                    0,
                    np.ones((1,), dtype=np.float64),
                    np.ones((1,), dtype=np.float64),
                    representation=gaussian.GaussianShellRepresentation.REAL_SPHERICAL,
                    role="auxiliary",
                )
                for center in (11, 17)
            ),
            source_id=source_id,
            role="auxiliary",
        )
    else:
        plan = gaussian.GaussianBasisPlan.from_contracted_s(
            [11, 17],
            np.ones((2, 1), dtype=np.float64),
            np.ones((2, 1), dtype=np.float64),
            source_id=source_id,
        )
    basis = plan.prepare(system)
    positions = structure.positions * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    return basis, positions


def _source(
    basis: PreparedGaussianBasis, positions: Array, representation: str
) -> FactorizedERITensor | PreparedDirectJK | None:
    gaussian = phx.operators.quantum.gaussian
    if representation == "exact":
        return None
    if representation == "factorized":
        return gaussian.PivotedCholeskyERIPlan(1.0e-12).factorize(
            gaussian.electron_repulsion_tensor(basis, positions)
        )
    if representation == "direct":
        return gaussian.DirectJKPlan().prepare(basis, positions)
    raise ValueError(f"Unknown test integral representation: {representation}")


@pytest.mark.parametrize("representation", ("exact", "factorized", "direct"))
def test_native_integral_owner_evaluates_selected_source(representation: str) -> None:
    basis, positions = _hydrogen_basis()
    gaussian = phx.operators.quantum.gaussian
    source = _source(basis, positions, representation)
    plan = gaussian.MolecularIntegralPlan(source)
    density = jnp.asarray([[0.8, 0.1], [0.1, 0.6]], dtype=jnp.float64)
    charges = jnp.ones((2,), dtype=jnp.float64)
    dense = plan.dense_tensor(basis, positions)
    one = plan.one_electron(basis, positions, charges)
    jk = plan.jk(basis, positions, density, dense)
    eri = np.asarray(
        source.reconstruct()
        if isinstance(source, FactorizedERITensor)
        else gaussian.electron_repulsion_tensor(basis, positions)
    )

    assert plan.source is source
    assert plan.representation == representation
    assert bool(one.successful) and bool(jk.successful)
    assert one.definition_id == jk.definition_id == plan.definition_id(basis)
    assert plan.definition_id(basis) == gaussian.molecular_integral_definition_id(
        basis, source
    )
    if representation == "exact":
        assert dense is not None
        np.testing.assert_allclose(dense, eri, rtol=0.0, atol=0.0)
        assert not jax.tree_util.tree_leaves(plan)
    else:
        assert dense is None
        assert jax.tree_util.tree_leaves(plan)
    np.testing.assert_allclose(one.overlap, gaussian.overlap_matrix(basis, positions))
    np.testing.assert_allclose(
        one.core_hamiltonian,
        gaussian.kinetic_matrix(basis, positions)
        + gaussian.nuclear_attraction_matrix(basis, positions, charges),
    )
    np.testing.assert_allclose(
        one.nuclear_repulsion, gaussian.nuclear_repulsion_energy(positions, charges)
    )
    np.testing.assert_allclose(
        jk.coulomb,
        np.einsum("cd,abcd->ab", np.asarray(density), eri),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jk.exchange,
        np.einsum("cd,acbd->ab", np.asarray(density), eri),
        rtol=2.0e-11,
        atol=2.0e-12,
    )


def test_integral_definition_binds_basis_and_actual_source_identity() -> None:
    basis, positions = _hydrogen_basis()
    other_basis, _ = _hydrogen_basis("distinct-integral-provider-basis")
    gaussian = phx.operators.quantum.gaussian
    exact = gaussian.MolecularIntegralPlan()
    factors = gaussian.PivotedCholeskyERIPlan(1.0e-12).factorize(
        gaussian.electron_repulsion_tensor(basis, positions)
    )
    changed_factors = gaussian.FactorizedERITensor(
        0.9 * factors.factors,
        factors.residual_bound,
        factors.source_id,
        factors.representation,
    )
    direct = gaussian.DirectJKPlan().prepare(basis, positions)
    moved_direct = gaussian.DirectJKPlan().prepare(basis, positions.at[1, 2].add(0.01))
    plans = tuple(
        gaussian.MolecularIntegralPlan(source)
        for source in (None, factors, changed_factors, direct, moved_direct)
    )

    assert basis.basis_function_count == other_basis.basis_function_count
    assert basis.prepared_id != other_basis.prepared_id
    assert exact.source is None
    assert exact.representation == "exact"
    assert exact.plan_id == gaussian.MolecularIntegralPlan(None).plan_id
    assert exact.definition_id(basis) == gaussian.molecular_integral_definition_id(basis)
    assert exact.definition_id(basis) != exact.definition_id(other_basis)
    assert len({plan.plan_id for plan in plans}) == len(plans)
    assert len({plan.definition_id(basis) for plan in plans}) == len(plans)
    assert gaussian.MolecularIntegralPlan(factors).plan_id == plans[1].plan_id
    assert gaussian.MolecularIntegralPlan(direct).definition_id(basis) == plans[
        3
    ].definition_id(basis)
    assert plans[1].definition_id(basis) != plans[1].definition_id(other_basis)
    assert any(leaf is factors.factors for leaf in jax.tree_util.tree_leaves(plans[1]))


def test_direct_integral_owner_rejects_equal_size_foreign_basis() -> None:
    basis, positions = _hydrogen_basis()
    other_basis, _ = _hydrogen_basis("foreign-direct-owner")
    gaussian = phx.operators.quantum.gaussian
    direct = gaussian.DirectJKPlan().prepare(basis, positions)
    plan = gaussian.MolecularIntegralPlan(direct)

    assert basis.basis_function_count == other_basis.basis_function_count
    with pytest.raises(ValueError, match="another Gaussian basis"):
        plan.validate_basis(other_basis)
    with pytest.raises(ValueError, match="another Gaussian basis"):
        plan.definition_id(other_basis)
    with pytest.raises(ValueError, match="another Gaussian basis"):
        gaussian.molecular_integral_definition_id(other_basis, direct)


def test_factorized_integral_owner_rejects_wrong_ao_capacity() -> None:
    basis, _ = _hydrogen_basis()
    gaussian = phx.operators.quantum.gaussian
    factors = gaussian.FactorizedERITensor(
        np.ones((1, 1, 1), dtype=np.float64), 0.0, "wrong-ao-capacity", "fixed"
    )
    plan = gaussian.MolecularIntegralPlan(factors)

    with pytest.raises(ValueError, match="Gaussian basis"):
        plan.validate_basis(basis)
    with pytest.raises(ValueError, match="Gaussian basis"):
        plan.definition_id(basis)
    with pytest.raises(ValueError, match="Gaussian basis"):
        gaussian.molecular_integral_definition_id(basis, factors)


def test_direct_integral_owner_preserves_geometry_validity_failure() -> None:
    basis, positions = _hydrogen_basis()
    gaussian = phx.operators.quantum.gaussian
    direct = gaussian.DirectJKPlan().prepare(basis, positions)
    plan = gaussian.MolecularIntegralPlan(direct)
    moved = positions.at[1, 2].add(0.5)
    density = jnp.eye(2, dtype=jnp.float64)
    result = plan.jk(basis, moved, density, None)

    assert not bool(result.successful)
    assert result.definition_id == plan.definition_id(basis)
    np.testing.assert_allclose(result.coulomb, direct.evaluate(moved, density).coulomb)


def test_integral_owner_rejects_unprepared_source() -> None:
    gaussian = phx.operators.quantum.gaussian
    with pytest.raises(TypeError, match="Integral source"):
        # ty: ignore[invalid-argument-type]
        gaussian.MolecularIntegralPlan(gaussian.DirectJKPlan())


def test_actual_density_factor_reuses_damped_factors_not_new_orbitals() -> None:
    from phydrax.operators.quantum.gaussian._cuest import _density_factors

    old = jnp.asarray([[1.0, 0.0]], dtype=jnp.float64)
    new = jnp.asarray([[0.0, 1.0]], dtype=jnp.float64)
    factors = jnp.concatenate((jnp.sqrt(0.3) * old, jnp.sqrt(0.7) * new), axis=0)
    density = 0.3 * old.T @ old + 0.7 * new.T @ new
    reused, certificate = _density_factors(density, 1.0e-12, factors)
    reconstructed, fallback = _density_factors(density, 1.0e-12, new)

    assert reused is factors
    assert not bool(certificate.used_eigensolver)
    assert bool(fallback.used_eigensolver)
    np.testing.assert_allclose(reconstructed.T @ reconstructed, density, atol=1.0e-14)
    assert float(certificate.reconstruction_residual) <= float(certificate.tolerance)
    assert int(certificate.factor_bytes) == factors.size * factors.dtype.itemsize


def test_actual_density_factor_reports_admitted_negative_roundoff() -> None:
    from phydrax.operators.quantum.gaussian._cuest import _density_factors

    density = jnp.diag(jnp.asarray([1.0, -1.0e-15], dtype=jnp.float64))
    factors, certificate = _density_factors(density, 1.0e-12)

    assert bool(certificate.used_eigensolver)
    assert float(certificate.minimum_eigenvalue) < 0.0
    np.testing.assert_allclose(certificate.negative_roundoff_magnitude, 1.0e-15)
    np.testing.assert_allclose(certificate.reconstruction_residual, 1.0e-15)
    np.testing.assert_allclose(factors.T @ factors, np.diag([1.0, 0.0]), atol=1.0e-15)
    with pytest.raises(ValueError, match="indefinite"):
        _density_factors(jnp.diag(jnp.asarray([1.0, -0.1], dtype=jnp.float64)), 1.0e-12)


def test_zero_actual_density_has_no_occupied_factor_operation() -> None:
    from phydrax.operators.quantum.gaussian._cuest import _density_factors

    factors, certificate = _density_factors(jnp.zeros((2, 2), dtype=jnp.float64), 1.0e-12)

    assert factors.shape == (0, 2)
    assert not bool(certificate.used_eigensolver)
    assert int(certificate.factor_bytes) == 0
    assert float(certificate.reconstruction_residual) == 0.0


@pytest.mark.parametrize("supply_factors", (False, True))
def test_supplied_factors_cannot_bypass_spectral_psd_admission(
    supply_factors: bool,
) -> None:
    from phydrax.operators.quantum.gaussian._cuest import _density_factors

    tolerance = 1.0e-12
    density = -0.75 * tolerance * jnp.ones((2, 2), dtype=jnp.float64)
    candidate = jnp.zeros((1, 2), dtype=jnp.float64) if supply_factors else None
    with pytest.raises(ValueError, match="indefinite"):
        _density_factors(density, tolerance, candidate)


@pytest.mark.parametrize("supply_factors", (False, True))
def test_supplied_factors_cannot_bypass_density_symmetry_admission(
    supply_factors: bool,
) -> None:
    from phydrax.operators.quantum.gaussian._cuest import _density_factors

    tolerance = 1.0e-12
    density = jnp.asarray(
        [[1.0, 0.75 * tolerance], [-0.75 * tolerance, 1.0]], dtype=jnp.float64
    )
    candidate = jnp.eye(2, dtype=jnp.float64) if supply_factors else None
    with pytest.raises(ValueError, match="symmetric density"):
        _density_factors(density, tolerance, candidate)


def test_candidate_roundoff_reports_bound_instead_of_unmeasured_zero() -> None:
    from phydrax.operators.quantum.gaussian._cuest import _density_factors

    tolerance = 1.0e-12
    density = jnp.diag(jnp.asarray([1.0, -0.25 * tolerance], dtype=jnp.float64))
    candidate = jnp.asarray([[1.0, 0.0]], dtype=jnp.float64)
    factors, certificate = _density_factors(density, tolerance, candidate)

    assert factors is candidate
    assert not bool(certificate.used_eigensolver)
    assert bool(jnp.isnan(certificate.minimum_eigenvalue))
    assert bool(jnp.isnan(certificate.negative_roundoff_magnitude))
    np.testing.assert_allclose(
        certificate.minimum_eigenvalue_lower_bound, -0.25 * tolerance
    )
    assert float(certificate.minimum_eigenvalue_lower_bound) >= -tolerance
    np.testing.assert_allclose(certificate.reconstruction_residual, 0.25 * tolerance)


@pytest.mark.parametrize(
    "invalid",
    (
        "position-shape",
        "center-count",
        "charge-shape",
        "nonfinite",
        "precision",
        "definition",
    ),
)
def test_geometry_binding_validates_recipe_before_optional_provider_load(
    invalid: str,
) -> None:
    gaussian = phx.operators.quantum.gaussian
    basis, positions = _hydrogen_basis()
    auxiliary, _ = _hydrogen_basis("spectral-admission-auxiliary", spherical=True)
    source = gaussian.CuestIntegralPlan(
        auxiliary,
        fitting_cutoff=1.0e-12,
        relative_conditioning=False,
        fitting_algorithm="matrix-power",
        screening_threshold=0.0,
        variable_workspace_bytes=1024,
        maximum_workspace_bytes=1024**2,
        density_factorization_tolerance=1.0e-12,
    ).prepare(basis)
    charges = jnp.ones((2,), dtype=jnp.float64)
    if invalid == "position-shape":
        positions = jnp.zeros((2, 2), dtype=jnp.float64)
    elif invalid == "center-count":
        positions = jnp.zeros((1, 3), dtype=jnp.float64)
        charges = jnp.ones((1,), dtype=jnp.float64)
    elif invalid == "charge-shape":
        charges = jnp.ones((3,), dtype=jnp.float64)
    elif invalid == "nonfinite":
        positions = jnp.full(positions.shape, jnp.nan, dtype=jnp.float64)
    elif invalid == "precision":
        positions = positions.astype(jnp.float32)
    elif invalid == "definition":
        import equinox as eqx

        source = eqx.tree_at(
            lambda prepared: prepared.orbital_layout.exponents,
            source,
            source.orbital_layout.exponents * 1.1,
        )
    else:
        raise ValueError(invalid)
    with pytest.raises(ValueError, match="(?i)(geometry|FP64|definition)"):
        source.bind_geometry(positions, charges)
