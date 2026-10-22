"""Real cuEST/PySCF parity; absent optional routes skip, admitted failures fail."""

from __future__ import annotations

from typing import Literal, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from benchmarks.cuest_molecular import build_case, CuestCase, external_reference
from phydrax.operators.quantum.gaussian import MolecularOneElectronResult


pytestmark = [pytest.mark.cuda_native, pytest.mark.usefixtures("cuest_runtime")]


@pytest.fixture
def case(cuest_runtime: None) -> CuestCase:
    # Runtime admission is shared by every case, including parametrized tests.
    # Only an absent optional route or independent reference may skip.
    pytest.importorskip(
        "pyscf", reason="independent matched DF integration oracle requires PySCF"
    )
    return build_case()


def admitted_one_electron(
    case: CuestCase, positions: np.ndarray | None = None
) -> MolecularOneElectronResult:
    coordinates = case.positions if positions is None else positions
    try:
        result = case.integrals.one_electron(
            case.basis, coordinates, np.ones(len(coordinates))
        )
        jax.block_until_ready(result)
    except (ImportError, OSError, RuntimeError) as error:
        pytest.fail(f"cuEST provider NOT ADMITTED: {type(error).__name__}: {error}")
    assert bool(result.successful), (
        "cuEST provider NOT ADMITTED: one-electron evaluation unsuccessful"
    )
    return result


@pytest.mark.parametrize("angular", (0, 1, 2, 3))
def test_s_t_v_match_independent_spherical_ao_reference(angular: int) -> None:
    pytest.importorskip("pyscf", reason="independent one-electron oracle requires PySCF")
    case = build_case(angular=angular)
    reference = external_reference(case)
    one = admitted_one_electron(case)
    permutation = case.pyscf_to_native
    for actual, integral in (
        (one.overlap, "int1e_ovlp"),
        (one.kinetic, "int1e_kin"),
        (one.nuclear_attraction, "int1e_nuc"),
    ):
        expected = reference.mol.intor(integral)[np.ix_(permutation, permutation)]
        np.testing.assert_allclose(actual, expected, rtol=2.0e-10, atol=2.0e-11)
        np.testing.assert_allclose(actual, np.asarray(actual).T, atol=2.0e-12)
    np.testing.assert_allclose(
        one.core_hamiltonian, one.kinetic + one.nuclear_attraction, atol=2.0e-12
    )
    np.testing.assert_allclose(
        one.nuclear_repulsion, reference.mol.energy_nuc(), atol=2.0e-12
    )
    assert one.definition_id == case.integrals.definition_id(case.basis)
    assert case.integrals.dense_tensor(case.basis, case.positions) is None


@pytest.mark.parametrize("algorithm", ("matrix-power", "qr"))
@pytest.mark.parametrize("relative", (False, True))
def test_jk_uses_actual_changed_damped_rank_deficient_and_zero_density(
    algorithm: Literal["matrix-power", "qr"], relative: bool
) -> None:
    pytest.importorskip(
        "pyscf", reason="independent matched DF J/K oracle requires PySCF"
    )
    case = build_case(fitting_algorithm=algorithm, relative_conditioning=relative)
    admitted_one_electron(case)
    reference = external_reference(case)
    reference.kernel()
    assert reference.converged
    permutation = case.pyscf_to_native
    inverse = np.argsort(permutation)
    ground = reference.make_rdm1()[np.ix_(permutation, permutation)]
    rng = np.random.default_rng(173)
    vectors = rng.normal(size=(ground.shape[0], 3))
    changed = vectors @ vectors.T / ground.shape[0] + 0.05 * np.eye(ground.shape[0])
    rank_one = np.diag(np.r_[0.7, np.zeros(ground.shape[0] - 1)])
    cases = {
        "ground": ground,
        "changed": changed,
        "damped": 0.37 * ground + 0.63 * changed,
        "rank_deficient": rank_one,
        "zero": np.zeros_like(ground),
        "ground_after_zero": ground,
    }
    observed = {}
    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    for name, density in cases.items():
        result = bound.jk(density)
        assert bool(result.successful), name
        pyscf_density = density[np.ix_(inverse, inverse)]
        expected_j, expected_k = reference.get_jk(dm=pyscf_density)
        np.testing.assert_allclose(
            result.coulomb,
            expected_j[np.ix_(permutation, permutation)],
            rtol=3.0e-8,
            atol=2.0e-9,
            err_msg=name,
        )
        np.testing.assert_allclose(
            result.exchange,
            expected_k[np.ix_(permutation, permutation)],
            rtol=3.0e-8,
            atol=2.0e-9,
            err_msg=name,
        )
        assert result.definition_id == case.integrals.definition_id(case.basis)
        observed[name] = (np.asarray(result.coulomb), np.asarray(result.exchange))
    assert not np.allclose(observed["ground"][0], observed["changed"][0])
    for index in (0, 1):
        np.testing.assert_allclose(observed["zero"][index], 0.0, atol=1.0e-14)
        np.testing.assert_allclose(
            observed["ground_after_zero"][index], observed["ground"][index], atol=2.0e-12
        )
        np.testing.assert_allclose(
            observed["damped"][index],
            0.37 * observed["ground"][index] + 0.63 * observed["changed"][index],
            atol=2.0e-10,
        )


def test_geometry_cache_does_not_reuse_stale_integrals(case: CuestCase) -> None:
    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    first = bound.one_electron()
    assert bool(first.successful)
    moved = case.positions.copy()
    moved[1] += np.asarray([0.12, -0.08, 0.21])
    changed = bound.refresh(positions=moved)
    second = changed.one_electron()
    assert bool(second.successful)
    reference = external_reference(case, moved)
    permutation = case.pyscf_to_native
    np.testing.assert_allclose(
        second.core_hamiltonian,
        reference.get_hcore()[np.ix_(permutation, permutation)],
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert not np.allclose(first.core_hamiltonian, second.core_hamiltonian)
    density = np.eye(case.basis.basis_function_count) * 0.4
    actual = changed.jk(density)
    assert bool(actual.successful)
    expected_j, expected_k = reference.get_jk(dm=density)
    np.testing.assert_allclose(
        actual.coulomb,
        expected_j[np.ix_(permutation, permutation)],
        rtol=3.0e-8,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        actual.exchange,
        expected_k[np.ix_(permutation, permutation)],
        rtol=3.0e-8,
        atol=2.0e-9,
    )
    restored = bound.one_electron()
    assert bool(restored.successful)
    np.testing.assert_allclose(
        restored.core_hamiltonian, first.core_hamiltonian, atol=2.0e-12
    )


def test_jk_jvp_and_vjp_are_explicitly_unsupported(case: CuestCase) -> None:
    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    density = jnp.eye(case.basis.basis_function_count, dtype=jnp.float64)

    def evaluate(value: Array) -> Array:
        return bound.jk(value).exchange

    with pytest.raises(NotImplementedError, match="(?i)(AD|autodiff|differentiat|JVP)"):
        jax.jvp(evaluate, (density,), (jnp.ones_like(density),))
    with pytest.raises(
        NotImplementedError, match="(?i)(AD|autodiff|differentiat|JVP|VJP)"
    ):
        jax.grad(lambda value: jnp.sum(evaluate(value)))(density)


def test_bound_jk_rejects_indefinite_density_with_or_without_candidate_factors(
    case: CuestCase,
) -> None:
    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    count = case.basis.basis_function_count
    assert count >= 2
    tolerance = case.source.density_factorization_tolerance
    density = -0.75 * tolerance * np.ones((count, count), dtype=np.float64)
    for candidate in (None, np.zeros((1, count), dtype=np.float64)):
        with pytest.raises(ValueError, match="(?i)indefinite"):
            bound.jk(density, factors=candidate)


def test_bound_execution_reuses_admission_without_environment_reprobe(
    case: CuestCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.operators.quantum.gaussian import _cuest

    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions), dtype=np.float64)
    )
    count = case.basis.basis_function_count
    density = np.eye(count, dtype=np.float64) * 0.4
    weighted = np.eye(count, dtype=np.float64) * 0.2

    def unexpected_admission(name: str) -> NoReturn:
        raise AssertionError("Warm bound execution must not reprobe CUDA admission.")

    monkeypatch.setattr(_cuest, "load_cuda_component", unexpected_admission)
    one = bound.one_electron()
    jk = bound.jk(density)
    _, _, _, gradient_successful = bound.gradient(density, weighted)
    assert bool(one.successful) and bool(jk.successful) and bool(gradient_successful)
