"""Real cuEST bound-recipe restore/rebuild, resource refusal, and molecular PES."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

import phydrax as phx
from benchmarks.cuest_molecular import build_case, CuestCase, external_reference


pytestmark = [pytest.mark.cuda_native, pytest.mark.usefixtures("cuest_runtime")]


@pytest.fixture
def case(cuest_runtime: None) -> CuestCase:
    case = build_case()
    try:
        one = case.integrals.one_electron(case.basis, case.positions, np.ones(2))
        jax.block_until_ready(one)
    except (ImportError, OSError, RuntimeError) as error:
        pytest.fail(f"cuEST provider NOT ADMITTED: {type(error).__name__}: {error}")
    assert bool(one.successful), "cuEST provider NOT ADMITTED"
    return case


def test_resource_shortage_is_native_failure_and_does_not_poison_normal_plan(
    case: CuestCase,
) -> None:
    # Positive, internally valid policies: failure must come from actual vendor
    # workspace admission, not from Python argument validation or missing cuEST.
    constrained = build_case(variable_workspace_bytes=1, maximum_workspace_bytes=1)
    with pytest.raises(RuntimeError):
        one = constrained.integrals.one_electron(
            constrained.basis, constrained.positions, np.ones(2)
        )
        jax.block_until_ready(one)
    restored = case.integrals.one_electron(case.basis, case.positions, np.ones(2))
    assert bool(restored.successful)
    state = case.hf.solve_atomic_units(case.positions)
    assert bool(state.evidence.successful)


def test_all_policies_and_actual_basis_change_definition_identity(
    case: CuestCase,
) -> None:
    gaussian = phx.operators.quantum.gaussian

    def changed_source(
        *,
        fitting_cutoff: float = case.source.fitting_cutoff,
        relative_conditioning: bool = case.source.relative_conditioning,
        fitting_algorithm: Literal["qr", "matrix-power"] = case.source.fitting_algorithm,
        screening_threshold: float = case.source.screening_threshold,
        variable_workspace_bytes: int = case.source.variable_workspace_bytes,
        maximum_workspace_bytes: int = case.source.maximum_workspace_bytes,
        maximum_cached_geometries: int = case.source.maximum_cached_geometries,
    ) -> phx.operators.quantum.gaussian.CuestIntegralPlan:
        return gaussian.CuestIntegralPlan(
            case.auxiliary,
            fitting_cutoff=fitting_cutoff,
            relative_conditioning=relative_conditioning,
            fitting_algorithm=fitting_algorithm,
            screening_threshold=screening_threshold,
            variable_workspace_bytes=variable_workspace_bytes,
            maximum_workspace_bytes=maximum_workspace_bytes,
            maximum_cached_geometries=maximum_cached_geometries,
            density_factorization_tolerance=case.source.density_factorization_tolerance,
        )

    identity = case.integrals.definition_id(case.basis)
    variants = (
        changed_source(fitting_cutoff=2.0e-12),
        changed_source(relative_conditioning=True),
        changed_source(fitting_algorithm="qr"),
        changed_source(screening_threshold=2.0e-14),
        changed_source(variable_workspace_bytes=32 * 1024**2),
        changed_source(maximum_workspace_bytes=1024 * 1024**2),
        changed_source(maximum_cached_geometries=2),
    )
    definitions = set()
    for source in variants:
        plan = gaussian.MolecularIntegralPlan(source)
        assert source.plan_id != case.source.plan_id
        definition = plan.definition_id(case.basis)
        assert definition != identity
        definitions.add(definition)
    assert len(definitions) == len(variants)
    # Changing real exponents must change observable values, not only a label.
    changed = build_case(exponent_scale=1.12)
    assert changed.integrals.definition_id(changed.basis) != identity
    original = case.integrals.one_electron(case.basis, case.positions, np.ones(2))
    actual = changed.integrals.one_electron(changed.basis, changed.positions, np.ones(2))
    assert bool(actual.successful)
    assert not np.allclose(original.kinetic, actual.kinetic)
    first_state = case.hf.solve_atomic_units(case.positions)
    second_state = changed.hf.solve_atomic_units(changed.positions)
    assert bool(first_state.evidence.successful) and bool(
        second_state.evidence.successful
    )
    assert abs(float(first_state.total_energy - second_state.total_energy)) > 1.0e-5
    with pytest.raises(ValueError, match="(?i)(another|definition|owner|plan)"):
        changed.hf.analytic_gradient_atomic_units(changed.positions, first_state)


@pytest.mark.parametrize("basis_kind", ("orbital", "auxiliary"))
def test_changed_basis_energy_matches_independent_df(
    case: CuestCase, basis_kind: str
) -> None:
    pytest.importorskip(
        "pyscf", reason="independent changed-basis DF oracle requires PySCF"
    )
    changed = (
        build_case(exponent_scale=1.12)
        if basis_kind == "orbital"
        else build_case(auxiliary_exponent_scale=1.2)
    )
    assert changed.integrals.definition_id(changed.basis) != case.integrals.definition_id(
        case.basis
    )
    actual = changed.hf.solve_atomic_units(changed.positions)
    reference = external_reference(changed)
    expected = reference.kernel()
    assert reference.converged and bool(actual.evidence.successful)
    np.testing.assert_allclose(actual.total_energy, expected, atol=2.0e-9)


def test_releasing_registry_rebuilds_same_bound_full_recipe_and_actual_density(
    case: CuestCase,
) -> None:
    from phydrax._cuda_native import cuda_native_resources, release_cuda_component

    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    rebuilt = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    np.testing.assert_array_equal(bound.key, rebuilt.key)
    count = case.basis.basis_function_count
    factors = np.zeros((2, count), dtype=np.float64)
    factors[0, 0], factors[1, 1] = 0.8, 0.3
    density = factors.T @ factors
    original_one = bound.one_electron()
    explicit = bound.jk(density, factors=factors)
    automatic = rebuilt.jk(density)
    # A stale occupied factor must not replace a changed/damped actual density.
    changed = 0.4 * density + 0.03 * np.eye(count)
    canonical = bound.jk(changed)
    jax.block_until_ready((original_one, explicit, automatic, canonical))
    partition = bound.prepared.partition
    before = [
        part
        for part in cuda_native_resources().partitions
        if part.component == "cuest" and part.partition == partition
    ]
    assert before and all(0 < part.live <= part.capacity == 1 for part in before)
    assert release_cuda_component("cuest") > 0
    assert not any(
        part.live
        for part in cuda_native_resources().partitions
        if part.component == "cuest"
    )
    # No re-prepare/re-bind: the retained dynamic recipe must reconstruct after
    # release while consuming the changed density, not the prior factors.
    stale = bound.jk(changed, factors=factors)
    restored_one = bound.one_electron()
    jax.block_until_ready((stale, restored_one))
    after = [
        part
        for part in cuda_native_resources().partitions
        if part.component == "cuest" and part.partition == partition
    ]
    assert after and all(
        0 < part.live <= part.capacity == 1 and part.rebuilds > 0 for part in after
    )
    # Canonical release_all removes the partition and its counters. A fresh
    # generation reconstructs once; the following one-electron call is a hit.
    assert all(part.rebuilds == 1 for part in after)
    assert bool(original_one.successful) and bool(restored_one.successful)
    np.testing.assert_allclose(
        restored_one.core_hamiltonian, original_one.core_hamiltonian, atol=2.0e-12
    )
    assert all(
        bool(value.successful) for value in (explicit, automatic, stale, canonical)
    )
    for field in ("coulomb", "exchange"):
        np.testing.assert_allclose(
            getattr(explicit, field), getattr(automatic, field), atol=2.0e-11
        )
        np.testing.assert_allclose(
            getattr(stale, field), getattr(canonical, field), atol=2.0e-11
        )
        assert not np.allclose(getattr(stale, field), getattr(explicit, field))
    assert (
        explicit.definition_id
        == stale.definition_id
        == case.integrals.definition_id(case.basis)
    )


@pytest.mark.parametrize("operation", ("one_electron", "jk", "gradient"))
def test_replaced_positions_refuse_execution_until_explicit_refresh(
    case: CuestCase, operation: str
) -> None:
    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    density = np.eye(case.basis.basis_function_count, dtype=np.float64) * 0.4
    moved = case.positions.copy()
    moved[1] += np.asarray([0.12, -0.08, 0.21])
    replaced = eqx.tree_at(lambda geometry: geometry.positions, bound, jnp.asarray(moved))
    with pytest.raises(ValueError, match="(?i)(binding|refresh|recipe)"):
        if operation == "one_electron":
            replaced.one_electron()
        elif operation == "jk":
            replaced.jk(density)
        else:
            replaced.gradient(density, np.zeros_like(density))
    with pytest.raises(ValueError, match="(?i)certificate"):
        replaced.restore()
    refreshed = replaced.refresh(positions=moved)
    fresh = case.source.prepare(case.basis).bind_geometry(moved, np.ones(len(moved)))
    np.testing.assert_array_equal(refreshed.key, fresh.key)
    assert not np.array_equal(refreshed.key, bound.key)
    actual, expected = refreshed.one_electron(), fresh.one_electron()
    assert bool(actual.successful) and bool(expected.successful)
    np.testing.assert_allclose(
        actual.core_hamiltonian, expected.core_hamiltonian, atol=2.0e-12
    )
    actual_jk, expected_jk = refreshed.jk(density), fresh.jk(density)
    assert bool(actual_jk.successful) and bool(expected_jk.successful)
    for field in ("coulomb", "exchange"):
        np.testing.assert_allclose(
            getattr(actual_jk, field), getattr(expected_jk, field), atol=2.0e-11
        )


def test_replaced_charges_require_refresh_and_remain_dynamic(case: CuestCase) -> None:
    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    original = bound.one_electron()
    charges = np.asarray([1.0, 1.25], dtype=np.float64)
    replaced = eqx.tree_at(lambda geometry: geometry.charges, bound, jnp.asarray(charges))
    with pytest.raises(ValueError, match="(?i)(binding|refresh|recipe)"):
        replaced.one_electron()
    refreshed = replaced.refresh(charges=charges)
    # Charges affect V and nuclear terms, not the basis/geometry DF memo identity.
    np.testing.assert_array_equal(refreshed.key, bound.key)
    actual = refreshed.one_electron()
    expected = case.integrals.one_electron(case.basis, case.positions, charges)
    assert (
        bool(original.successful)
        and bool(actual.successful)
        and bool(expected.successful)
    )
    np.testing.assert_allclose(actual.overlap, original.overlap, atol=2.0e-12)
    np.testing.assert_allclose(actual.kinetic, original.kinetic, atol=2.0e-12)
    np.testing.assert_allclose(
        actual.core_hamiltonian, expected.core_hamiltonian, atol=2.0e-12
    )
    assert not np.allclose(actual.nuclear_attraction, original.nuclear_attraction)


def test_same_content_tree_replacement_restores_and_rebuilds_full_recipe(
    case: CuestCase,
) -> None:
    from phydrax._cuda_native import release_cuda_component

    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    density = np.eye(case.basis.basis_function_count, dtype=np.float64) * 0.4
    original_one, original_jk = bound.one_electron(), bound.jk(density)
    copied = jax.tree.map(
        lambda leaf: jnp.array(leaf, copy=True) if eqx.is_array(leaf) else leaf, bound
    )
    with pytest.raises(ValueError, match="(?i)(binding|refresh|recipe)"):
        copied.one_electron()
    assert release_cuda_component("cuest") > 0
    restored = copied.restore()
    np.testing.assert_array_equal(restored.key, bound.key)
    actual_one, actual_jk = restored.one_electron(), restored.jk(density)
    assert all(
        bool(value.successful)
        for value in (original_one, original_jk, actual_one, actual_jk)
    )
    np.testing.assert_allclose(
        actual_one.core_hamiltonian, original_one.core_hamiltonian, atol=2.0e-12
    )
    for field in ("coulomb", "exchange"):
        np.testing.assert_allclose(
            getattr(actual_jk, field), getattr(original_jk, field), atol=2.0e-11
        )


def test_corrupted_binding_key_cannot_be_restored(case: CuestCase) -> None:
    bound = case.source.prepare(case.basis).bind_geometry(
        case.positions, np.ones(len(case.positions))
    )
    corrupted = eqx.tree_at(
        lambda geometry: geometry.key, bound, bound.key.at[0].add(jnp.int64(1))
    )
    with pytest.raises(ValueError, match="(?i)(binding|refresh|recipe)"):
        corrupted.one_electron()
    with pytest.raises(ValueError, match="(?i)certificate"):
        corrupted.restore()


@pytest.mark.parametrize("mutation", ("exponents", "source"))
def test_prepared_recipe_definition_mismatch_refuses_binding(
    case: CuestCase, mutation: str
) -> None:
    prepared = case.source.prepare(case.basis)
    if mutation == "exponents":
        changed = eqx.tree_at(
            lambda recipe: recipe.orbital_layout.exponents,
            prepared,
            prepared.orbital_layout.exponents * 1.12,
        )
    else:
        different = build_case(fitting_cutoff=2.0e-12)
        changed = eqx.tree_at(lambda recipe: recipe.plan, prepared, different.source)
    with pytest.raises(ValueError, match="(?i)definition"):
        changed.bind_geometry(case.positions, np.ones(len(case.positions)))
    bound = prepared.bind_geometry(case.positions, np.ones(len(case.positions)))
    replaced = eqx.tree_at(lambda geometry: geometry.prepared, bound, changed)
    with pytest.raises(ValueError, match="(?i)(binding|refresh|recipe)"):
        replaced.one_electron()
    with pytest.raises(ValueError, match="(?i)definition"):
        replaced.restore()


def workflow(
    case: CuestCase,
    task: phx.chemistry.GroundStateTaskPlan | None = None,
    *,
    numerical: bool = True,
) -> tuple[phx.chemistry.ElectronicCalculationPlan, phx.chemistry.MolecularRHFProvider]:
    model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.HartreeFockMethodPlan(
            phx.chemistry.ElectronicReferenceKind.RESTRICTED
        ),
        basis=phx.chemistry.BasisSetReference(
            "explicit-cuest-qualification",
            case.basis.plan.source_id,
            convention="spherical",
        ),
    )
    provider = phx.chemistry.MolecularRHFProvider(case.hf, model.model_chemistry_id)
    calculation = phx.chemistry.ElectronicCalculationPlan(
        case.system,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        model,
        phx.chemistry.GroundStateTaskPlan.energy_and_forces() if task is None else task,
        numerical=provider.numerical if numerical else None,
    )
    return calculation, provider


def test_typed_force_result_archive_pes_and_born_oppenheimer_step(
    case: CuestCase, tmp_path: Path
) -> None:
    calculation, provider = workflow(case)
    prepared = provider.prepare(calculation)
    result = prepared.evaluate(case.structure.positions)
    assert isinstance(result, phx.chemistry.ElectronicEnergyForceEvaluation)
    assert bool(result.successful)
    direct_state, direct_forces = case.hf.evaluate(case.structure.positions)
    conversion = float(
        phx.units.conversion_factor(
            phx.units.HARTREE, case.system.units.scale.energy_unit
        )
    )
    np.testing.assert_allclose(
        np.asarray(result.energy, dtype=np.float64),
        np.asarray(direct_state.total_energy, dtype=np.float64) * conversion,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        np.asarray(result.forces, dtype=np.float64),
        np.asarray(direct_forces, dtype=np.float64),
        atol=2.0e-9,
    )
    np.testing.assert_array_equal(
        result.header.stable_particle_ids, case.system.particle_ids
    )
    assert case.integrals.definition_id(case.basis) in result.header.artifact_ids
    path = tmp_path / "cuest-electronic-result.phx"
    phx.chemistry.write_electronic_result_archive(
        path, result, run_id="real-cuest-df-rhf"
    )
    restored = phx.chemistry.read_electronic_result_archive(path)
    assert isinstance(restored, phx.chemistry.ElectronicEnergyForceEvaluation)
    assert restored.result_id == result.result_id
    np.testing.assert_allclose(
        np.asarray(restored.forces, dtype=np.float64),
        np.asarray(result.forces, dtype=np.float64),
        atol=0.0,
    )
    lifecycle = phx.chemistry.chemistry_lifecycle(
        calculation, result, provider_capability_id=provider.capabilities.capabilities_id
    )
    assert lifecycle.run.status == "completed"
    surface = phx.chemistry.ElectronicPotentialEnergySurface(prepared)
    evaluated = surface.evaluate(case.structure.positions)
    assert bool(evaluated.successful)
    assert not surface.capabilities.differentiable and not surface.capabilities.hessian
    np.testing.assert_allclose(
        np.asarray(evaluated.energy, dtype=np.float64),
        np.asarray(result.energy, dtype=np.float64),
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        np.asarray(evaluated.forces, dtype=np.float64),
        np.asarray(result.forces, dtype=np.float64),
        atol=2.0e-9,
    )
    adapter = phx.chemistry.SurfaceExternalAtomisticProvider(surface)
    dynamics = phx.atomistic.BornOppenheimerVelocityVerletPlan(
        case.system.prepare(), adapter, 1.0e-3
    )
    state = dynamics.initialize(
        case.structure.positions, velocity=jnp.zeros_like(case.structure.positions)
    )
    step = dynamics.step(state)
    assert bool(step.successful)
    assert int(step.state.step_index) == 1
    assert not np.array_equal(
        np.asarray(step.state.positions), np.asarray(case.structure.positions)
    )


def test_typed_workflow_refuses_hessian_and_unbound_numerical_definition(
    case: CuestCase,
) -> None:
    calculation, provider = workflow(
        case, phx.chemistry.GroundStateTaskPlan.energy_forces_and_hessian()
    )
    with pytest.raises(
        phx.chemistry.ElectronicCapabilityError, match="(?i)(hessian|propert|derivative)"
    ):
        provider.prepare(calculation)
    unbound, provider = workflow(case, numerical=False)
    with pytest.raises(
        phx.chemistry.ElectronicCapabilityError, match="(?i)(numerical|definition)"
    ):
        provider.prepare(unbound)


def test_typed_force_workflow_reuses_one_local_geometry_binding(
    case: CuestCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gaussian = phx.operators.quantum.gaussian
    original = gaussian.PreparedCuestIntegrals.bind_geometry
    bindings = 0

    def bind(
        prepared: phx.operators.quantum.gaussian.PreparedCuestIntegrals,
        positions: ArrayLike,
        charges: ArrayLike,
        /,
    ) -> phx.operators.quantum.gaussian.CuestGeometry:
        nonlocal bindings
        bindings += 1
        return original(prepared, positions, charges)

    monkeypatch.setattr(gaussian.PreparedCuestIntegrals, "bind_geometry", bind)
    calculation, provider = workflow(case)
    result = calculation.prepare(provider).evaluate(case.structure.positions)
    assert isinstance(result, phx.chemistry.ElectronicEnergyForceEvaluation)
    assert bool(result.successful)
    assert bindings == 1
