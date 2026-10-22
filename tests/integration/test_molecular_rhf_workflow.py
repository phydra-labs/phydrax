from pathlib import Path
from typing import Any, Literal, NoReturn

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


def _case(**options: Any) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        np.asarray([1, 1]),
        np.asarray([[0.0, 0.0, -0.4], [0.0, 0.0, 0.4]], dtype=np.float64),
        np.asarray([1.008, 1.008]),
        units.scale,
        particle_ids=np.asarray([11, 17]),
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
        [11, 17],
        np.asarray([[3.42525091, 0.62391373, 0.16885540]] * 2),
        np.asarray([[0.15432897, 0.53532814, 0.44463454]] * 2),
        source_id="molecular-rhf-workflow-sto-3g-h",
    ).prepare(system)
    plan = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        **options,
    )
    model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.HartreeFockMethodPlan(),
        basis=phx.chemistry.BasisSetReference(
            "STO-3G hydrogen", basis.plan.source_id, convention="cartesian"
        ),
    )
    provider = phx.chemistry.MolecularRHFProvider(plan, model.model_chemistry_id)
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        plan.sector,
        model,
        phx.chemistry.GroundStateTaskPlan.energy_and_forces(),
        numerical=provider.numerical,
    )
    return structure, provider, calculation


def test_native_exact_typed_pes_units_work_and_restart(tmp_path: Path) -> None:
    structure, provider, calculation = _case()
    prepared = calculation.prepare(provider)
    result = prepared.evaluate(structure.positions)
    assert isinstance(result, phx.chemistry.ElectronicEnergyForceEvaluation)
    assert bool(result.successful)
    assert int(result.header.work.energy_evaluations) == 1
    assert int(result.header.work.force_evaluations) == 1
    assert int(result.header.work.hessian_evaluations) == 0
    assert int(result.header.work.property_evaluations) == 0
    assert not provider.capabilities.execution.differentiable
    assert not provider.capabilities.execution.batching
    assert provider.capabilities.observables.derivative_orders == (0, 1)

    length = float(
        phx.units.conversion_factor(
            calculation.system.units.scale.length_unit, phx.units.BOHR
        )
    )
    energy = float(
        phx.units.conversion_factor(
            phx.units.HARTREE, calculation.system.units.scale.energy_unit
        )
    )
    coordinate_bohr = np.asarray(structure.positions) * length
    state = provider.plan.solve_atomic_units(coordinate_bohr)
    gradient = provider.plan.analytic_gradient_atomic_units(coordinate_bohr, state)
    np.testing.assert_allclose(result.energy, state.total_energy * energy, atol=1.0e-10)
    np.testing.assert_allclose(
        result.forces, gradient.forces * energy * length, atol=1.0e-9
    )
    definition = provider.plan.integrals.definition_id(provider.plan.basis)
    assert definition in result.header.artifact_ids
    assert calculation.numerical.numerical_id in result.header.artifact_ids
    assert state.state_id in result.header.artifact_ids
    assert dict(result.header.source_unit_ids)["energy"] == phx.units.HARTREE.unit_id

    shifted = np.asarray(structure.positions).copy()
    shifted[1, 2] += 0.08
    changed = prepared.evaluate(shifted)
    assert bool(changed.successful)
    assert changed.header.geometry_id != result.header.geometry_id
    assert changed.result_id != result.result_id
    assert not np.isclose(float(changed.energy), float(result.energy))

    envelope = phx.chemistry.chemistry_lifecycle(
        calculation,
        result,
        provider_capability_id=provider.capabilities.capabilities_id,
    )
    archive = tmp_path / "rhf-result"
    phx.chemistry.write_electronic_result_archive(archive, result, run_id="rhf-pes")
    restored = phx.chemistry.read_electronic_result_archive(archive)
    assert restored.result_id == result.result_id
    assert restored.header.artifact_ids == result.header.artifact_ids
    assert envelope is not None
    changed_numerical = phx.chemistry.ElectronicNumericalPlan(
        integrals=provider.numerical.integrals,
        derivative_route=provider.numerical.derivative_route,
        component_plan_ids=(*provider.numerical.component_plan_ids, "changed-definition"),
    )
    relabeled = phx.chemistry.ElectronicCalculationPlan(
        calculation.system,
        calculation.state,
        calculation.model_chemistry,
        calculation.task,
        numerical=changed_numerical,
    )
    with pytest.raises(ValueError, match="another calculation or numerical definition"):
        phx.chemistry.chemistry_lifecycle(
            relabeled,
            restored,
            provider_capability_id=provider.capabilities.capabilities_id,
        )

    surface = phx.chemistry.ElectronicPotentialEnergySurface(prepared)
    assert surface.capabilities.conservative
    assert not surface.capabilities.hessian
    assert not surface.capabilities.differentiable
    surface_result = surface.evaluate(structure.positions)
    assert bool(surface_result.successful)
    assert surface_result.hessian is None
    assert surface_result.source_result_id == result.result_id
    optimized = phx.chemistry.MolecularGeometryOptimizationPlan(
        calculation.system,
        surface,
        convergence=phx.chemistry.MolecularGeometryConvergencePlan(
            maximum_force=0.03,
            rms_force=0.03,
            maximum_steps=32,
            maximum_evaluations=128,
        ),
    ).run(structure)
    assert bool(optimized.successful)
    assert float(optimized.final_evaluation.energy) < float(result.energy)


def test_force_admission_uses_gradient_success_not_only_scf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure, provider, calculation = _case()
    original = phx.chemistry.MolecularHartreeFockPlan._gradient_at_geometry

    def not_stationary(
        plan: phx.chemistry.MolecularHartreeFockPlan,
        positions: Array,
        state: phx.chemistry.RestrictedMeanFieldState,
        geometry: phx.operators.quantum.gaussian.CuestGeometry | None,
        /,
    ) -> phx.chemistry.MolecularGradientResult:
        assert bool(state.evidence.successful)
        gradient = original(plan, positions, state, geometry)
        return eqx.tree_at(lambda value: value.successful, gradient, jnp.asarray(False))

    monkeypatch.setattr(
        phx.chemistry.MolecularHartreeFockPlan,
        "_gradient_at_geometry",
        not_stationary,
    )
    result = calculation.prepare(provider).evaluate(structure.positions)
    assert not bool(result.successful)
    assert not bool(result.header.convergence.converged)
    assert (
        int(result.header.status)
        == phx.chemistry.ElectronicCalculationStatus.ELECTRONIC_NOT_CONVERGED
    )
    assert int(result.header.work.energy_evaluations) == 1
    assert int(result.header.work.force_evaluations) == 1
    surface = phx.chemistry.ElectronicPotentialEnergySurface(
        calculation.prepare(provider)
    )
    assert not bool(surface.evaluate(structure.positions).successful)


def test_nonconverged_scf_is_a_failed_typed_force_result() -> None:
    structure, provider, calculation = _case(
        convergence=phx.chemistry.SCFConvergencePlan(maximum_iterations=1),
        guess=phx.chemistry.InitialGuessPlan(phx.chemistry.InitialGuessKind.ZERO),
    )
    result = calculation.prepare(provider).evaluate(structure.positions)
    assert not bool(result.successful)
    assert (
        int(result.header.status)
        == phx.chemistry.ElectronicCalculationStatus.ELECTRONIC_NOT_CONVERGED
    )


def test_energy_only_avoids_force_work(monkeypatch: pytest.MonkeyPatch) -> None:
    structure, provider, calculation = _case()
    energy_calculation = phx.chemistry.ElectronicCalculationPlan(
        calculation.system,
        calculation.state,
        calculation.model_chemistry,
        phx.chemistry.GroundStateTaskPlan.energy(),
        numerical=provider.numerical,
    )

    def unexpected_gradient(
        plan: phx.chemistry.MolecularHartreeFockPlan,
        positions: Array,
        state: phx.chemistry.RestrictedMeanFieldState,
        geometry: phx.operators.quantum.gaussian.CuestGeometry | None,
        /,
    ) -> NoReturn:
        raise AssertionError("Energy-only evaluation must not request a gradient")

    monkeypatch.setattr(
        phx.chemistry.MolecularHartreeFockPlan,
        "_gradient_at_geometry",
        unexpected_gradient,
    )
    result = energy_calculation.prepare(provider).evaluate(structure.positions)
    assert isinstance(result, phx.chemistry.ElectronicEnergyEvaluation)
    assert bool(result.successful)
    assert int(result.header.work.energy_evaluations) == 1
    assert int(result.header.work.force_evaluations) == 0


def test_molecular_provider_refuses_unimplemented_requests_and_numerical_aliases() -> (
    None
):
    structure, provider, calculation = _case()
    for numerical in (
        phx.chemistry.ElectronicNumericalPlan(),
        phx.chemistry.ElectronicNumericalPlan(
            integrals=phx.chemistry.IntegralRepresentationKind.DENSITY_FITTING,
            component_plan_ids=provider.numerical.component_plan_ids,
        ),
    ):
        wrong = phx.chemistry.ElectronicCalculationPlan(
            calculation.system,
            calculation.state,
            calculation.model_chemistry,
            calculation.task,
            numerical=numerical,
        )
        with pytest.raises(
            phx.chemistry.ElectronicCapabilityError, match="numerical definition"
        ):
            wrong.prepare(provider)
    hessian = phx.chemistry.ElectronicCalculationPlan(
        calculation.system,
        calculation.state,
        calculation.model_chemistry,
        phx.chemistry.GroundStateTaskPlan.energy_forces_and_hessian(),
        numerical=provider.numerical,
    )
    with pytest.raises(phx.chemistry.ElectronicCapabilityError, match="hessian"):
        hessian.prepare(provider)
    prepared = calculation.prepare(provider)
    with pytest.raises(phx.chemistry.ElectronicCapabilityError, match="cell vectors"):
        prepared.evaluate(structure.positions, np.eye(3))
    with pytest.raises(ValueError, match="real array"):
        prepared.evaluate(np.asarray(structure.positions, dtype=np.complex128))
    with pytest.raises(ValueError, match="shape"):
        prepared.evaluate(np.zeros((2, 2, 3)))


def test_molecular_provider_refuses_nonrestricted_fractional_and_fixed_factors() -> None:
    _, provider, calculation = _case()
    plan = provider.plan
    uhf = phx.chemistry.MolecularHartreeFockPlan(
        plan.system,
        plan.basis,
        plan.sector,
        phx.chemistry.ElectronicReferenceKind.UNRESTRICTED,
    )
    with pytest.raises(phx.chemistry.ElectronicCapabilityError, match="restricted HF"):
        phx.chemistry.MolecularRHFProvider(uhf, provider.model_chemistry_id)
    fractional = phx.chemistry.MolecularHartreeFockPlan(
        plan.system,
        plan.basis,
        plan.sector,
        plan.reference,
        occupations=phx.chemistry.ElectronicOccupationPlan(
            phx.chemistry.ElectronicOccupationKind.FERMI_DIRAC,
            smearing_energy=0.01,
        ),
    )
    with pytest.raises(
        phx.chemistry.ElectronicCapabilityError, match="integer occupations"
    ):
        phx.chemistry.MolecularRHFProvider(fractional, provider.model_chemistry_id)
    gaussian = phx.operators.quantum.gaussian
    factors = gaussian.FactorizedERITensor(
        np.ones((1, 2, 2)), 1.0, "fixed-workflow-fixture", "density-fitting"
    )
    fixed = phx.chemistry.MolecularHartreeFockPlan(
        plan.system,
        plan.basis,
        plan.sector,
        plan.reference,
        integrals=gaussian.MolecularIntegralPlan(factors),
    )
    with pytest.raises(phx.chemistry.ElectronicCapabilityError, match="fixed factors"):
        phx.chemistry.MolecularRHFProvider(fixed, provider.model_chemistry_id)
    wrong_model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.HartreeFockMethodPlan(),
        basis=phx.chemistry.BasisSetReference("wrong", "not-the-actual-basis"),
    )
    wrong_provider = phx.chemistry.MolecularRHFProvider(
        plan, wrong_model.model_chemistry_id
    )
    wrong_calculation = phx.chemistry.ElectronicCalculationPlan(
        calculation.system,
        calculation.state,
        wrong_model,
        calculation.task,
        numerical=wrong_provider.numerical,
    )
    with pytest.raises(phx.chemistry.ElectronicCapabilityError, match="basis source"):
        wrong_calculation.prepare(wrong_provider)


def test_cuest_numerical_admission_binds_auxiliary_and_fitting_definition(
    tmp_path: Path,
) -> None:
    structure, exact, calculation = _case()
    gaussian = phx.operators.quantum.gaussian
    plan = exact.plan

    def provider_for(
        *,
        exponent: float = 0.8,
        cutoff: float = 1.0e-10,
        relative: bool = True,
        screening: float = 1.0e-12,
        algorithm: Literal["qr", "matrix-power"] = "qr",
    ) -> Any:
        auxiliary = gaussian.GaussianBasisPlan(
            tuple(
                gaussian.GaussianShellPlan(
                    center,
                    0,
                    np.asarray([exponent]),
                    np.asarray([[1.0]]),
                    representation=gaussian.GaussianShellRepresentation.REAL_SPHERICAL,
                    role="auxiliary",
                )
                for center in (11, 17)
            ),
            source_id="workflow-explicit-auxiliary",
            role="auxiliary",
        ).prepare(plan.system)
        source = gaussian.CuestIntegralPlan(
            auxiliary,
            fitting_cutoff=cutoff,
            relative_conditioning=relative,
            fitting_algorithm=algorithm,
            screening_threshold=screening,
            variable_workspace_bytes=1024 * 1024,
            maximum_workspace_bytes=16 * 1024 * 1024,
        )
        df_plan = phx.chemistry.MolecularHartreeFockPlan(
            plan.system,
            plan.basis,
            plan.sector,
            plan.reference,
            integrals=gaussian.MolecularIntegralPlan(source),
        )
        return phx.chemistry.MolecularRHFProvider(df_plan, exact.model_chemistry_id)

    provider = provider_for()
    assert (
        provider.numerical.integrals
        is phx.chemistry.IntegralRepresentationKind.GAUSSIAN_DENSITY_FITTING
    )
    assert provider.capabilities.observables.derivative_orders == (0, 1)
    assert not provider.capabilities.execution.differentiable
    df_calculation = phx.chemistry.ElectronicCalculationPlan(
        calculation.system,
        calculation.state,
        calculation.model_chemistry,
        calculation.task,
        numerical=provider.numerical,
    )
    prepared = df_calculation.prepare(provider)
    assert isinstance(prepared, phx.chemistry.PreparedMolecularRHFCalculation)
    assert df_calculation.calculation_id != calculation.calculation_id
    # The restarted value is an actual native-exact result, not a synthetic
    # cuEST energy. Neither a new source nor a changed fitting policy may
    # relabel it under the same physical model/state/task.
    native_result = calculation.prepare(exact).evaluate(structure.positions)
    path = tmp_path / "native-before-df-cutover"
    phx.chemistry.write_electronic_result_archive(path, native_result, run_id="before-df")
    restored = phx.chemistry.read_electronic_result_archive(path)
    with pytest.raises(ValueError, match="another calculation or numerical definition"):
        phx.chemistry.chemistry_lifecycle(
            df_calculation,
            restored,
            provider_capability_id=provider.capabilities.capabilities_id,
        )
    for other in (
        provider_for(exponent=1.2),
        provider_for(cutoff=1.0e-8),
        provider_for(relative=False),
        provider_for(screening=1.0e-9),
        provider_for(algorithm="matrix-power"),
    ):
        assert other.numerical.numerical_id != provider.numerical.numerical_id
        with pytest.raises(
            phx.chemistry.ElectronicCapabilityError, match="numerical definition"
        ):
            df_calculation.prepare(other)
        changed_calculation = phx.chemistry.ElectronicCalculationPlan(
            calculation.system,
            calculation.state,
            calculation.model_chemistry,
            calculation.task,
            numerical=other.numerical,
        )
        with pytest.raises(
            ValueError, match="another calculation or numerical definition"
        ):
            phx.chemistry.chemistry_lifecycle(
                changed_calculation,
                restored,
                provider_capability_id=other.capabilities.capabilities_id,
            )

    ks_model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.KohnShamMethodPlan(phx.chemistry.DensityFunctionalPlan.pbe()),
        basis=calculation.model_chemistry.basis,
    )
    ks_provider = phx.chemistry.MolecularRHFProvider(
        provider.plan, ks_model.model_chemistry_id
    )
    ks_calculation = phx.chemistry.ElectronicCalculationPlan(
        calculation.system,
        calculation.state,
        ks_model,
        calculation.task,
        numerical=ks_provider.numerical,
    )
    with pytest.raises(phx.chemistry.ElectronicCapabilityError, match="method family"):
        ks_calculation.prepare(ks_provider)
