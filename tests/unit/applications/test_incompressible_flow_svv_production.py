from pathlib import Path
from typing import NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.applications import incompressible_flow as flow
from phydrax.applications.incompressible_flow._production import (
    prepare_constant_power_periodic_method,
)
from phydrax.equations import CompiledIncompressibleSpectralDynamics
from phydrax.solver import (
    _production_runtime,
    PreparedCourantGuardedETDRKMethod,
    StaleRuntimeCheckpointError,
)


def _inputs(
    *, svv: bool = True
) -> tuple[
    CompiledIncompressibleSpectralDynamics,
    PreparedCourantGuardedETDRKMethod,
    flow.PeriodicModalTurbulenceStatisticsPlan,
    flow.PeriodicSpectralProductionCase,
    Array,
]:
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(6) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        tuple(phx.discretization.AxisDomain.periodic(0.0, 2.0 * np.pi) for _ in range(3))
    )
    viscosity = (
        phx.discretization.SpectralVanishingViscosityPlan(
            strength=0.1,
            order=1.0,
            cutoff_scale=0.5,
            cutoff_exponent=0.25,
            kernel="tadmor_step",
        )
        if svv
        else None
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(3, 0.05),
        space,
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.PaddingDealiasingPlan(2)
        ),
        spectral_viscosity=viscosity,
    )
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space, component_shape=(3,)
    )
    method = phx.solver.CourantGuardedETDRKMethod(
        phx.solver.ETDRKMethod(4),
        maximum_courant_number=0.5,
    ).prepare(dynamics, coordinates=coordinates)
    basis = flow.SolenoidalHermitianFourierBasis(
        dynamics.projector, maximum_wavenumber=2.1
    )
    initial = basis.evaluate(jnp.linspace(0.2, 0.8, basis.coordinate_size))
    statistics = flow.PeriodicModalTurbulenceStatisticsPlan(
        dynamics, np.linspace(0.0, 4.0, 5)
    )
    case = flow.PeriodicSpectralProductionCase(
        dynamics, initial, case_id="svv-production"
    )
    return dynamics, method, statistics, case, initial


@pytest.mark.parametrize("svv", [False, True])
def test_svv_statistics_formula_and_validity(svv: bool) -> None:
    dynamics, _, plan, _, velocity = _inputs(svv=svv)
    result = plan.evaluate(0.0, velocity)
    if dynamics.spectral_viscosity is None:
        expected = 0.0
        assert result.spectral_viscosity_id is None
    else:
        expected = -np.sum(
            np.asarray(dynamics.spectral_viscosity.decay_rate)[..., None]
            * np.abs(np.asarray(velocity)) ** 2
        )
        assert result.spectral_viscosity_id == dynamics.spectral_viscosity.prepared_id
    np.testing.assert_allclose(
        result.spectral_viscosity_dissipation, expected, atol=1e-12
    )
    np.testing.assert_allclose(
        result.spectral_viscosity_dissipation_shells.integral.sum(), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        result.mean_spectral_viscosity_dissipation, expected / plan.volume, atol=1e-12
    )
    denominator = result.molecular_dissipation + expected - result.sgs_energy_rate
    np.testing.assert_allclose(
        result.numerical_dissipation_fraction, expected / denominator, atol=1e-12
    )
    assert bool(result.numerical_dissipation_fraction_valid)
    assert bool(result.spectral_viscosity_available) == svv
    assert bool(result.kolmogorov_scale_valid) == (not svv)
    assert bool(result.taylor_microscale_valid) == (not svv)
    zero = plan.evaluate(0.0, jnp.zeros_like(velocity))
    assert float(zero.numerical_dissipation_fraction) == 0.0
    assert not bool(zero.numerical_dissipation_fraction_valid)


def test_svv_stage_rate_consistency() -> None:
    dynamics, _, plan, _, velocity = _inputs()
    stage = dynamics.stage(jnp.asarray(0.0), velocity, None)
    missing = eqx.tree_at(lambda item: item.rates.spectral_viscosity_rate, stage, None)
    with pytest.raises(ValueError, match="require a spectral-viscosity rate"):
        plan.evaluate(0.0, velocity, stage=missing)
    no_svv, _, no_plan, _, _ = _inputs(svv=False)
    foreign = eqx.tree_at(
        lambda item: item.rates.spectral_viscosity_rate,
        no_svv.stage(jnp.asarray(0.0), velocity, None),
        jnp.zeros_like(velocity),
        is_leaf=lambda value: value is None,
    )
    with pytest.raises(ValueError, match="no-SVV"):
        no_plan.evaluate(0.0, velocity, stage=foreign)


@pytest.mark.parametrize("accepted", [False, True])
def test_courant_svv_production_preserves_guard(tmp_path: Path, accepted: bool) -> None:
    dynamics, method, statistics, case, velocity = _inputs()
    restriction = method.step_restriction(velocity)
    step = float(restriction.maximum_step) * (0.1 if accepted else 2.0)
    plan = flow.PeriodicSpectralProductionPlan(
        dynamics,
        method,
        statistics,
        case,
        start_time=0.0,
        end_time=step,
        step_size=step,
        checkpoint_interval=1,
    )
    assert plan.statistics_evaluator.value_size == 82
    assert plan.statistics_evaluator(jnp.asarray(0.0), velocity, None).shape == (82,)
    prepared = plan.prepare(tmp_path / "courant-svv")
    initial = prepared.initialize(velocity)
    following, transition = prepared.step(initial)
    assert bool(transition.successful) == accepted
    if accepted:
        assert bool(
            prepared.statistics_snapshot(
                following.time, following.accepted_state
            ).successful
        )
    else:
        np.testing.assert_array_equal(following.accepted_state, velocity)
        np.testing.assert_array_equal(
            following.moment_states[0].mean, initial.moment_states[0].mean
        )


def test_courant_forcing_compositions_are_refused() -> None:
    dynamics, method, statistics, case, velocity = _inputs()
    forcing = flow.ConstantPowerFourierForcingPlan(
        dynamics.projector,
        maximum_wavenumber=1.1,
        power_input=0.1,
        minimum_forced_energy=1e-10,
    )
    with pytest.raises(TypeError, match="Courant-guarded"):
        prepare_constant_power_periodic_method(method, forcing)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Courant-guarded"):
        flow.PeriodicSpectralProductionPlan(
            dynamics,
            method,
            statistics,
            case,
            start_time=0.0,
            end_time=0.01,
            step_size=0.01,
            checkpoint_interval=1,
            constant_power_forcing=forcing,
            constant_power_wiring="adapter",
        )
    basis = flow.SolenoidalHermitianFourierBasis(
        dynamics.projector, maximum_wavenumber=1.1
    )
    ou = flow.SolenoidalOUForcingPlan(basis, correlation_time=0.7, rms_acceleration=0.2)
    realization = phx.stochastic.OrnsteinUhlenbeckRealization(
        jax.random.key(29), (basis.coordinate_size,), support=(0.0, 1.0), tolerance=1e-6
    )
    with pytest.raises(TypeError, match="Courant-guarded"):
        flow.prepare_ou_forced_periodic_method(method, ou, realization)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Courant-guarded"):
        flow.PreparedOUForcedETDRKMethod(method, ou, realization)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Courant-guarded"):
        flow.PeriodicSpectralProductionPlan(
            dynamics,
            method,
            statistics,
            case,
            start_time=0.0,
            end_time=0.01,
            step_size=0.01,
            checkpoint_interval=1,
            ou_forcing=ou,
            ou_realization=realization,
        )
    plain = flow.prepare_ou_forced_periodic_method(method.base_method, ou, realization)
    assert plain.etdrk is method.base_method
    assert velocity.shape == dynamics.state_shape


def test_periodic_checkpoint_refuses_changed_statistics_before_array_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dynamics, method, statistics, case, velocity = _inputs()
    changed_statistics = flow.PeriodicModalTurbulenceStatisticsPlan(
        dynamics, np.linspace(0.0, 4.0, 6)
    )

    def production(
        layout: flow.PeriodicModalTurbulenceStatisticsPlan,
    ) -> flow.PeriodicSpectralProductionPlan:
        return flow.PeriodicSpectralProductionPlan(
            dynamics,
            method,
            layout,
            case,
            start_time=0.0,
            end_time=0.001,
            step_size=0.001,
            checkpoint_interval=1,
        )

    original = production(statistics)
    changed = production(changed_statistics)
    assert original.statistics_evaluator.value_size == 82
    assert changed.statistics_evaluator.value_size == 89
    assert original.statistics_evaluator.evaluator_id != (
        changed.statistics_evaluator.evaluator_id
    )
    old_inventory = original.manifest.inventory
    new_inventory = changed.manifest.inventory
    assert old_inventory is not None and new_inventory is not None
    assert old_inventory.changed_roles(new_inventory) == ("program",)
    path = tmp_path / "statistics-layout"
    prepared = original.prepare(path)
    following, transition = prepared.step(prepared.initialize(velocity))
    assert bool(transition.successful)
    checkpointed = prepared.checkpoint(following)
    same_layout = original.prepare(path)
    resumed = same_layout.resume(same_layout.initialize(velocity))
    coordinates = method.base_method.coordinates
    assert coordinates is not None
    np.testing.assert_array_equal(
        coordinates.to_real_coordinates(resumed.accepted_state),
        coordinates.to_real_coordinates(following.accepted_state),
    )
    assert resumed.last_checkpoint_id == checkpointed.last_checkpoint_id
    foreign = changed.prepare(path)
    template = foreign.initialize(velocity)

    def refuse_array_restore(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Checkpoint arrays must not be decoded on stale identity.")

    monkeypatch.setattr(
        _production_runtime, "read_runtime_checkpoint", refuse_array_restore
    )
    with pytest.raises(StaleRuntimeCheckpointError) as refusal:
        foreign.resume(template)
    assert refusal.value.roles == ("program",)


def test_periodic_statistics_vector_matches_independent_slot_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamics, method, statistics, case, velocity = _inputs()
    plan = flow.PeriodicSpectralProductionPlan(
        dynamics,
        method,
        statistics,
        case,
        start_time=0.0,
        end_time=0.001,
        step_size=0.001,
        checkpoint_interval=1,
    )
    evaluator = plan.statistics_evaluator
    snapshot = evaluator.snapshot(0.0, velocity, None)
    # This table is the consumer layout, independent of evaluator packing.
    shell_fields = (
        "energy_shells",
        "molecular_dissipation_shells",
        "spectral_viscosity_dissipation_shells",
        "advective_transfer_shells",
        "sgs_transfer_shells",
        "forcing_injection_shells",
    )
    scalar_fields = (
        "kinetic_energy",
        "mean_kinetic_energy",
        "molecular_dissipation",
        "mean_molecular_dissipation",
        "spectral_viscosity_dissipation",
        "mean_spectral_viscosity_dissipation",
        "numerical_dissipation_fraction",
        "advective_energy_rate",
        "mean_advective_energy_rate",
        "sgs_energy_rate",
        "mean_sgs_energy_rate",
        "forcing_power",
        "mean_forcing_power",
        "enstrophy",
        "mean_enstrophy",
        "helicity",
        "mean_helicity",
        "taylor_microscale",
        "kolmogorov_scale",
        "kmax_kolmogorov",
        "integral_scale",
        "energy_tail_fraction",
        "molecular_dissipation_tail_fraction",
        "divergence_norm",
        "velocity_reality_defect",
        "sgs_modeled_dissipation",
        "sgs_energy_identity_defect",
        "sgs_projection_energy_defect",
        "sgs_regularization_activity_count",
        "sgs_dynamic_coefficient_minimum",
        "sgs_dynamic_coefficient_mean",
        "sgs_dynamic_coefficient_maximum",
        "sgs_backscatter_activity_count",
        "sgs_backscatter_limit_count",
        "sgs_accepted_update_count",
        "sgs_rejected_update_count",
        "sgs_maximum_kinematic_viscosity",
        "sgs_advective_step_limit",
        "sgs_diffusive_step_limit",
        "sgs_combined_step_limit",
        "sgs_etdrk_step_limit",
        "sgs_fully_explicit_step_limit",
        "sgs_available",
        "sgs_regularization_available",
        "sgs_stability_available",
        "forcing_available",
        "helicity_valid",
        "taylor_microscale_valid",
        "kolmogorov_scale_valid",
        "integral_scale_valid",
        "energy_tail_valid",
        "molecular_dissipation_tail_valid",
        "finite",
        "successful",
    )
    assert len(scalar_fields) == 54

    def oracle(record: flow.PeriodicModalTurbulenceStatistics) -> np.ndarray:
        shell_values = [
            np.asarray(getattr(record, field).integral) for field in shell_fields
        ]
        shell_values.append(np.asarray(record.resolved_spectral_flux))
        scalar_values = []
        for field in scalar_fields:
            value = np.asarray(getattr(record, field), dtype=np.float64)
            if field.endswith("step_limit") and not np.isfinite(value):
                value = np.asarray(0.0, dtype=np.float64)
            scalar_values.append(value)
        return np.concatenate((*shell_values, np.asarray(scalar_values)))

    np.testing.assert_array_equal(
        evaluator(jnp.asarray(0.0), velocity, None), oracle(snapshot)
    )
    # Label every packed slot uniquely so physically equal zeros/flags cannot
    # hide a permutation. Only packing sees this tagged snapshot.
    leaves = (
        tuple(getattr(snapshot, field).integral for field in shell_fields)
        + (snapshot.resolved_spectral_flux,)
        + tuple(getattr(snapshot, field) for field in scalar_fields)
    )
    replacements = tuple(
        jnp.arange(leaf.size, dtype=jnp.float64).reshape(leaf.shape) + 100.0 * (index + 1)
        for index, leaf in enumerate(leaves)
    )
    tagged = eqx.tree_at(
        lambda record: (
            tuple(getattr(record, field).integral for field in shell_fields)
            + (record.resolved_spectral_flux,)
            + tuple(getattr(record, field) for field in scalar_fields)
        ),
        snapshot,
        replacements,
    )

    def tagged_snapshot(
        self: object, time: object, state: object, args: object, /
    ) -> flow.PeriodicModalTurbulenceStatistics:
        return tagged

    monkeypatch.setattr(type(evaluator), "snapshot", tagged_snapshot)
    np.testing.assert_array_equal(
        evaluator(jnp.asarray(0.0), velocity, None), oracle(tagged)
    )
