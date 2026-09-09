"""Observable inventory, transfer, reference and native artifact contracts."""

from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.atmosphere._interactive_column import InteractiveMoistColumnPlan
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.geophysics._flux_closure import (
    admit_column_flux,
    column_flux_space,
    column_flux_tasks,
    ColumnFluxBinding,
    conditional_column_flux_target,
    ConservativeColumnTransfer,
    deploy_column_flux,
)


def _binding(mass=(1.0, 3.0, 8.0), interval=2.0):
    thermo = MoistThermodynamicPlan()
    tasks = column_flux_tasks(
        "flux-regression",
        thermo,
        training_resolutions_m=(20.0,),
        training_intervals_s=(2.0,),
        training_regimes=("test",),
        training_forcing_ids=("none",),
    )
    field, measure = column_flux_space(mass, "test-column")
    return ColumnFluxBinding(
        tasks, field, measure, interval, 20.0, "test", "none"
    ), thermo


def _energy(thermo, dry, vapor, liquid=0.0, ice=0.0):
    ed, ev, el, ei = thermo.phase_energies(285.0)
    return (
        jnp.asarray(dry) * ed
        + jnp.asarray(vapor) * ev
        + jnp.asarray(liquid) * el
        + jnp.asarray(ice) * ei
    )


def test_paired_flux_cancels_inventory_on_nonuniform_layers_and_scales_with_dt():
    binding, thermo = _binding()
    dry = binding.measure.weights
    vapor = jnp.asarray([0.04, 0.12, 0.32])
    energy = _energy(thermo, dry, vapor)
    wf, ef = jnp.asarray([0.004, -0.006]), jnp.asarray([9000.0, -13000.0])
    result = admit_column_flux(
        binding,
        vapor,
        energy,
        wf,
        ef,
        thermodynamics=thermo,
        dry_mass=dry,
        artifact_ids=("vapor-test", "energy-test"),
    )
    water, heat = result.require_inventories()
    np.testing.assert_allclose(result.vapor_increment, [-0.008, 0.020, -0.012], rtol=1e-6)
    np.testing.assert_allclose(result.energy_increment, [-18000.0, 44000.0, -26000.0])
    np.testing.assert_allclose(jnp.sum(water), jnp.sum(vapor), rtol=1e-6)
    np.testing.assert_allclose(jnp.sum(heat), jnp.sum(energy), rtol=1e-6)
    np.testing.assert_allclose(result.inventory_residual, 0, atol=1e-7)
    longer = admit_column_flux(
        replace(binding, interval_seconds=4.0),
        vapor,
        energy,
        wf,
        ef,
        thermodynamics=thermo,
        dry_mass=dry,
        artifact_ids=("vapor-test", "energy-test"),
    )
    np.testing.assert_allclose(longer.vapor_increment, 2 * result.vapor_increment)
    assert longer.binding_id != result.binding_id


def test_vapor_donor_cannot_borrow_liquid_or_throughflow_and_native_step_rejects():
    binding, thermo = _binding()
    dry, vapor, liquid = (
        binding.measure.weights,
        jnp.asarray([0.02, 0.001, 0.02]),
        jnp.asarray([0.0, 2.0, 0.0]),
    )
    result = admit_column_flux(
        binding,
        vapor,
        _energy(thermo, dry, vapor, liquid),
        jnp.asarray([0.005, 0.005]),
        jnp.zeros(2),
        thermodynamics=thermo,
        dry_mass=dry,
        liquid_mass=liquid,
        artifact_ids=("vapor-test", "energy-test"),
    )
    # Net middle-layer tendency is zero but its outgoing donor flux exhausts vapor.
    assert not bool(result.admitted)
    with pytest.raises(ValueError):
        result.require_inventories()
    # Condensate loading destabilizes virtual theta: disable the independent
    # convective closure so this test isolates vapor-hook donor admission.
    plan = InteractiveMoistColumnPlan(
        thermodynamics=thermo,
        rain_fall_speed=0.0,
        snow_fall_speed=0.0,
        mixing_length=0.0,
    )
    state = plan.initialize(
        dry, vapor, 285.0, dry, cloud_liquid_mass=liquid, rain_mass=1.0
    )
    assert bool(plan.step(state, 2.0).successful)
    rejected = plan.step(
        state, 2.0, water_flux=jnp.asarray([0.005, 0.005]), energy_flux=jnp.zeros(2)
    )
    assert not bool(rejected.successful)
    np.testing.assert_array_equal(rejected.state.vapor_mass, state.vapor_mass)
    np.testing.assert_array_equal(rejected.state.rain_mass, state.rain_mass)
    assert float(rejected.state.time) == float(state.time)


def test_measure_units_reference_and_runtime_interval_are_bound():
    binding, thermo = _binding()
    dry, vapor = binding.measure.weights, jnp.asarray([[0.02, 0.04, 0.08]])
    energy = _energy(thermo, dry, vapor)
    with pytest.raises(ValueError):
        replace(binding, measure_unit=phx.units.KILOGRAM)
    with pytest.raises(ValueError):
        admit_column_flux(
            binding,
            vapor,
            energy,
            jnp.zeros((1, 2)),
            jnp.zeros((1, 2)),
            thermodynamics=MoistThermodynamicPlan(reference_temperature=274.0),
            dry_mass=dry,
            artifact_ids=("v", "e"),
        )
    with pytest.raises(ValueError):
        admit_column_flux(
            binding,
            vapor,
            energy,
            jnp.zeros((1, 2)),
            jnp.zeros((1, 2)),
            thermodynamics=thermo,
            dry_mass=jnp.ones(3),
            artifact_ids=("v", "e"),
        )
    batch = binding.batch(vapor, energy, forcing=np.zeros(1))
    with pytest.raises(ValueError):
        replace(binding, interval_seconds=3.0).validate_batch(batch)
    with pytest.raises(ValueError):
        replace(binding, resolution_m=10.0).validate_batch(batch)


def _nonuniform_transfer(thermo, *, bad=False):
    source, sm = column_flux_space([1.0, 3.0, 2.0, 6.0], "fine")
    target, tm = column_flux_space([4.0, 8.0], "coarse")
    matrix = (
        jnp.asarray([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
        if bad
        else jnp.asarray([[0.25, 0.75, 0.0, 0.0], [0.0, 0.0, 0.25, 0.75]])
    )
    transfer = phx.discretization.FieldTransfer(
        source,
        target,
        phx.linalg.DenseLinearOperator(
            matrix, source=source.vector_space, target=target.vector_space
        ),
        properties=phx.discretization.TransferProperties(
            conservative=True, constant_preserving=True, positivity_preserving=True
        ),
    )
    return ConservativeColumnTransfer(transfer, sm, tm, thermo.plan_id)


def test_native_measure_restriction_preserves_extensives_and_separates_numerics():
    original_binding, thermo = _binding()
    transfer = _nonuniform_transfer(thermo)
    binding = replace(
        original_binding, field=transfer.transfer.target, measure=transfer.target_measure
    )
    fine = jnp.asarray([[[0.01, 100.0], [0.09, 300.0], [0.02, 200.0], [0.18, 600.0]]])
    coarse = transfer.inventories(fine)
    np.testing.assert_allclose(coarse, [[[0.1, 400.0], [0.2, 800.0]]])
    physical = jnp.asarray([[[-0.004, -20.0], [0.0, 0.0], [0.004, 20.0], [0.0, 0.0]]])
    numerical = 0.5 * physical
    target = conditional_column_flux_target(
        binding,
        transfer,
        fine,
        fine + physical + numerical,
        coarse,
        coarse,
        interval_bounds=[[0.0, 2.0]],
        fine_reference_after=fine + physical,
        coarse_reference_after=coarse,
        reference_id="known-reference",
    )
    np.testing.assert_allclose(target.flux, [[[0.002, 10.0]]], rtol=2e-6)
    np.testing.assert_allclose(
        target.numerical_increment, [[[-0.002, -10.0], [0.002, 10.0]]], rtol=1e-5
    )
    with pytest.raises(ValueError):
        _nonuniform_transfer(thermo, bad=True)
    with pytest.raises(ValueError):
        conditional_column_flux_target(
            binding,
            transfer,
            fine,
            fine + 1.0,
            coarse,
            coarse,
            interval_bounds=[[0.0, 2.0]],
        )
    with pytest.raises(ValueError):
        conditional_column_flux_target(
            binding, transfer, fine, fine, coarse, coarse, interval_bounds=[[0.0, 3.0]]
        )


def test_equal_total_distinct_cases_cannot_broadcast_one_fine_endpoint():
    original, thermo = _binding()
    transfer = _nonuniform_transfer(thermo)
    binding = replace(
        original, field=transfer.transfer.target, measure=transfer.target_measure
    )
    fine = jnp.asarray(
        [
            [[0.01, 100.0], [0.09, 300.0], [0.02, 200.0], [0.18, 600.0]],
            [[0.02, 200.0], [0.18, 600.0], [0.01, 100.0], [0.09, 300.0]],
        ]
    )
    change = jnp.asarray(
        [
            [[-0.004, -20.0], [0.0, 0.0], [0.004, 20.0], [0.0, 0.0]],
            [[0.006, 30.0], [0.0, 0.0], [-0.006, -30.0], [0.0, 0.0]],
        ]
    )
    final, coarse = fine + change, transfer.inventories(fine)
    bounds = [[0.0, 2.0], [0.0, 2.0]]
    # Both cases have identical global inventories. A budget check alone
    # cannot detect wrongly broadcasting the first final profile to the second.
    valid = conditional_column_flux_target(
        binding,
        transfer,
        fine,
        final,
        coarse,
        coarse,
        interval_bounds=bounds,
        fine_reference_after=final,
        coarse_reference_after=coarse,
        reference_id="casewise-reference",
    )
    np.testing.assert_allclose(
        valid.flux, [[[0.002, 10.0]], [[-0.003, -15.0]]], rtol=1e-5
    )
    with pytest.raises(ValueError):
        conditional_column_flux_target(
            binding,
            transfer,
            fine,
            final[:1],
            coarse,
            coarse,
            interval_bounds=bounds,
        )
    with pytest.raises(ValueError):
        conditional_column_flux_target(
            binding,
            transfer,
            fine,
            final,
            coarse,
            coarse,
            interval_bounds=bounds,
            fine_reference_after=final[:1],
            coarse_reference_after=coarse,
            reference_id="casewise-reference",
        )


def test_native_artifact_reload_preserves_flux_and_rejects_wrong_restart_identity(
    tmp_path,
):
    binding, thermo = _binding()
    dry = binding.measure.weights
    vapor = jnp.asarray([[0.02, 0.04, 0.08]])
    energy = _energy(thermo, dry, vapor)
    batch = binding.batch(vapor, energy, forcing=np.zeros(1))
    native, op = phx.nn.operator.training, phx.nn.operator
    artifacts = []
    for index, task in enumerate(binding.tasks):
        model = op.architectures.DeepONet(
            branch={
                "vapor_per_dry_mass": phx.nn.models.MLP(
                    in_size=3, out_size=4, width_size=4, depth=1, key=jr.key(index)
                )
            },
            trunk=phx.nn.models.MLP(
                in_size=1, out_size=4, width_size=4, depth=1, key=jr.key(index + 2)
            ),
            latent_size=4,
            coord_dim=1,
        )
        # Explicitly untrained weights isolate artifact/restart behavior from optimizer behavior.
        original = native.TrainedOperator(
            model,
            task,
            training_evidence=op.OperatorTrainingEvidence("task_specific"),
            output_field_map={"output": task.fields[-1].name},
            artifact_id=f"artifact-test-{index}",
            provenance={"untrained_baseline": True},
        )
        native.save_operator_artifact(tmp_path / str(index), original)
        restored = native.load_trained_operator(tmp_path / str(index))
        name = task.fields[-1].name
        np.testing.assert_array_equal(
            original.predict(batch).field(name).values,
            restored.predict(batch).field(name).values,
        )
        artifacts.append(restored)
    with pytest.raises(ValueError):
        deploy_column_flux(
            tuple(artifacts),
            binding,
            batch,
            vapor_mass=vapor,
            total_energy=energy,
            thermodynamics=thermo,
            dry_mass=dry,
            artifact_ids=("wrong-resume-artifact", "artifact-test-1"),
        )
    changed_binding = replace(
        binding,
        tasks=column_flux_tasks(
            "different-training-data",
            thermo,
            training_resolutions_m=(20.0,),
            training_intervals_s=(2.0,),
            training_regimes=("test",),
            training_forcing_ids=("none",),
        ),
    )
    with pytest.raises(ValueError):
        deploy_column_flux(
            tuple(artifacts),
            changed_binding,
            batch,
            vapor_mass=vapor,
            total_energy=energy,
            thermodynamics=thermo,
            dry_mass=dry,
            artifact_ids=tuple(model.artifact_id for model in artifacts),
        )
