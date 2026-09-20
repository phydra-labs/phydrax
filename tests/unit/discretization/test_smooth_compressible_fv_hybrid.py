#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._model import AbstractArrayModel
from phydrax.closure_data._dataset import NormalizerProvenance, TrainOnlyNormalizer
from phydrax.closure_data._kinetic_equilibrium import (
    energy_equilibrium_numeric_revision,
    EnergyEquilibriumSupportEnvelope,
    LearnedEnergyEquilibriumBindingPlan,
)
from phydrax.closure_data._state import FlowStateSchema
from phydrax.discretization._axis import TensorGridPlan, UniformCellAxisSpec
from phydrax.discretization.discrete_velocity._energy_equilibrium import (
    PositiveEnergyEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._hybrid import (
    FixedConformingFVKineticInterfacePlan,
)
from phydrax.discretization.discrete_velocity._hybrid_runtime import (
    DynamicHybridCompositeState,
    DynamicHybridOwnershipPlan,
    DynamicHybridOwnershipState,
    FixedHybridStageEvidence,
    FixedPartitionHybridState,
    PreparedFixedPartitionHybridRuntime,
)
from phydrax.discretization.discrete_velocity._quadrature import d2v17_quadrature
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
)
from phydrax.discretization.finite_volume import (
    FiniteVolumeBoundaryPair,
    FiniteVolumeBoundarySet,
    FiniteVolumeMethodPlan,
    FiniteVolumePlan,
    FluxPositivityPlan,
    HLLCFluxPlan,
    MUSCLReconstruction,
    SupersonicOutflowBoundary,
)
from phydrax.equations._conservation import (
    compile_conservation_problem,
    ConservationProblemIR,
)
from phydrax.equations._hyperbolic_systems import EulerSystem
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.solver._finite_volume_runtime import (
    FiniteVolumeStepPolicy,
    PreparedFiniteVolumeRuntime,
)


class _ZeroDualModel(AbstractArrayModel):
    weight: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.weight = jnp.zeros((2, 4), dtype=jnp.float64)
        self.in_size = 4
        self.out_size = 2

    def __call__(self, values, /, *, key=None):
        del key
        return self.weight @ values


def _material():
    return IdealGasMaterial(1.4, 1.0)


def _conserved():
    return jnp.asarray((1.0, 0.03, -0.02, 1.251), dtype=jnp.float64)


def _learned_binding(method, *, velocity_bound=0.5):
    equilibrium_plan = PositiveEnergyEquilibriumPlan(method.quadrature)
    schema = FlowStateSchema(
        ("density", "momentum_x", "momentum_y", "total_energy"),
        ("1", "1", "1", "1"),
        (1.0, 1.0, 1.0, 1.0),
        density_name="density",
        total_energy_name="total_energy",
    )
    provenance = NormalizerProvenance(
        partition_id="hybrid-test-partition",
        training_assignment_ids=("assignment",),
        training_sample_ids=("sample",),
        feature_name="conserved-state",
        schema_id=schema.schema_id,
    )
    normalizer = TrainOnlyNormalizer(
        _conserved(),
        jnp.ones((4,), dtype=jnp.float64),
        provenance,
        epsilon=1.0e-12,
    )
    training_id = "hybrid-test-training"
    support = EnergyEquilibriumSupportEnvelope(
        rho_bounds=(0.5, 2.0),
        u_x_bounds=(-velocity_bound, velocity_bound),
        u_y_bounds=(-velocity_bound, velocity_bound),
        temperature_bounds=(0.1, 2.0),
        maximum_mach=1.0,
        minimum_hull_margin=1.0e-8,
        minimum_particle_equilibrium_margin=0.0,
        schema_id=schema.schema_id,
        material_id=method.material.material_id,
        normalizer_id=normalizer.normalizer_id,
        quadrature_id=method.quadrature.quadrature_id,
        equilibrium_plan_id=equilibrium_plan.plan_id,
        training_preparation_id=training_id,
    )
    plan = LearnedEnergyEquilibriumBindingPlan(
        equilibrium_plan,
        schema,
        method.material,
        normalizer,
        support,
        input_component_names=schema.component_names,
        semantic_id="hybrid-test-learned-dual",
        training_preparation_id=training_id,
    )
    model = _ZeroDualModel()
    return plan.prepare(
        model,
        energy_equilibrium_numeric_revision(plan.semantic_id, model),
    )


def _kinetic_method():
    return SmoothCompressibleD2VKineticMethod(
        d2v17_quadrature(),
        _material(),
        ConstantTransport(0.03, 0.04),
    )


def _spatial(method):
    energy = PositiveEnergyEquilibriumPlan(method.quadrature)
    transport = D2V17PeriodicTransportPlan(
        method.quadrature,
        (5, 5),
        (0.01, 0.01),
        0.01,
    )
    return PreparedSmoothCompressibleD2V17SpatialDynamics(
        method,
        energy,
        transport,
        conservation_tolerance=1.0e-10,
    )


def _kinetic_state(spatial, conserved=None):
    value = _conserved() if conserved is None else jnp.asarray(conserved)
    density = value[0]
    velocity = value[1:3] / density
    pressure = spatial.method.material.pressure(
        density,
        (value[-1] - 0.5 * jnp.sum(value[1:3] * velocity)) / density,
    )
    target_flux = (value[-1] + pressure) * velocity
    oracle = spatial.energy_plan.solve(value[-1], target_flux)
    equilibrium, evidence = spatial.method.equilibrium_from_energy_dual_with_evidence(
        value,
        oracle.dual,
        spatial.energy_plan,
    )
    assert bool(evidence.successful)
    shape = spatial.transport.spatial_shape + (
        spatial.method.quadrature.population_count,
    )
    return SmoothCompressibleKineticState(
        jnp.broadcast_to(equilibrium.particle_populations, shape),
        jnp.broadcast_to(equilibrium.total_energy_populations, shape),
    )


def _finite_volume_runtime(method, *, retries=0):
    grid = TensorGridPlan(
        (UniformCellAxisSpec(5), UniformCellAxisSpec(5)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (0.05, 0.05))))
    system = EulerSystem(2, material=method.material)
    discretization = FiniteVolumePlan(
        grid,
        component_names=system.component_names,
    ).prepare()
    outflow = FiniteVolumeBoundaryPair(
        SupersonicOutflowBoundary(),
        SupersonicOutflowBoundary(),
    )
    problem = ConservationProblemIR(
        "fixed-hybrid-test",
        "state",
        system,
        FiniteVolumeBoundarySet(("x", "y"), (outflow, outflow)),
    )
    scheme = FiniteVolumeMethodPlan(
        MUSCLReconstruction(),
        HLLCFluxPlan(),
    )
    dynamics = compile_conservation_problem(problem, discretization, scheme).dynamics
    return PreparedFiniteVolumeRuntime(
        dynamics,
        FluxPositivityPlan(),
        FiniteVolumeStepPolicy(cfl=20.0, maximum_retries=retries),
    )


def _fixed_runtime(*, retries=0, velocity_bound=0.5):
    method = _kinetic_method()
    spatial = _spatial(method)
    finite_volume = _finite_volume_runtime(method, retries=retries)
    learned = _learned_binding(method, velocity_bound=velocity_bound)
    upper_interfaces = tuple(
        FixedConformingFVKineticInterfacePlan(
            method,
            finite_volume.dynamics.system,
            jnp.asarray((1.0, 0.0)),
            f"upper-x-{y_index}",
        )
        for y_index in range(5)
    )
    lower_interfaces = tuple(
        FixedConformingFVKineticInterfacePlan(
            method,
            finite_volume.dynamics.system,
            jnp.asarray((-1.0, 0.0)),
            f"lower-x-{y_index}",
        )
        for y_index in range(5)
    )
    interfaces = (*upper_interfaces, *lower_interfaces)
    return PreparedFixedPartitionHybridRuntime(
        finite_volume,
        spatial,
        learned,
        interfaces,
        (0,) * len(interfaces),
        (
            *((5, y_index) for y_index in range(5)),
            *((0, y_index) for y_index in range(5)),
        ),
        (
            *((4, y_index) for y_index in range(5)),
            *((0, y_index) for y_index in range(5)),
        ),
        (
            *((0, y_index) for y_index in range(5)),
            *((4, y_index) for y_index in range(5)),
        ),
        conservation_tolerance=2.0e-10,
    )


def _fixed_state(runtime):
    average = jnp.broadcast_to(_conserved(), (5, 5, 4))
    finite_volume = runtime.finite_volume.initialize_state(
        average,
        0.0,
        runtime.required_step_size,
    )
    return runtime.initialize_state(finite_volume, _kinetic_state(runtime.spatial))


def _assert_same_tree(left, right):
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_value, right_value in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(left_value, right_value)


def test_fixed_runtime_uses_stage_weighted_equal_opposite_population_flux_once():
    runtime = _fixed_runtime()
    state = _fixed_state(runtime)

    result = runtime.advance(state, jnp.asarray(0.01))

    assert bool(result.evidence.accepted)
    assert result.evidence.shock_owner == "finite_volume"
    assert result.evidence.ownership_differentiability == "none"
    stages = tuple(stage.evidence for stage in result.stage_flux_trace.stages)
    assert all(isinstance(stage, FixedHybridStageEvidence) for stage in stages)
    assert all(len(stage.common_fluxes) == len(runtime.interfaces) for stage in stages)
    assert all(
        flux.learned_lift_evidence is not None
        for stage in stages
        for flux in stage.common_fluxes
    )
    assert not np.allclose(
        stages[0].common_fluxes[0].common_conservative_flux,
        stages[2].common_fluxes[0].common_conservative_flux,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.evidence.audit.kinetic_outer_boundary_exchange,
        0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        result.evidence.audit.kinetic_moment_exchange_residual,
        0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        result.evidence.audit.finite_volume_interface_flux_residual,
        0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        result.evidence.audit.global_conservation_residual,
        0.0,
        atol=2.0e-10,
    )
    np.testing.assert_array_equal(
        result.candidate.finite_volume.cell_average(),
        result.finite_volume.attempted.runtime_state.cell_average(),
    )


def test_fixed_runtime_refuses_wrong_dt_and_any_fv_retry_policy():
    with pytest.raises(ValueError, match="forbids finite-volume retries"):
        _fixed_runtime(retries=1)
    runtime = _fixed_runtime()
    state = _fixed_state(runtime)

    with pytest.raises(ValueError, match="exact fixed lattice step"):
        runtime.advance(state, jnp.asarray(0.005))
    wrong_fv = eqx.tree_at(
        lambda value: value.step_size,
        state.finite_volume,
        jnp.asarray(0.005),
    )
    with pytest.raises(ValueError, match="exact fixed lattice step"):
        runtime.advance(
            FixedPartitionHybridState(wrong_fv, state.kinetic), jnp.asarray(0.01)
        )


def test_failed_learned_support_rolls_back_both_sides_and_checkpoints_only_commit():
    runtime = _fixed_runtime(velocity_bound=0.001)
    state = _fixed_state(runtime)

    result = runtime.advance(state, jnp.asarray(0.01))

    assert not bool(result.evidence.accepted)
    assert bool(result.evidence.rollback_applied)
    assert not bool(result.evidence.learned_lifts_accepted)
    _assert_same_tree(result.runtime_state, state)
    with pytest.raises(ValueError, match="Only jointly accepted"):
        runtime.checkpoint(result.candidate, "rejected-candidate")
    checkpoint = runtime.checkpoint(result.runtime_state, "accepted-after-rejection")
    _assert_same_tree(runtime.restore(checkpoint), state)
    foreign = _fixed_runtime(velocity_bound=0.002)
    with pytest.raises(ValueError, match="artifact is incompatible"):
        foreign.restore(checkpoint)


def test_finite_volume_runtime_without_provider_retains_existing_ssprk_result():
    method = _kinetic_method()
    runtime = _finite_volume_runtime(method)
    average = jnp.broadcast_to(_conserved(), (5, 5, 4))
    state = runtime.initialize_state(average, 0.0, 0.001)

    result = runtime.advance(state)
    direct = runtime._candidate(
        state.time,
        average,
        result.accepted_step_size,
        None,
    )

    assert result.stage_flux_trace is None
    np.testing.assert_allclose(
        result.runtime_state.cell_average(),
        direct.state.reshape(result.runtime_state.cell_average().shape),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def _dynamic_state(plan, owned, *, finite_volume=None, kinetic=None):
    ownership = plan.initialize(owned)
    fv = jnp.broadcast_to(_conserved(), plan.spatial_shape + (4,))
    if finite_volume is not None:
        fv = jnp.asarray(finite_volume)
    kinetic_state = _kinetic_state(_spatial(plan.method)) if kinetic is None else kinetic
    return DynamicHybridCompositeState(fv, kinetic_state, ownership)


def test_dynamic_ownership_hysteresis_dwell_shock_and_dilation_are_deterministic():
    method = _kinetic_method()
    learned = _learned_binding(method)
    plan = DynamicHybridOwnershipPlan(
        method,
        learned,
        (5, 5),
        enter_threshold=0.8,
        exit_threshold=0.2,
        minimum_dwell_steps=2,
        finite_volume_stencil_radius=(1, 0),
        kinetic_reach=(1, 0),
    )
    initial = jnp.zeros((5, 5), dtype="bool").at[2, 2].set(True)
    state = plan.initialize(initial)
    score = jnp.zeros((5, 5)).at[2, 2].set(0.5)
    shock = jnp.zeros((5, 5), dtype="bool").at[0, 0].set(True)

    first = plan.propose(state, score, shock)
    second = plan.propose(state, score, shock)

    np.testing.assert_array_equal(first.finite_volume_owned, second.finite_volume_owned)
    assert bool(first.finite_volume_owned[2, 2])
    assert bool(first.finite_volume_owned[0, 0])
    assert bool(jnp.all(first.finite_volume_owned[:, 0]))
    assert plan.dilation_radius == (2, 0)
    assert plan.shock_owner == "finite_volume"
    assert plan.ownership_differentiability == "none"

    blocked = DynamicHybridOwnershipPlan(
        method,
        learned,
        (5, 5),
        enter_threshold=0.8,
        exit_threshold=0.2,
        minimum_dwell_steps=3,
        finite_volume_stencil_radius=(0, 0),
        kinetic_reach=(0, 0),
    )
    blocked_state = DynamicHybridCompositeState(
        jnp.broadcast_to(_conserved(), (5, 5, 4)),
        _kinetic_state(_spatial(method)),
        DynamicHybridOwnershipState(
            initial,
            jnp.zeros((5, 5), dtype=jnp.int32),
            jnp.full((5, 5), -1, dtype=jnp.int32),
            jnp.zeros((5, 5), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    blocked_decision = blocked.propose(
        blocked_state.ownership,
        jnp.zeros((5, 5)),
        jnp.zeros((5, 5), dtype="bool"),
    )
    assert bool(blocked_decision.finite_volume_owned[2, 2])


def test_dynamic_bidirectional_migration_is_exact_and_history_checkpoints():
    method = _kinetic_method()
    learned = _learned_binding(method)
    plan = DynamicHybridOwnershipPlan(
        method,
        learned,
        (5, 5),
        enter_threshold=0.8,
        exit_threshold=0.2,
        minimum_dwell_steps=0,
        finite_volume_stencil_radius=(0, 0),
        kinetic_reach=(0, 0),
    )
    owned = jnp.zeros((5, 5), dtype="bool").at[1, 1].set(True)
    kinetic = _kinetic_state(_spatial(method))
    perturbed_conserved = _conserved().at[0].set(1.1).at[-1].set(1.35)
    perturbed_equilibrium = _kinetic_state(_spatial(method), perturbed_conserved)
    kinetic = SmoothCompressibleKineticState(
        kinetic.particle_populations.at[3, 3].set(
            perturbed_equilibrium.particle_populations[3, 3]
        ),
        kinetic.total_energy_populations.at[3, 3].set(
            perturbed_equilibrium.total_energy_populations[3, 3]
        ),
    )
    state = _dynamic_state(plan, owned, kinetic=kinetic)
    score = jnp.full((5, 5), 0.5).at[1, 1].set(0.0).at[3, 3].set(1.0)
    decision = plan.propose(state.ownership, score, jnp.zeros((5, 5), dtype="bool"))

    deferred = plan.migrate(state, decision, jnp.asarray(False))
    assert not bool(deferred.evidence.accepted)
    _assert_same_tree(deferred.runtime_state, state)

    result = plan.migrate(state, decision, jnp.asarray(True))

    assert bool(result.evidence.accepted)
    assert bool(result.decision.exited_finite_volume[1, 1])
    assert bool(result.decision.entered_finite_volume[3, 3])
    np.testing.assert_array_equal(
        result.runtime_state.finite_volume_conserved[3, 3],
        method.moments(kinetic).conserved[3, 3],
    )
    np.testing.assert_allclose(
        result.evidence.kinetic_to_finite_volume_residual,
        0.0,
        atol=0.0,
    )
    assert int(result.runtime_state.ownership.transition_count[1, 1]) == 1
    assert int(result.runtime_state.ownership.transition_count[3, 3]) == 1
    checkpoint = plan.checkpoint(result.runtime_state, "dynamic-accepted")
    restored = plan.restore(checkpoint)
    _assert_same_tree(restored, result.runtime_state)


def test_dynamic_failed_fv_to_kinetic_lift_rolls_back_all_fields_and_history():
    method = _kinetic_method()
    learned = _learned_binding(method, velocity_bound=0.1)
    plan = DynamicHybridOwnershipPlan(
        method,
        learned,
        (5, 5),
        enter_threshold=0.8,
        exit_threshold=0.2,
        minimum_dwell_steps=0,
        finite_volume_stencil_radius=(0, 0),
        kinetic_reach=(0, 0),
    )
    owned = jnp.zeros((5, 5), dtype="bool").at[2, 2].set(True)
    finite_volume = jnp.broadcast_to(_conserved(), (5, 5, 4))
    unsupported = jnp.asarray((1.0, 0.3, 0.0, 1.4))
    finite_volume = finite_volume.at[2, 2].set(unsupported)
    state = _dynamic_state(plan, owned, finite_volume=finite_volume)
    score = jnp.full((5, 5), 0.5).at[2, 2].set(0.0)
    decision = plan.propose(state.ownership, score, jnp.zeros((5, 5), dtype="bool"))

    result = plan.migrate(state, decision, jnp.asarray(True))

    assert not bool(result.evidence.accepted)
    assert bool(result.evidence.rollback_applied)
    assert not bool(result.evidence.learned_support.successful[2, 2])
    _assert_same_tree(result.runtime_state, state)
    with pytest.raises(ValueError, match="Only accepted dynamic ownership"):
        plan.checkpoint(result.candidate, "rejected-migration")
