import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.control._direct_collocation_refinement import _maximum_state_error


def test_refinement_state_error_ignores_quaternion_pose_sign():
    geometry = phx.metrix.QuaternionPoseStateGeometry()
    local_space = phx.linalg.ArraySpace((6,), dtype=jnp.float32)
    state_layout = phx.dynamics.StateLayout(
        (7,),
        geometry=geometry,
        local_space=local_space,
        tangent_space=local_space,
        layout_id="test:refinement-quaternion-pose",
    )
    pose = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.2, -0.4, 0.7])
    references = jnp.stack((pose, pose.at[4:].add(0.1)))
    points = references.at[:, :4].multiply(-1.0)

    assert jnp.allclose(
        _maximum_state_error(state_layout, references, points),
        0.0,
    )


def _source_result(*, bounds=None):
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: jnp.asarray((time,)),
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id="refinement-time-forcing",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.asarray((0.0,)),
        running_cost=lambda time, state, control, args: control[0] ** 2,
        problem_id="refinement-time-forcing",
    )
    mesh = phx.discretization.TemporalMesh.uniform(
        0.0,
        1.0,
        4,
        role="collocation",
        mesh_id="refinement-source-mesh",
    )
    plan = phx.control.DirectCollocationPlan(
        mesh,
        method=phx.solver.ThetaMethod(0.5, endpoint=False),
        audit=phx.control.DirectCollocationAuditPolicy(off_grid_points=2),
        derivatives=phx.control.DirectCollocationDerivativePolicy(verify=False),
        plan_id="refinement-source-plan",
    )
    return phx.control.solve_direct_collocation(
        problem,
        plan,
        0.5 * mesh.nodes[:, None] ** 2,
        jnp.zeros((mesh.num_steps, 1)),
        bounds=bounds,
        method=phx.optim.PrimalDualInteriorPoint(
            mode="dense-filter", max_dense_dimension=128
        ),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-8,
            relative_optimality=0.0,
            maximum_steps=80,
        ),
    )


def _policy(**overrides):
    values = {
        "mode": "uniform",
        "maximum_levels": 2,
        "maximum_intervals": 32,
        "off_grid_defect_tolerance": 0.04,
        "relative_objective_tolerance": 1.0,
        "state_tolerance": 1.0,
        "control_tolerance": 1.0,
        "minimum_defect_reduction": 0.1,
    }
    values.update(overrides)
    return phx.control.DirectCollocationRefinementPolicy(**values)


def test_off_grid_audit_retains_per_interval_defects():
    result = _source_result()
    audit = result.diagnostics.off_grid
    assert bool(result.successful)
    assert audit.times.shape == (4, 2)
    assert audit.dynamics_residuals.shape == (4, 2, 1)
    assert audit.interval_defects.shape == (4,)
    assert audit.interval_path_violations.shape == (4,)
    assert jnp.all(audit.interval_defects > 0.0)
    assert jnp.allclose(audit.maximum_defect, jnp.max(audit.interval_defects))
    assert bool(audit.finite)
    assert not audit.certified


def test_uniform_refinement_transfers_only_primal_decisions():
    source = _source_result()
    selection = phx.control.select_direct_collocation_intervals(source, _policy())
    assert selection.selected_indices.tolist() == [0, 1, 2, 3]
    assert selection.target_intervals == 8
    transfer = phx.control.refine_direct_collocation(
        source,
        _policy(),
        selection=selection,
    )
    assert transfer.target_plan.mesh.num_steps == 8
    assert transfer.decision.states.shape == (9, 1)
    assert transfer.decision.controls.shape == (8, 1)
    assert transfer.old_node_state_error == 0.0
    assert transfer.control_representation_error == 0.0
    assert transfer.parameter_error == 0.0
    assert transfer.duration_error == 0.0
    assert not transfer.dual_transferred


def test_bulk_refinement_selects_smallest_defect_mass_prefix():
    source = _source_result()
    policy = _policy(mode="bulk-defect", bulk_fraction=0.5)
    selection = phx.control.select_direct_collocation_intervals(source, policy)
    assert selection.selected_indices.size == 2
    assert selection.target_intervals == 6
    assert not selection.capacity_exceeded


def test_refinement_requires_provider_for_mesh_shaped_bounds():
    state_guess = 0.5 * jnp.linspace(0.0, 1.0, 5)[:, None] ** 2
    source = _source_result(
        bounds=phx.control.DirectCollocationBounds(
            states=phx.optim.Bounds(
                state_guess - 1.0,
                state_guess + 1.0,
            )
        )
    )
    with pytest.raises(ValueError, match="bound provider"):
        phx.control.refine_direct_collocation(source, _policy())
    transfer = phx.control.refine_direct_collocation(
        source,
        _policy(),
        bounds_provider=lambda plan, decision: phx.control.DirectCollocationBounds(
            states=phx.optim.Bounds(-1.0, 2.0)
        ),
    )
    assert transfer.bounds.states is not None


def test_refinement_study_reduces_sampled_defect_and_converges():
    source = _source_result()
    study = phx.control.solve_refined_direct_collocation(
        source,
        _policy(),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="dense-filter", max_dense_dimension=256
        ),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-8,
            relative_optimality=0.0,
            maximum_steps=80,
        ),
    )
    assert bool(study.converged)
    assert int(study.status) == phx.control.DIRECT_REFINEMENT_CONVERGED
    assert len(study.levels) == 1
    assert study.final_result.compilation.plan.mesh.num_steps == 8
    assert (
        study.final_result.diagnostics.maximum_off_grid_defect
        < source.diagnostics.maximum_off_grid_defect
    )
    assert study.levels[0].defect_reduction >= 0.4


def test_refinement_capacity_and_failed_source_are_explicit():
    source = _source_result()
    selection = phx.control.select_direct_collocation_intervals(
        source,
        _policy(maximum_intervals=6),
    )
    assert selection.capacity_exceeded
    failed = eqx.tree_at(
        lambda result: result.status,
        source,
        jnp.asarray(phx.control.DIRECT_COLLOCATION_OPTIMIZER_FAILED),
    )
    study = phx.control.solve_refined_direct_collocation(
        failed,
        _policy(),
        method=phx.optim.PrimalDualInteriorPoint(
            mode="dense-filter", max_dense_dimension=128
        ),
        termination=phx.optim.OptimizationTermination(),
    )
    assert int(study.status) == phx.control.DIRECT_REFINEMENT_SOURCE_FAILED
    assert not bool(study.converged)
