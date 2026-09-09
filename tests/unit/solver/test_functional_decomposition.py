#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.random as jr
import numpy as np
import optax

import phydrax as phx


def _fixed_penalty(condition, *, count=8):
    component = condition.on
    batch = component.sample(phx.domain.PointSampling(count), key=jr.key(0))
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component),
        batch,
    )
    return phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(realization),
    )


def _broken_pair_problem():
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 2)
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {
            cover.patch_ids[0]: cover.patches[0].domain.Parameter(1.0),
            cover.patch_ids[1]: cover.patches[1].domain.Parameter(-1.0),
        },
    )
    pairing = cover.pairings[0]
    jump = phx.conditions.SubdomainValueJump(
        family.field_name(pairing.left_patch_id),
        family.field_name(pairing.right_patch_id),
        pairing,
    )
    scoped = phx.solver.ScopedFunctionalTerm(
        _fixed_penalty(jump),
        phx.solver.PairScope(pairing.pairing_id),
    )
    return phx.solver.FunctionalDecompositionProblem.broken(family, (scoped,))


def _values(result):
    return tuple(float(field.func.value) for field in result.family.fields)


def test_joint_partition_of_unity_trains_canonical_local_parameters():
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 1)
    patch = cover.patches[0]
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {patch.patch_id: patch.domain.Parameter(1.0)},
    )
    condition = phx.conditions.Residual("u", domain.component(), lambda field: field)
    problem = phx.solver.FunctionalDecompositionProblem.partition_of_unity(
        family,
        _fixed_penalty(condition),
    )
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.JointDecompositionTraining(2),
            required_window_regularity=2,
        ),
    )

    result = phx.solver.solve_functional_decomposition(
        prepared,
        optax.sgd(0.1),
        jit=True,
        keep_best=False,
    )

    assert result.completed
    assert result.global_field is not None
    assert result.broken is None
    np.testing.assert_allclose(_values(result), (0.64,), atol=1.0e-12)
    np.testing.assert_allclose(
        result.evidence.training_loss,
        0.64**2,
        atol=1.0e-12,
    )


def test_jacobi_uses_one_snapshot_while_gauss_seidel_uses_latest_patch():
    problem = _broken_pair_problem()
    jacobi = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.BlockDecompositionTraining(1, 1, sweep="jacobi")
        ),
    )
    gauss_seidel = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.BlockDecompositionTraining(1, 1, sweep="gauss-seidel")
        ),
    )

    jacobi_result = phx.solver.solve_functional_decomposition(
        jacobi,
        optax.sgd(0.1),
        jit=True,
    )
    gauss_seidel_result = phx.solver.solve_functional_decomposition(
        gauss_seidel,
        optax.sgd(0.1),
        jit=True,
    )

    np.testing.assert_allclose(_values(jacobi_result), (0.6, -0.6), atol=1.0e-12)
    np.testing.assert_allclose(
        _values(gauss_seidel_result),
        (0.6, -0.68),
        atol=1.0e-12,
    )
    assert jacobi_result.state.local_steps == (1, 1)
    assert gauss_seidel_result.state.local_steps == (1, 1)


def test_block_decomposition_host_control_stops_after_committed_sweep():
    prepared = phx.solver.prepare_functional_decomposition(
        _broken_pair_problem(),
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.BlockDecompositionTraining(3, 1, sweep="jacobi")
        ),
    )
    phases = []

    def phase(event):
        return int(event.record.coordinates.phase)

    session = phx.execution.IterationSession(
        "functional-decomposition-test",
        sinks=(
            phx.execution.CallableIterationSink(
                lambda event: phases.append(phase(event)), "collector"
            ),
        ),
        control=phx.execution.CallableIterationHostControl(
            lambda event: phase(event) == int(phx.execution.IterationPhase.COMMIT),
            "stop-after-first-sweep",
        ),
    )
    result = phx.solver.solve_functional_decomposition(
        prepared,
        optax.sgd(0.1),
        session=session,
    )

    assert result.status == "stopped"
    assert result.state.completed_sweeps == 1
    assert phases == [
        int(phx.execution.IterationPhase.START),
        int(phx.execution.IterationPhase.COMMIT),
        int(phx.execution.IterationPhase.TERMINAL),
    ]
    assert result.iteration_session_state is not None
    assert result.iteration_session_state.stop_requested


def test_checkpoint_resume_matches_uninterrupted_block_training(tmp_path):
    problem = _broken_pair_problem()
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.BlockDecompositionTraining(3, 2, sweep="jacobi")
        ),
    )
    optimizer = optax.adam(0.02)

    uninterrupted = phx.solver.solve_functional_decomposition(
        prepared,
        optimizer,
        seed=4,
        jit=True,
    )
    partial = phx.solver.solve_functional_decomposition(
        prepared,
        optimizer,
        seed=4,
        jit=True,
        max_sweeps=1,
    )
    phx.solver.save_functional_decomposition_checkpoint(
        tmp_path / "decomposition",
        partial.state,
        prepared,
    )
    restored = phx.solver.load_functional_decomposition_checkpoint(
        tmp_path / "decomposition",
        prepared,
        partial.state,
    )
    resumed = phx.solver.solve_functional_decomposition(
        prepared,
        optimizer,
        seed=4,
        jit=True,
        state=restored,
    )

    assert resumed.state.completed_sweeps == 3
    assert resumed.state.local_steps == (6, 6)
    assert eqx.tree_equal(resumed.state.functions, uninterrupted.state.functions)
    assert eqx.tree_equal(
        resumed.state.optimizer_states,
        uninterrupted.state.optimizer_states,
    )


def test_schwarz_strategy_requires_pair_terms_and_reports_interface_defect():
    problem = _broken_pair_problem()
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(
                2,
                1,
                sweep="gauss-seidel",
            )
        ),
    )
    result = phx.solver.solve_functional_decomposition(
        prepared,
        optax.sgd(0.1),
        jit=True,
    )

    assert result.state.strategy == "schwarz"
    assert result.state.completed_sweeps == 2
    assert result.broken is not None
    assert result.evidence.maximum_pair_loss < 4.0


def test_coarse_correction_freezes_base_and_trains_assembled_fine_field():
    domain = phx.domain.Interval1d(0.0, 1.0)
    base = domain.Parameter(1.0)
    base_term = _fixed_penalty(
        phx.conditions.Residual("u", domain.component(), lambda field: field)
    )
    base_solver = phx.solver.FunctionalSolver(
        functions={"u": base},
        terms=(base_term,),
    )
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        2,
        overlap_fraction=0.2,
    )
    family = phx.domain.LocalFieldFamily(
        "du",
        cover,
        {patch.patch_id: patch.domain.Parameter(-0.25) for patch in cover.patches},
    )
    correction = phx.solver.FunctionalCoarseCorrection(
        base_solver,
        "u",
        family,
    )

    trained = correction.training_solver.solve(
        num_iter=1,
        optim=optax.sgd(0.05),
        jit=True,
        keep_best=False,
        log_every=0,
    )
    final = correction.finalize(trained)

    assert float(base_solver.functions["u"].func.value) == 1.0
    assert float(final.loss(key=jr.key(3))) < float(base_solver.loss(key=jr.key(3)))
