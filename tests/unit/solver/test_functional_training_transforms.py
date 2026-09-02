import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax._trainable import partition_trainable
from phydrax.solver._functional_surrogate import prepare_functional_update


def _fixed_term(domain, field_name, operator, *, points, label, blocks=None):
    component = domain.component()
    condition = phx.conditions.Residual(field_name, component, operator, label=label)
    batch = component.points({domain.labels[0]: jnp.asarray(points)})
    source = phx.integration.fixed(
        phx.integration.from_samples(phx.integration.mean_over(component), batch)
    )
    return phx.terms.ResidualPenalty(
        condition,
        source,
        blocks=blocks,
        label=label,
    )


def test_pseudo_transient_root_uses_explicit_relaxation_map():
    domain = phx.domain.Interval1d(0.0, 1.0)
    current = domain.Parameter(jnp.asarray(2.0))
    previous = domain.Parameter(jnp.asarray(1.0))
    term = _fixed_term(
        domain,
        "u",
        lambda value: value,
        points=[[0.1], [0.4], [0.7], [0.9]],
        label="equation",
    )
    solver = phx.solver.FunctionalSolver(functions={"u": current}, terms=(term,))
    params, fixed = partition_trainable(solver.functions)
    physical = solver.objective.prepare_training(
        (0,),
        scale=1.0,
        evaluation_key=jr.key(0),
        sampling_key=jr.key(1),
        iteration=1,
    )
    policy = phx.solver.PseudoTransientPolicy(
        0,
        phx.solver.ResidualRelaxationMap("u", lambda value: value),
        inverse_step=3.0,
        freshness="experimental_fixed",
    )
    update = prepare_functional_update(
        physical,
        params,
        fixed,
        solver.enforcement,
        training=phx.solver.FunctionalTrainingPlan(
            pseudo_transient=(policy,)
        ),
        previous_functions={"u": previous},
        pseudo_inverse_steps=(jnp.asarray(3.0),),
    )

    assert jnp.allclose(update.surrogate_loss(params, fixed), 25.0)
    assert jnp.allclose(update.physical_values(solver.functions).total, 4.0)


def test_gauss_newton_uses_pseudo_transient_residual_roots():
    domain = phx.domain.Interval1d(0.0, 1.0)
    solver = phx.solver.FunctionalSolver(
        functions={"u": domain.Parameter(jnp.asarray(2.0))},
        terms=(
            _fixed_term(
                domain,
                "u",
                lambda value: value,
                points=[[0.1], [0.4], [0.7], [0.9]],
                label="equation",
            ),
        ),
    )
    training = phx.solver.FunctionalTrainingPlan(
        pseudo_transient=(
            phx.solver.PseudoTransientPolicy(
                0,
                phx.solver.ResidualRelaxationMap(
                    "u",
                    lambda value: value,
                    map_id="identity",
                ),
                freshness="experimental_fixed",
            ),
        )
    )
    trained = solver.solve(
        num_iter=1,
        optim=phx.optim.GaussNewton(),
        keep_best=False,
        log_every=0,
        jit=False,
        training=training,
    )

    assert jnp.allclose(trained.functions["u"].func(), 1.0, atol=1e-8)


def test_causal_gates_reduce_later_slab_contribution():
    domain = phx.domain.TimeInterval(0.0, 1.0)
    field = domain.Parameter(jnp.asarray(1.0))
    term = _fixed_term(
        domain,
        "u",
        lambda value: value,
        points=[0.1, 0.5, 0.7, 0.8],
        label="dynamics",
    )
    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))
    params, fixed = partition_trainable(solver.functions)
    physical = solver.objective.prepare_training(
        (0,),
        scale=1.0,
        evaluation_key=jr.key(2),
        sampling_key=jr.key(3),
        iteration=1,
    )
    causal = phx.solver.CausalResidualPolicy(
        0,
        "t",
        phx.sampling.collocation.CausalTimeSlabSchedule(
            (0.0, 0.5, 1.0), causal_strength=1.0
        ),
    )
    update = prepare_functional_update(
        physical,
        params,
        fixed,
        solver.enforcement,
        training=phx.solver.FunctionalTrainingPlan(causal=(causal,)),
    )

    expected = 0.25 * (1.0 + 3.0 * jnp.exp(-1.0))
    assert jnp.allclose(update.surrogate_loss(params, fixed), expected)
    assert jnp.allclose(update.physical_values(solver.functions).total, 1.0)


def _two_term_solver():
    domain = phx.domain.Interval1d(0.0, 1.0)
    u = domain.Parameter(jnp.asarray(1.0))
    v = domain.Parameter(jnp.asarray(10.0))
    points = [[0.2], [0.8]]
    first = _fixed_term(
        domain,
        "u",
        lambda value: value,
        points=points,
        label="u",
    )
    second = _fixed_term(
        domain,
        "v",
        lambda value: value,
        points=points,
        label="v",
    )
    return phx.solver.FunctionalSolver(
        functions={"u": u, "v": v}, terms=(first, second)
    )


def test_gradient_norm_balancing_is_mean_one_and_reports_orthogonal_alignment():
    solver = _two_term_solver()
    params, fixed = partition_trainable(solver.functions)
    physical = solver.objective.prepare_training(
        (0, 1),
        scale=1.0,
        evaluation_key=jr.key(4),
        sampling_key=jr.key(5),
        iteration=1,
    )
    balance = phx.solver.FunctionalTermBalancePolicy(
        (
            phx.terms.ResidualBlockRef(0),
            phx.terms.ResidualBlockRef(1),
        ),
        method="gradient_norm",
        start=1,
        every=1,
        momentum=0.0,
    )
    diagnostics = phx.solver.FunctionalDiagnosticsPolicy(
        every=1, gradient_alignment=True
    )
    update = prepare_functional_update(
        physical,
        params,
        fixed,
        solver.enforcement,
        training=phx.solver.FunctionalTrainingPlan(
            term_balance=balance,
            diagnostics=diagnostics,
        ),
        term_multipliers=jnp.ones((2,)),
    )

    assert jnp.allclose(jnp.mean(update.term_multipliers), 1.0)
    assert update.term_multipliers[0] > update.term_multipliers[1]
    assert jnp.allclose(update.intra_gradient_alignment, 0.0, atol=1e-6)
    assert update.diagnostic_gradient is not None


def test_ntk_trace_balancing_preserves_equal_linear_sensitivities():
    solver = _two_term_solver()
    params, fixed = partition_trainable(solver.functions)
    physical = solver.objective.prepare_training(
        (0, 1),
        scale=1.0,
        evaluation_key=jr.key(6),
        sampling_key=jr.key(7),
        iteration=1,
    )
    balance = phx.solver.FunctionalTermBalancePolicy(
        (
            phx.terms.ResidualBlockRef(0),
            phx.terms.ResidualBlockRef(1),
        ),
        method="ntk_trace",
        start=1,
        every=1,
        momentum=0.0,
        ntk_probes=2,
        maximum_relative_standard_error=1.0,
    )
    update = prepare_functional_update(
        physical,
        params,
        fixed,
        solver.enforcement,
        training=phx.solver.FunctionalTrainingPlan(term_balance=balance),
        term_multipliers=jnp.ones((2,)),
    )

    assert jnp.allclose(update.term_multipliers, jnp.ones((2,)), atol=1e-5)
    assert jnp.all(jnp.isfinite(update.balance_statistics))


def test_stateful_transforms_tolerate_unselected_sampled_terms():
    solver = _two_term_solver()
    balance = phx.solver.FunctionalTermBalancePolicy(
        (
            phx.terms.ResidualBlockRef(0),
            phx.terms.ResidualBlockRef(1),
        ),
        start=1,
        every=1,
    )
    pseudo = phx.solver.PseudoTransientPolicy(
        0,
        phx.solver.ResidualRelaxationMap(
            "u",
            lambda value: value,
            map_id="identity-u",
        ),
        freshness="experimental_fixed",
    )
    trained = solver.solve(
        num_iter=2,
        optim=optax.adam(1e-2),
        keep_best=False,
        log_every=0,
        train_term_sample_size=1,
        training=phx.solver.FunctionalTrainingPlan(
            pseudo_transient=(pseudo,),
            term_balance=balance,
        ),
    )

    assert trained.training_state is not None
    assert jnp.all(jnp.isfinite(trained.training_state.term_multipliers))
