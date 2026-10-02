#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

import phydrax as phx
import phydrax.solver.functional_decomposition._checkpoint as decomposition_checkpoint
from phydrax._strict import StrictModule


def _fixed_penalty(condition: Any, *, count: Any = 8) -> Any:
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


def _broken_pair_problem() -> Any:
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


def _values(result: Any) -> Any:
    return tuple(float(field.func.value) for field in result.family.fields)


def test_functional_decomposition_scenario_1() -> None:
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
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 1)
    patch = cover.patches[0]
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {patch.patch_id: patch.domain.Parameter(jnp.asarray(0.0))},
    )
    prepared = phx.solver.prepare_functional_decomposition(
        phx.solver.FunctionalDecompositionProblem.broken(
            family,
            (
                phx.solver.ScopedFunctionalTerm(
                    _PoleTerm(family.field_name(patch.patch_id)),
                    phx.solver.PatchScope(patch.patch_id),
                ),
            ),
        ),
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.BlockDecompositionTraining(1, 1, sweep="jacobi")
        ),
    )

    # The value, gradient, candidate, and optimizer state are finite; the step
    # lands on the pole, so only the candidate objective is nonfinite.
    with pytest.raises(FloatingPointError, match="rejected as non-finite"):
        phx.solver.solve_functional_decomposition(prepared, optax.sgd(-0.5), jit=False)
    nonfinite_state_optimizer = optax.GradientTransformation(
        # ty: ignore[invalid-argument-type]
        lambda parameters: jnp.asarray(0.0),
        # ty: ignore[invalid-argument-type]
        lambda gradients, optimizer_state, params=None: (
            jax.tree.map(jnp.zeros_like, gradients),
            jnp.asarray(jnp.inf),
        ),
    )
    with pytest.raises(FloatingPointError, match="rejected as non-finite"):
        phx.solver.solve_functional_decomposition(
            prepared, nonfinite_state_optimizer, jit=True
        )
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


class _PoleTerm(phx.terms.AbstractScalarTerm):
    """`(u - 1)^-2` of one constant field, unchecked, so a pole is a nonfinite value."""

    fields: tuple[str, ...] = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(self, field_name: Any) -> None:
        self.fields = (field_name,)
        self.label = None

    def loss(
        self, functions: Any, /, *, key: Any = None, iter_: Any = None, **kwargs: Any
    ) -> Any:
        del key, iter_, kwargs
        return (functions[self.fields[0]].func.value - 1.0) ** -2


class _Affine(StrictModule, phx.ParameterOwner):
    offset: jax.Array
    slope: jax.Array

    def __call__(self, x: Any, /) -> Any:
        return self.offset + self.slope * x[0]


def test_periodic_self_seam_term_trains_one_patch_field() -> None:
    # ``u = a + b x`` on ``[0, 1]`` with seam residual ``u(1) - u(0) = b``: the
    # penalty ``b^2`` has gradient ``(0, 2b)``, so one SGD step at rate 0.1 maps
    # ``(a, b) = (0.5, 2)`` to ``(0.5, 1.6)`` and leaves the offset untouched.
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 1, periodic=True)
    (patch,) = cover.patches
    (seam,) = cover.pairings
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {
            patch.patch_id: patch.domain.Function("x")(
                _Affine(jnp.asarray(0.5), jnp.asarray(2.0))
            )
        },
    )
    condition = phx.conditions.Periodic(family.ref(patch.patch_id), seam)
    problem = phx.solver.FunctionalDecompositionProblem.broken(
        family,
        phx.solver.ScopedFunctionalTerm(
            _fixed_penalty(condition), phx.solver.PairScope(seam.pairing_id)
        ),
    )
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(phx.solver.JointDecompositionTraining(1)),
    )

    result = phx.solver.solve_functional_decomposition(
        prepared, optax.sgd(0.1), jit=True, keep_best=False
    )

    ends = patch.domain.component().points({"x": jnp.asarray([[0.0], [1.0]])})
    np.testing.assert_allclose(
        result.family.field(patch.patch_id)(ends).data, (0.5, 2.1), atol=1.0e-12
    )


def test_block_decomposition_host_control_stops_after_committed_sweep() -> None:
    prepared = phx.solver.prepare_functional_decomposition(
        _broken_pair_problem(),
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.BlockDecompositionTraining(3, 1, sweep="jacobi")
        ),
    )
    phases = []

    def phase(event: Any) -> Any:
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


def test_checkpoint_resume_matches_uninterrupted_block_training(
    tmp_path: Any, monkeypatch: Any
) -> None:
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
    checkpoint_directory = tmp_path / "decomposition"
    state_path = next(checkpoint_directory.glob("state-*.eqx"))
    deserialize = decomposition_checkpoint.eqx.tree_deserialise_leaves

    def replace_path_after_open(state_stream: Any, *args: Any, **kwargs: Any) -> Any:
        replacement = checkpoint_directory / "replacement.eqx"
        replacement.write_bytes(b"replacement")
        replacement.replace(state_path)
        return deserialize(state_stream, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            decomposition_checkpoint.eqx,
            "tree_deserialise_leaves",
            replace_path_after_open,
        )
        # A state file replaced while it is being read fails closed.
        with pytest.raises(ValueError, match="changed during reading"):
            phx.solver.load_functional_decomposition_checkpoint(
                tmp_path / "decomposition",
                prepared,
                partial.state,
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
        resumed.state.kernel_states,
        uninterrupted.state.kernel_states,
    )


def test_functional_decomposition_scenario_2() -> None:
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

    # ty: ignore[unresolved-attribute]
    assert float(base_solver.functions["u"].func.value) == 1.0
    assert float(final.loss(key=jr.key(3))) < float(base_solver.loss(key=jr.key(3)))
