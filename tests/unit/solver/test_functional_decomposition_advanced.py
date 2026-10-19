#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import json
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

import phydrax as phx
import phydrax.solver.functional_decomposition._export as decomposition_export
from phydrax.domain._function import TrainableConstCallable


def _fixed_penalty(condition: Any, count: Any = 8) -> Any:
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


def _pou_problem() -> Any:
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(
        domain,
        "x",
        2,
        overlap_fraction=0.2,
    )
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {
            cover.patch_ids[0]: cover.patches[0].domain.Parameter(1.0),
            cover.patch_ids[1]: cover.patches[1].domain.Parameter(-1.0),
        },
    )
    condition = phx.conditions.Residual("u", domain.component(), lambda field: field)
    problem = phx.solver.FunctionalDecompositionProblem.partition_of_unity(
        family,
        _fixed_penalty(condition),
    )
    return problem


def _broken_problem(patches: int = 2) -> Any:
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", patches)
    family = phx.domain.LocalFieldFamily(
        "u",
        cover,
        {
            patch.patch_id: patch.domain.Parameter(1.0 if index % 2 == 0 else -1.0)
            for index, patch in enumerate(cover.patches)
        },
    )
    terms = tuple(
        phx.solver.ScopedFunctionalTerm(
            _fixed_penalty(
                phx.conditions.SubdomainValueJump(
                    family.ref(pairing.left_patch_id),
                    family.ref(pairing.right_patch_id),
                    pairing,
                )
            ),
            phx.solver.PairScope(pairing.pairing_id),
        )
        for pairing in cover.pairings
    )
    problem = phx.solver.FunctionalDecompositionProblem.broken(family, terms)
    return problem


def test_functional_decomposition_advanced_scenario_1() -> None:
    problem = _pou_problem()
    first, second = problem.cover.patch_ids
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.BlockDecompositionTraining(
                2,
                1,
                sweep="colored",
                active_patch_ids=(first,),
                fixed_patch_ids=(second,),
            )
        ),
    )

    result = phx.solver.solve_functional_decomposition(
        prepared,
        optax.adam(0.05),
        jit=True,
    )

    assert float(result.family.field(first).func.value) < 1.0
    assert float(result.family.field(second).func.value) == -1.0
    assert result.state.local_steps == (2, 0)
    assert result.state.kernel_states[1] is None
    problem = _broken_problem()
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(
                3,
                1,
                sweep="jacobi",
                relaxation=0.5,
            ),
            trace_points=8,
        ),
    )
    initial = phx.solver.capture_schwarz_trace_state(
        problem,
        problem.functions,
        prepared.trace_batches,
        sweep=0,
    )

    result = phx.solver.solve_functional_decomposition(
        prepared,
        optax.sgd(0.1),
        jit=True,
    )

    assert result.state.trace_state is not None
    assert result.state.trace_state.sweep == 3
    assert result.state.trace_state.maximum_defect < initial.maximum_defect
    assert result.state.local_steps == (3, 3)
    problem = _broken_problem()
    family = problem.family
    pairing = family.cover.pairings[0]
    points = pairing.component.sample(phx.domain.PointSampling(6), key=jr.key(2))
    functions = family.solver_functions()
    basis = jnp.ones((6, 1))

    mortar = phx.terms.MortarInterfacePenalty(
        family.ref(pairing.left_patch_id),
        family.ref(pairing.right_patch_id),
        pairing,
        points,
        basis,
    )
    augmented = phx.terms.AugmentedValueConstraint.initialize(
        family.ref(pairing.left_patch_id),
        family.ref(pairing.right_patch_id),
        pairing,
        points,
        functions,
        penalty=2.0,
    )

    np.testing.assert_allclose(mortar.loss(functions), 4.0)
    np.testing.assert_allclose(augmented.loss(functions), 4.0)
    updated, evidence = augmented.update(functions)
    np.testing.assert_allclose(updated.multiplier, -4.0)
    np.testing.assert_allclose(evidence.primal_residual, 2.0)

    smooth = phx.domain.LocalFieldFamily(
        "v",
        family.cover,
        {
            patch.patch_id: patch.domain.Function("x")(lambda x: x[0] ** 2)
            for patch in family.cover.patches
        },
    )
    nitsche = phx.terms.NitscheInterfaceFunctional(
        smooth.ref(pairing.left_patch_id),
        smooth.ref(pairing.right_patch_id),
        pairing,
        points,
        lambda field: phx.operators.grad(field, var="x"),
        lambda field: phx.operators.grad(field, var="x"),
        penalty=10.0,
    )
    np.testing.assert_allclose(nitsche.loss(smooth.solver_functions()), 0.0)


def test_point_interface_mortar_preserves_vector_value_axes_without_sample_axis() -> None:
    cover = phx.domain.cartesian_subdomain_cover(phx.domain.Interval1d(0.0, 1.0), "x", 2)
    left, right = cover.patches
    family = phx.domain.LocalFieldFamily(
        "vector",
        cover,
        {
            left.patch_id: left.domain.Function()(jnp.zeros((2,), dtype=jnp.float64)),
            right.patch_id: right.domain.Function()(
                jnp.asarray([3.0, 4.0], dtype=jnp.float64)
            ),
        },
    )
    pairing = cover.pairings[0]
    points = pairing.component.sample(phx.domain.PointSampling(6))
    term = phx.terms.MortarInterfacePenalty(
        family.ref(left.patch_id),
        family.ref(right.patch_id),
        pairing,
        points,
        jnp.ones((3, 1), dtype=jnp.float64),
        weights=jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float64),
    )
    assert points.structure.blocks == ()
    assert eqx.filter_jit(term.loss)(family.solver_functions()) == pytest.approx(25.0)


def test_functional_decomposition_advanced_scenario_2() -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    solver = phx.solver.FunctionalSolver(
        functions={"u": domain.Parameter(1.0)},
        terms=(
            _fixed_penalty(
                phx.conditions.Residual("u", domain.component(), lambda value: value)
            ),
        ),
    )
    paths = tuple(
        path
        for path in phx.nn.parameters.ParameterSubspace.array_leaf_paths(solver.functions)
        if ".func.value" in path
    )
    subspace = phx.nn.parameters.ParameterSubspace.from_leaf_paths(
        solver.functions,
        paths,
    )

    result = phx.solver.dense_local_curvature_step(solver, subspace)

    assert result.accepted
    parameter = result.functions["u"].func
    if not isinstance(parameter, TrainableConstCallable):
        raise RuntimeError(
            "curvature result parameter must be a trainable constant; got another callable"
        )
    assert abs(float(parameter.value)) < 1.0e-4
    assert result.approximation == "exact-local-dense-hessian"
    domain = phx.domain.Interval1d(0.0, 1.0)
    coordinates = jnp.linspace(0.0, 1.0, 9)
    points = domain.component().points({"x": coordinates[:, None]})
    test_space = phx.terms.polynomial_test_space(
        domain.component(),
        points,
        2.0 * coordinates - 1.0,
        2,
        test_space_id="legendre-2",
    )
    term = phx.terms.LocalizedResidualNorm(
        phx.conditions.Residual("u", domain.component(), lambda value: value),
        test_space,
    )

    np.testing.assert_allclose(term.loss({"u": domain.Parameter(1.0)}), 1.0)
    assert test_space.evidence.verified
    assert not test_space.evidence.orthonormal
    domain = phx.domain.Interval1d(0.0, 1.0)
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", 2)
    local = {}
    terms = []
    for index, patch in enumerate(cover.patches):
        model = phx.nn.models.MLP(
            in_size=1,
            out_size="scalar",
            hidden_sizes=(),
            rwf=False,
            key=jr.key(index),
        )
        local[patch.patch_id] = patch.domain.Model("x")(model)
    family = phx.domain.LocalFieldFamily("u", cover, local)
    for patch in cover.patches:
        condition = phx.conditions.Residual(
            family.ref(patch.patch_id).solver_name,
            patch.interior,
            lambda field: field - 1.0,
        )
        terms.append(
            phx.solver.ScopedFunctionalTerm(
                _fixed_penalty(condition),
                phx.solver.PatchScope(patch.patch_id),
            )
        )
    prepared = phx.solver.prepare_functional_decomposition(
        phx.solver.FunctionalDecompositionProblem.broken(family, terms),
        phx.solver.FunctionalDecompositionPlan(phx.solver.JointDecompositionTraining(0)),
    )
    local_result = phx.solver.solve_local_kfac(
        prepared,
        cover.patch_ids[0],
        num_iter=1,
        jit=False,
    )

    scalar_solver = phx.solver.FunctionalSolver(
        functions={"u": domain.Parameter(1.0)},
        terms=(
            _fixed_penalty(
                phx.conditions.Residual(
                    "u",
                    domain.component(),
                    lambda field: field,
                )
            ),
        ),
    )
    paths = tuple(
        path
        for path in phx.nn.parameters.ParameterSubspace.array_leaf_paths(
            scalar_solver.functions
        )
        if ".func.value" in path
    )
    subspace = phx.nn.parameters.ParameterSubspace.from_leaf_paths(
        scalar_solver.functions,
        paths,
    )
    matrix_free = phx.solver.matrix_free_gauss_newton_step(
        scalar_solver,
        subspace,
    )

    assert local_result.approximation == "local-block-kfac"
    assert matrix_free.matrix_free
    assert matrix_free.accepted
    assert matrix_free.final_loss < matrix_free.initial_loss
    domain = phx.domain.Interval1d(0.0, 1.0)
    base_term = _fixed_penalty(
        phx.conditions.Residual("u", domain.component(), lambda value: value)
    )
    base = phx.solver.FunctionalSolver(
        functions={"u": domain.Parameter(1.0)},
        terms=(base_term,),
    )
    levels = []
    for index, count in enumerate((1, 2)):
        cover = phx.domain.cartesian_subdomain_cover(
            domain,
            "x",
            count,
            overlap_fraction=0.0 if count == 1 else 0.2,
        )
        family = phx.domain.LocalFieldFamily(
            f"du-{index}",
            cover,
            {patch.patch_id: patch.domain.Parameter(-0.1) for patch in cover.patches},
        )
        levels.append(phx.domain.SubdomainLevel(f"level-{index}", family))
    hierarchy = phx.domain.SubdomainHierarchy(
        levels,
        hierarchy_id="two-level",
    )

    result = phx.solver.train_functional_hierarchy(
        base,
        "u",
        hierarchy,
        phx.solver.FunctionalHierarchyPlan((1, 1)),
        optax.sgd(0.01),
        jit=False,
    )

    assert len(result.level_solvers) == 2
    assert float(result.solver.loss(key=jr.key(7))) < float(base.loss(key=jr.key(7)))


def test_sharding_hybrid_and_deployment_roundtrip(
    tmp_path: Any, monkeypatch: Any
) -> None:
    problem = _pou_problem()
    family = problem.family
    plan = phx.solver.FunctionalDecompositionShardingPlan(family.cover)
    sharded = phx.solver.place_local_field_family(family, plan)
    participants = tuple(
        phx.solver.FixedPatchParticipant(
            patch.patch_id,
            {"u": sharded.family.field(patch.patch_id)},
        )
        for patch in family.cover.patches
    )
    hybrid = phx.solver.HybridFunctionalDecomposition(family.cover, participants)
    artifact = phx.solver.DecompositionDeploymentArtifact(
        hybrid.family("u"),
        assembly="partition-of-unity",
    )

    phx.solver.save_decomposition_artifact(tmp_path / "deployment", artifact)
    checkpoint_directory = tmp_path / "deployment"
    state_path = next(checkpoint_directory.glob("state-*.eqx"))
    deserialize = decomposition_export.eqx.tree_deserialise_leaves

    def replace_path_after_open(state_stream: Any, *args: Any, **kwargs: Any) -> Any:
        replacement = checkpoint_directory / "replacement.eqx"
        replacement.write_bytes(b"replacement")
        replacement.replace(state_path)
        return deserialize(state_stream, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            decomposition_export.eqx,
            "tree_deserialise_leaves",
            replace_path_after_open,
        )
        # A state file replaced while it is being read fails closed.
        with pytest.raises(ValueError, match="changed during reading"):
            phx.solver.load_decomposition_artifact(tmp_path / "deployment", artifact)
    phx.solver.save_decomposition_artifact(tmp_path / "deployment", artifact)
    restored = phx.solver.load_decomposition_artifact(
        tmp_path / "deployment",
        artifact,
    )

    assert sharded.evidence.verified
    assert eqx.tree_equal(restored.family.fields, artifact.family.fields)
    assert restored.artifact_id == artifact.artifact_id
    phx.solver.save_decomposition_artifact(checkpoint_directory, artifact)
    manifest_path = checkpoint_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state_file"] = f"../{manifest['state_file']}"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="canonical basename"):
        phx.solver.load_decomposition_artifact(checkpoint_directory, artifact)

    phx.solver.save_decomposition_artifact(checkpoint_directory, artifact)
    manifest = json.loads(manifest_path.read_text())
    manifest["state_sha256"] = "not-a-checksum"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum is invalid"):
        phx.solver.load_decomposition_artifact(checkpoint_directory, artifact)


def test_functional_decomposition_advanced_scenario_3() -> None:
    problem = _broken_problem()
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(1, 1),
            trace_points=6,
        ),
    )
    quantity = phx.solver.SchwarzTraceQuantity(
        "value",
        lambda field: field,
        lambda field: field,
    )
    previous = phx.solver.capture_generalized_trace_state(
        problem,
        problem.functions,
        prepared.trace_batches,
        quantity,
        sweep=0,
    )
    proposed = phx.solver.capture_generalized_trace_state(
        problem,
        problem.functions,
        prepared.trace_batches,
        quantity,
        sweep=1,
    )
    relaxation = phx.nonlinear.AitkenRelaxation()
    increment = proposed.exchanges[0].left_target - previous.exchanges[0].left_target
    omega = relaxation.update(increment, increment, jnp.asarray(1.0))
    asynchronous = phx.solver.solve_asynchronous_schwarz(
        prepared,
        phx.solver.AsynchronousSchwarzPlan(
            4,
            1,
            maximum_staleness=1,
        ),
        optax.sgd(0.1),
        jit=True,
    )

    assert proposed.exchanges[0].quantity_id == "value"
    assert float(omega) == 1.0
    assert asynchronous.state.patch_revisions == (2, 2)
    assert asynchronous.state.maximum_observed_staleness == 1
    assert phx.solver.FunctionalCyclePlan(1, (1, 1), kind="v").order(2) == (
        (0, True),
        (1, True),
        (0, False),
    )
    assert phx.solver.FunctionalCyclePlan(1, (1, 1), kind="f").order(2) == (
        (0, True),
        (0, True),
        (1, True),
        (0, False),
    )
    devices = tuple(jax.devices()[:1])
    pou = phx.solver.distributed_pou_collective(
        jnp.asarray([[[2.0], [4.0]]]),
        jnp.asarray([[1.0, 1.0]]),
        devices=devices,
    )
    schwarz = phx.solver.distributed_schwarz_exchange(
        jnp.asarray([[3.0, 5.0]]),
        jnp.asarray([0]),
        devices=devices,
    )

    np.testing.assert_allclose(pou.value, [[[2.0], [4.0]]])
    np.testing.assert_allclose(schwarz.value, [[3.0, 5.0]])
    assert pou.evidence.verified
    assert schwarz.evidence.verified


def test_multi_device_collectives_use_shard_map_global_semantics() -> None:
    devices = tuple(jax.devices()[:2])
    if len(devices) < 2:
        pytest.skip("requires two real or explicitly configured JAX devices")
    pou = phx.solver.distributed_pou_collective(
        jnp.asarray([[[2.0], [4.0]], [[4.0], [8.0]]]),
        jnp.asarray([[1.0, 1.0], [1.0, 3.0]]),
        devices=devices,
    )
    schwarz = phx.solver.distributed_schwarz_exchange(
        jnp.asarray([[3.0, 5.0], [7.0, 11.0]]),
        jnp.asarray([1, 0]),
        devices=devices,
    )

    np.testing.assert_allclose(pou.value, [[[3.0], [7.0]], [[3.0], [7.0]]])
    np.testing.assert_allclose(schwarz.value, [[7.0, 11.0], [3.0, 5.0]])


def _trace_targets(state: Any) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(target).reshape(-1)
            for exchange in state.exchanges
            for target in (exchange.left_target, exchange.right_target)
        ]
    )


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    "sweep",
    ["jacobi", "gauss-seidel", "colored"],
    ids=["jacobi", "gauss-seidel", "colored"],
)
def test_aitken_relaxes_complete_sweep(sweep: Any) -> None:
    problem = _broken_problem()
    policy = phx.nonlinear.AitkenRelaxation(initial_relaxation=0.7)
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(3, 1, sweep=sweep, relaxation=policy),
            trace_points=6,
        ),
    )
    raw_prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(3, 1, sweep=sweep, relaxation=1.0),
            trace_points=6,
        ),
    )
    optimizer = optax.sgd(0.1)
    state = None
    previous = None
    omega = 0.7
    for step in range(3):
        entry = (
            phx.solver.capture_schwarz_trace_state(
                problem, problem.functions, prepared.trace_batches, sweep=0
            )
            if state is None
            else state.trace_state
        )
        raw_state = (
            None
            if state is None
            else eqx.tree_at(
                lambda value: (value.previous_increment, value.omega),
                state,
                (None, None),
            )
        )
        raw = phx.solver.solve_functional_decomposition(
            raw_prepared, optimizer, state=raw_state, max_sweeps=1, jit=True
        )
        increment = _trace_targets(raw.state.trace_state) - _trace_targets(entry)
        if previous is not None:
            delta = increment - previous
            denominator = np.vdot(delta, delta).real
            candidate = (
                omega
                if denominator == 0
                else -omega * np.vdot(previous, delta).real / denominator
            )
            omega = float(
                np.clip(candidate if np.isfinite(candidate) else omega, 0.05, 1.5)
            )
        result = phx.solver.solve_functional_decomposition(
            prepared, optimizer, state=state, max_sweeps=1, jit=True
        )
        np.testing.assert_allclose(
            _trace_targets(result.state.trace_state),
            _trace_targets(entry) + omega * increment,
            rtol=1e-12,
            atol=1e-12,
        )
        relaxed_increment = result.state.previous_increment
        relaxed_omega = result.state.omega
        if relaxed_increment is None or relaxed_omega is None:
            raise AssertionError("Aitken sweep must retain its increment and omega.")
        np.testing.assert_allclose(relaxed_increment, increment)
        assert float(relaxed_omega) == pytest.approx(omega)
        assert result.state.completed_sweeps == step + 1
        assert eqx.tree_equal(result.state.functions, raw.state.functions)
        state = result.state
        previous = increment


@pytest.mark.strict_jax
@pytest.mark.parametrize(
    "relaxation",
    [0.7, phx.nonlinear.AitkenRelaxation(initial_relaxation=0.7)],
    ids=["fixed-float", "aitken"],
)
@pytest.mark.parametrize(
    "sweep",
    ["jacobi", "gauss-seidel", "colored"],
    ids=["jacobi", "gauss-seidel", "colored"],
)
def test_schwarz_chunk_and_checkpoint_resume(
    sweep: Any, relaxation: Any, tmp_path: Any
) -> None:
    problem = _broken_problem()
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(
                3, 1, sweep=sweep, relaxation=relaxation
            ),
            trace_points=6,
        ),
    )
    optimizer = optax.sgd(0.02, momentum=0.9)
    uninterrupted = phx.solver.solve_functional_decomposition(
        prepared, optimizer, jit=True
    )
    partial = phx.solver.solve_functional_decomposition(
        prepared, optimizer, max_sweeps=1, jit=True
    )
    chunked = phx.solver.solve_functional_decomposition(
        prepared, optimizer, state=partial.state, jit=True
    )
    phx.solver.save_functional_decomposition_checkpoint(
        tmp_path / "schwarz", partial.state, prepared
    )
    restored = phx.solver.load_functional_decomposition_checkpoint(
        tmp_path / "schwarz", prepared, partial.state
    )
    resumed = phx.solver.solve_functional_decomposition(
        prepared, optimizer, state=restored, jit=True
    )
    assert eqx.tree_equal(chunked.state, uninterrupted.state)
    assert eqx.tree_equal(resumed.state, uninterrupted.state)
    if isinstance(relaxation, float):
        assert resumed.state.previous_increment is None
        assert resumed.state.omega is None


@pytest.mark.strict_jax
def test_aitken_refuses_heterogeneous_trace_quantities() -> None:
    problem = _broken_problem(3)
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(
                3, 1, relaxation=phx.nonlinear.AitkenRelaxation()
            ),
            trace_points=6,
        ),
    )
    partial = phx.solver.solve_functional_decomposition(
        prepared, optax.sgd(0.1), max_sweeps=1
    )
    trace_state = partial.state.trace_state
    if trace_state is None:
        raise AssertionError("A broken Schwarz sweep must retain its trace state.")
    exchange = trace_state.exchanges[1]
    incompatible = phx.solver.TraceExchangeState(
        pairing_id=exchange.pairing_id,
        points=exchange.points,
        left_values=exchange.left_values,
        right_values=exchange.right_values,
        left_target=exchange.left_target,
        right_target=exchange.right_target,
        quantity_id="flux",
    )
    traces = phx.solver.SchwarzTraceState(
        (trace_state.exchanges[0], incompatible), sweep=1
    )
    state = eqx.tree_at(lambda value: value.trace_state, partial.state, traces)
    with pytest.raises(ValueError, match="quantity"):
        phx.solver.solve_functional_decomposition(prepared, optax.sgd(0.1), state=state)


@pytest.mark.strict_jax
def test_fixed_float_jacobi_targets_keep_existing_relaxation() -> None:
    problem = _broken_problem()
    prepared = phx.solver.prepare_functional_decomposition(
        problem,
        phx.solver.FunctionalDecompositionPlan(
            phx.solver.SchwarzDecompositionTraining(1, 1, sweep="jacobi", relaxation=0.5),
            trace_points=6,
        ),
    )
    initial = phx.solver.capture_schwarz_trace_state(
        problem, problem.functions, prepared.trace_batches, sweep=0
    )
    result = phx.solver.solve_functional_decomposition(prepared, optax.sgd(0.1), jit=True)
    raw = phx.solver.capture_schwarz_trace_state(
        problem, result.state.functions, prepared.trace_batches, sweep=1
    )
    np.testing.assert_allclose(
        _trace_targets(result.state.trace_state),
        0.5 * _trace_targets(initial) + 0.5 * _trace_targets(raw),
        rtol=1e-13,
        atol=1e-13,
    )
    assert result.state.previous_increment is None
    assert result.state.omega is None
