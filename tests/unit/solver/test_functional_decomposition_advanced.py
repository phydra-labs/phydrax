#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

import phydrax as phx


def _fixed_penalty(condition, count=8):
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


def _pou_problem():
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


def _broken_problem():
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
        family.ref(pairing.left_patch_id),
        family.ref(pairing.right_patch_id),
        pairing,
    )
    problem = phx.solver.FunctionalDecompositionProblem.broken(
        family,
        phx.solver.ScopedFunctionalTerm(
            _fixed_penalty(jump),
            phx.solver.PairScope(pairing.pairing_id),
        ),
    )
    return problem


def test_pou_colored_block_schedule_respects_fixed_patch_state():
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
    assert result.state.optimizer_states[1] is None


def test_relaxed_schwarz_owns_fixed_trace_state_and_reduces_defect():
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


def test_shared_functional_update_kernel_accepts_and_rejects_atomically():
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(1.0)
    term = _fixed_penalty(
        phx.conditions.Residual("u", domain.component(), lambda value: value)
    )
    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))
    paths = tuple(
        path
        for path in phx.nn.parameters.ParameterSubspace.array_leaf_paths(solver.functions)
        if ".func.value" in path
    )
    subspace = phx.nn.parameters.ParameterSubspace.from_leaf_paths(
        solver.functions,
        paths,
    )
    kernel = solver.update_kernel(optax.sgd(0.1), subspace, jit=True)

    state, evidence = kernel.advance(kernel.initialize(), key=jr.key(1))

    assert bool(evidence.accepted)
    assert state.step == 1
    np.testing.assert_allclose(state.functions["u"].func.value, 0.8)


def test_mortar_nitsche_and_augmented_interface_terms_are_physical():
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


def test_dense_local_curvature_uses_phydrax_linear_solve():
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
    assert abs(float(result.functions["u"].func.value)) < 1.0e-4
    assert result.approximation == "exact-local-dense-hessian"


def test_local_test_space_applies_discrete_riesz_geometry():
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


def test_arbitrary_hierarchy_trains_ordered_nonlinear_corrections():
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


def test_sharding_hybrid_and_deployment_roundtrip(tmp_path):
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
    restored = phx.solver.load_decomposition_artifact(
        tmp_path / "deployment",
        artifact,
    )

    assert sharded.evidence.verified
    assert eqx.tree_equal(restored.family.fields, artifact.family.fields)
    assert restored.artifact_id == artifact.artifact_id


def test_generalized_trace_aitken_and_bounded_asynchronous_schedule():
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
    relaxed, aitken = phx.solver.aitken_relax_trace_state(previous, proposed)
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

    assert relaxed.exchanges[0].quantity_id == "value"
    assert float(aitken.relaxation) == 1.0
    assert asynchronous.state.patch_revisions == (2, 2)
    assert asynchronous.state.maximum_observed_staleness == 1


def test_local_kfac_and_matrix_free_gauss_newton_curvature_routes():
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


def test_cycle_orders_and_real_device_collectives():
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


def test_multi_device_collectives_use_shard_map_global_semantics():
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
