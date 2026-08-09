#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _measure(
    points,
    weights,
    *,
    mass=None,
    mask=None,
    provenance="barycenter-test",
):
    points = jnp.asarray(points, dtype=float)
    weights = jnp.asarray(weights, dtype=float)
    return phx.integration.discrete(
        points,
        cx.Field(weights, dims=("atom",)),
        axes="atom",
        mask=(
            None
            if mask is None
            else cx.Field(jnp.asarray(mask, dtype=bool), dims=("atom",))
        ),
        normalized=True,
        target_mass=None if mass is None else jnp.asarray(mass),
        provenance=provenance,
    )


def _problem(
    measures,
    support,
    *,
    measure_weights=None,
    cost=None,
):
    if measure_weights is None:
        measure_weights = jnp.ones((len(measures),)) / len(measures)
    return phx.transport.fixed_support_barycenter_problem(
        tuple(measures),
        support,
        measure_weights=jnp.asarray(measure_weights),
        cost=(phx.transport.SquaredEuclideanCost() if cost is None else cost),
    )


def _solver(*, block_size=None, **kwargs):
    options = dict(
        max_iterations=500,
        min_iterations=1,
        tolerance=1e-9,
        check_every=2,
        block_size=block_size,
        early_stop=False,
        store_history=True,
    )
    options.update(kwargs)
    return phx.transport.SinkhornBarycenter(0.25, **options)


def test_identical_measures_preserve_symmetric_weights_mass_and_couplings():
    measure = _measure(
        [[-1.0, 0.5], [1.0, -0.5]],
        [0.5, 0.5],
        mass=3.0,
        provenance="identical",
    )
    problem = _problem((measure, measure), measure)
    result = _solver()(problem)
    target = result.as_target()

    assert result.converged
    assert jnp.allclose(result.probabilities, jnp.asarray([0.5, 0.5]), atol=1e-8)
    assert jnp.allclose(jnp.sum(result.padded_couplings(), axis=(1, 2)), 3.0)
    assert result.coupling(0).shape == (2, 2)
    assert target.target_mass == 3.0
    assert target.points.shape == (2, 2)
    assert result.provenance.approximate is False
    assert result.diagnostics.residual_history.ndim == 1
    assert result.diagnostics.per_measure_residual_history.shape[0] == 2


def test_dirac_translated_weighted_and_permuted_barycenters_are_physical():
    left = _measure([[0.0]], [1.0], provenance="left-dirac")
    right = _measure([[2.0]], [1.0], provenance="right-dirac")
    support = _measure([[0.0], [2.0]], [0.5, 0.5], provenance="declared")
    left_weighted = _solver()(
        _problem((left, right), support, measure_weights=[0.8, 0.2])
    )
    right_weighted = _solver()(
        _problem((right, left), support, measure_weights=[0.2, 0.8])
    )
    symmetric = _solver()(
        _problem(
            (
                _measure([[-2.0], [0.0]], [0.5, 0.5]),
                _measure([[2.0], [4.0]], [0.5, 0.5]),
            ),
            _measure([[0.0], [2.0]], [0.5, 0.5]),
        )
    )

    assert left_weighted.converged
    assert left_weighted.probabilities[0] > left_weighted.probabilities[1]
    assert jnp.allclose(
        left_weighted.probabilities,
        right_weighted.probabilities,
        atol=1e-9,
    )
    assert jnp.allclose(symmetric.probabilities, jnp.asarray([0.5, 0.5]), atol=1e-8)
    assert jnp.all(jnp.isfinite(left_weighted.per_measure_objectives))
    assert jnp.allclose(
        left_weighted.objective,
        jnp.sum(
            left_weighted.problem.measure_weights
            * left_weighted.per_measure_objectives
        ),
    )


def test_atom_permutation_does_not_change_declared_support_barycenter():
    first = _measure([[-1.0], [0.5], [2.0]], [0.2, 0.3, 0.5])
    permuted = _measure([[2.0], [-1.0], [0.5]], [0.5, 0.2, 0.3])
    second = _measure([[-0.5], [1.5]], [0.4, 0.6])
    support = _measure([[-1.0], [0.0], [1.0], [2.0]], [0.25] * 4)
    direct = _solver()(_problem((first, second), support))
    reordered = _solver()(_problem((permuted, second), support))

    assert direct.converged
    assert reordered.converged
    assert jnp.allclose(direct.probabilities, reordered.probabilities, atol=1e-9)
    assert jnp.allclose(direct.objective, reordered.objective, atol=1e-9)


def test_padded_unequal_support_and_dense_blockwise_execution_agree():
    first = _measure([[0.0], [1.0]], [0.25, 0.75], provenance="two-atoms")
    second = _measure(
        [[-1.0], [0.5], [2.0], [99.0]],
        [0.2, 0.3, 0.5, 7.0],
        mask=[True, True, True, False],
        provenance="masked-four-atoms",
    )
    support = _measure([[-0.5], [0.75], [1.75]], [0.2, 0.5, 0.3])
    problem = _problem((first, second), support, measure_weights=[0.3, 0.7])
    dense = _solver()(problem)
    blockwise = _solver(block_size=2)(problem)

    assert problem.measure_points.shape == (2, 4, 1)
    assert problem.measure_atom_counts == (2, 4)
    assert not problem.measure_active[1, -1]
    assert jnp.allclose(dense.probabilities, blockwise.probabilities)
    assert jnp.allclose(dense.padded_couplings(), blockwise.padded_couplings())
    assert jnp.allclose(dense.objective, blockwise.objective)
    assert dense.provenance.execution == "dense"
    assert blockwise.provenance.execution == "blockwise"
    assert jnp.all(dense.padded_couplings()[:, -1:, :] >= 0.0)
    assert jnp.allclose(dense.padded_couplings()[1, -1], 0.0)


def test_problem_rejects_invalid_mass_measure_weights_and_encoding():
    support = _measure([[0.0], [1.0]], [0.5, 0.5], mass=2.0)
    wrong_mass = _measure([[0.0], [1.0]], [0.5, 0.5], mass=3.0)
    correct = _measure([[0.0], [1.0]], [0.5, 0.5], mass=2.0)
    wrong_event = _measure([[0.0, 1.0], [1.0, 2.0]], [0.5, 0.5], mass=2.0)

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="common physical mass"):
        _problem((correct, wrong_mass), support)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="strictly positive"):
        _problem((correct, correct), support, measure_weights=[1.0, 0.0])
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="sum to one"):
        _problem((correct, correct), support, measure_weights=[0.4, 0.4])
    with pytest.raises(ValueError, match="feature size"):
        _problem((correct, wrong_event), support)
    compatible = phx.transport.FixedSupportBarycenterProblem(
        (correct, wrong_event),
        support,
        measure_weights=jnp.asarray([0.5, 0.5]),
        cost=phx.transport.SquaredEuclideanCost(),
        encoders=(None, lambda points: points[..., :1]),
    )
    assert compatible.feature_size == 1


def test_external_measure_realization_replay_is_supported():
    first = _measure([[0.0], [1.0]], [0.5, 0.5])
    second = phx.integration.weighted(
        jnp.asarray([[1.0], [2.0]]),
        jnp.log(jnp.asarray([0.5, 0.5])),
        normalized=True,
        sample_axes=0,
        provenance="weighted-barycenter-input",
    )
    support = _measure([[0.5], [1.5]], [0.5, 0.5])
    realization = phx.integration.materialize(first)
    problem = _problem((realization, second), phx.integration.materialize(support))
    solver = _solver()
    first_result = solver(problem)
    replay = solver(problem)

    assert jnp.array_equal(first_result.probabilities, replay.probabilities)
    assert jnp.array_equal(first_result.measure_potentials, replay.measure_potentials)
    assert first_result.problem.provenance.measures[0] == first.provenance
    assert first_result.problem.provenance.measures[1] == "weighted-barycenter-input"


def test_fixed_barycenter_is_jittable_vmappable_and_differentiable():
    support = _measure([[-1.0], [0.0], [1.0]], [1 / 3, 1 / 3, 1 / 3])
    solver = phx.transport.SinkhornBarycenter(
        0.5,
        max_iterations=40,
        tolerance=1e-7,
        check_every=2,
        early_stop=False,
    )

    def objective(shift):
        first = _measure([[-1.0 + shift], [1.0 + shift]], [0.4, 0.6])
        second = _measure([[-0.5], [0.5]], [0.5, 0.5])
        return solver(_problem((first, second), support)).objective

    compiled = jax.jit(objective)(jnp.asarray(0.2))
    mapped = jax.vmap(objective)(jnp.asarray([-0.2, 0.0, 0.2]))
    gradient = jax.grad(objective)(jnp.asarray(0.2))

    assert jnp.isfinite(compiled)
    assert mapped.shape == (3,)
    assert jnp.all(jnp.isfinite(mapped))
    assert jnp.isfinite(gradient)


def test_free_support_midpoint_retains_every_inner_solve_and_provenance():
    left = _measure([[0.0]], [1.0], provenance="left")
    right = _measure([[2.0]], [1.0], provenance="right")
    initialization = _measure([[0.25]], [1.0], provenance="explicit-initialization")
    problem = _problem((left, right), initialization)
    solver = phx.transport.FreeSupportBarycenter(
        _solver(max_iterations=100, tolerance=1e-10),
        max_iterations=4,
        tolerance=1e-8,
    )
    result = solver(problem)
    replay = solver(problem)

    assert result.converged
    assert jnp.allclose(
        result.barycenter.problem.support_points,
        jnp.asarray([[1.0]]),
        atol=1e-8,
    )
    assert len(result.inner_results) == 4
    assert result.provenance.retained_inner_solves == 4
    assert result.provenance.initialization == "explicit-initialization"
    assert result.provenance.local_optimization
    assert result.local_optimum
    assert jnp.array_equal(
        result.diagnostics.objective_history,
        replay.diagnostics.objective_history,
    )
    assert result.as_target().target_mass == 1.0


def test_free_support_reports_collapse_without_repairing_support():
    left = _measure([[0.0]], [1.0])
    right = _measure([[2.0]], [1.0])
    initialization = _measure([[-0.1], [0.1]], [0.5, 0.5])
    problem = _problem((left, right), initialization)
    result = phx.transport.FreeSupportBarycenter(
        _solver(max_iterations=100, tolerance=1e-10),
        max_iterations=3,
        collapse_tolerance=1e-8,
    )(problem)

    assert result.diagnostics.status == int(phx.transport.TransportStatus.SUPPORT_COLLAPSE)
    assert result.diagnostics.collapse_iteration == 1
    assert not result.converged
    assert jnp.array_equal(
        result.barycenter.problem.support_points,
        problem.support_points,
    )
    assert (
        phx.transport.status_message(phx.transport.TransportStatus.SUPPORT_COLLAPSE)
        == "free barycenter support collapsed"
    )


def test_free_support_rejects_nonquadratic_barycentric_costs():
    first = _measure([[0.0], [1.0]], [0.5, 0.5])
    second = _measure([[1.0], [2.0]], [0.5, 0.5])
    support = _measure([[0.25], [1.25]], [0.5, 0.5])
    problem = _problem(
        (first, second),
        support,
        cost=phx.transport.PeriodicSquaredEuclideanCost([4.0]),
    )
    solver = phx.transport.FreeSupportBarycenter(_solver(), max_iterations=2)

    with pytest.raises(TypeError, match="squared or weighted squared Euclidean"):
        solver(problem)


def test_fixed_solver_reports_declared_stagnation_status():
    first = _measure([[-2.0], [0.4], [3.0]], [0.1, 0.2, 0.7])
    second = _measure([[-1.0], [1.0]], [0.8, 0.2])
    support = _measure([[-2.5], [0.0], [2.5]], [0.2, 0.3, 0.5])
    result = phx.transport.SinkhornBarycenter(
        0.01,
        max_iterations=2,
        tolerance=0.0,
        check_every=1,
        stagnation_patience=1,
        stagnation_tolerance=0.999999,
        early_stop=True,
    )(_problem((first, second), support))

    assert result.diagnostics.status == int(
        phx.transport.TransportStatus.MARGINAL_STAGNATION
    )
    assert not result.converged


def test_barycenter_public_catalog_is_exactly_declared():
    symbols = {
        "BarycenterDiagnostics",
        "BarycenterProblemProvenance",
        "BarycenterProvenance",
        "BarycenterResult",
        "FixedSupportBarycenterProblem",
        "FreeSupportBarycenter",
        "FreeSupportBarycenterDiagnostics",
        "FreeSupportBarycenterProvenance",
        "FreeSupportBarycenterResult",
        "SinkhornBarycenter",
        "fixed_support_barycenter_problem",
        "require_barycenter_converged",
    }
    assert symbols <= set(phx.transport.__all__)
    assert all(getattr(phx.transport, symbol) is not None for symbol in symbols)
