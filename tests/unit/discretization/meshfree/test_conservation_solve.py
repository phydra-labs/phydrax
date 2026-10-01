# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from examples.meshfree_learned_edge_flux import prepare_recovery
from phydrax.discretization.meshfree._conservation_solve import (
    MeshfreeConservationProblem,
    MeshfreeContractionStatus,
    prepare_meshfree_conservation_solve,
)
from phydrax.discretization.meshfree._constitutive import (
    EdgeFeatureField,
    EdgeFrameFeatures,
    LipschitzEdgeFlux,
    MonotoneEdgeConductance,
)
from phydrax.discretization.meshfree._coverage import EdgeFeatureCoverage
from phydrax.nn.layers import Linear
from phydrax.nn.models import PartiallyInputConvexNetwork
from phydrax.nonlinear import NonlinearStatus
from phydrax.solver.coupling import ParameterBinding, RuntimeInput
from phydrax.units import DIMENSIONLESS


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


@pytest.mark.parametrize("dimension,size", [(1, 5), (2, 9), (3, 15)])
def test_manufactured_conservative_state_and_boundary_ledger(
    dimension: int, size: int
) -> None:
    recovery = prepare_recovery(size=size, dimension=dimension, seed=3)
    solved = jax.jit(
        lambda coefficient: recovery.prepared.solve(
            parameters={"constitutive-strength": coefficient}
        )
    )(jnp.asarray(recovery.truth_scale))
    assert bool(solved.accepted)
    np.testing.assert_allclose(solved.state, recovery.reference, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        solved.state[recovery.prepared.problem.boundary_mask],
        recovery.reference[recovery.prepared.problem.boundary_mask],
        atol=0,
    )
    np.testing.assert_allclose(
        solved.ledger.source_integral, solved.ledger.outward_boundary_flux, atol=1e-10
    )
    assert abs(float(solved.ledger.internal_flux_sum)) < 1e-12
    assert float(solved.ledger.maximum_equation_defect) < 1e-10
    assert bool(solved.constitutive_evidence.coercivity_certified)


def test_parameter_tangent_and_explicit_adjoint_agree_with_finite_difference() -> None:
    recovery = prepare_recovery(size=12, dimension=2, seed=5)
    prepared = recovery.prepared
    coefficient = jnp.asarray(1.3)

    def solve(value: jax.Array) -> jax.Array:
        return prepared.solve(parameters={"constitutive-strength": value}).state

    primal, tangent = jax.jit(
        lambda value: jax.jvp(solve, (value,), (jnp.ones_like(value),))
    )(coefficient)
    step = 1e-5
    central = (solve(coefficient + step) - solve(coefficient - step)) / (2 * step)
    np.testing.assert_allclose(tangent, central, rtol=3e-6, atol=1e-8)
    cotangent = jnp.linspace(0.1, 1.0, primal.size)
    solved = prepared.solve(parameters={"constitutive-strength": coefficient})
    adjoint = prepared.adjoint(
        solved, cotangent, parameters={"constitutive-strength": coefficient}
    )
    assert bool(adjoint.accepted)
    assert int(adjoint.primal_status) == int(NonlinearStatus.SUCCESS)
    free = prepared.residual.free_indices
    integrated = prepared.problem.exterior.incidence.transpose_mv(-solved.edge_flux)
    explicit_gradient = -jnp.vdot(adjoint.value, integrated[free] / coefficient)
    implicit_gradient = jax.jit(
        jax.grad(lambda value: jnp.vdot(cotangent, solve(value)))
    )(coefficient)
    np.testing.assert_allclose(
        implicit_gradient, explicit_gradient, rtol=2e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent), implicit_gradient, rtol=2e-8, atol=1e-10
    )


def test_model_bound_convex_potential_parameter_has_implicit_gradient_and_fixed_traits() -> (
    None
):
    recovery = prepare_recovery(size=9, dimension=2, seed=7)
    original = recovery.prepared.problem
    assert isinstance(original.law, MonotoneEdgeConductance)
    potential = original.law.potential
    assert isinstance(potential, PartiallyInputConvexNetwork)
    port = phx.ValuePort(
        "edge-law",
        event_shape=(),
        component_ids=("oriented-slope",),
        representation="oriented-edge-slope",
        dimensions=(DIMENSIONLESS,),
    )
    ports = phx.ModelPorts(inputs=(), outputs=(port,))
    parameter = ParameterBinding(
        "edge-law",
        port,
        targets=(RuntimeInput("meshfree", "law"),),
        role="coefficient",
        derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
    )
    initial = phx.bind_component(
        original.law, phx.ComponentAuthority.MODEL, owner_ports=ports
    )
    problem = MeshfreeConservationProblem(
        original.exterior,
        original.law,
        features=original.features,
        source=original.source / recovery.truth_scale,
        boundary_values=recovery.reference,
        parameter_bindings=(parameter,),
    )
    prepared = prepare_meshfree_conservation_solve(
        problem, parameters={"edge-law": initial}
    )

    def solve(raw_weight: jax.Array) -> jax.Array:
        updated_potential = eqx.tree_at(
            lambda model: model.state_layers[0].weight,
            potential,
            raw_weight.reshape((1, 1)),
        )
        learned = eqx.tree_at(
            lambda constitutive: constitutive.potential, original.law, updated_potential
        )
        component = phx.bind_component(
            learned, phx.ComponentAuthority.MODEL, owner_ports=ports
        )
        return prepared.solve(parameters={"edge-law": component}).state

    point = jnp.asarray(0.25)
    implicit_gradient = eqx.filter_jit(jax.grad(lambda value: jnp.sum(solve(value))))(
        point
    )
    step = 1e-5
    central = jnp.sum(solve(point + step) - solve(point - step)) / (2 * step)
    np.testing.assert_allclose(implicit_gradient, central, rtol=1e-5, atol=1e-8)
    assert abs(float(implicit_gradient)) > 1e-6
    changed_potential = eqx.tree_at(
        lambda model: model.context_lift.activation, potential, jax.nn.softplus
    )
    changed_traits = eqx.tree_at(
        lambda constitutive: constitutive.potential, original.law, changed_potential
    )
    changed = phx.bind_component(
        changed_traits, phx.ComponentAuthority.MODEL, owner_ports=ports
    )
    with pytest.raises(ValueError, match="metadata changed"):
        prepared.solve(parameters={"edge-law": changed})


def test_failed_primal_has_no_usable_public_state_or_implicit_gradient() -> None:
    recovery = prepare_recovery(size=7, dimension=2)
    failed = recovery.prepared.solve(
        parameters={"constitutive-strength": jnp.asarray(-0.5)}, implicit=False
    )
    assert not bool(failed.accepted)
    assert int(failed.primal_status) != int(NonlinearStatus.SUCCESS)
    assert bool(jnp.all(jnp.isnan(failed.state)))
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(lambda: failed.require_success())()

    def state(value: jax.Array) -> jax.Array:
        return recovery.prepared.solve(parameters={"constitutive-strength": value}).state

    coefficient = jnp.asarray(-0.5)
    tangent = eqx.filter_jit(
        lambda value: jax.jvp(state, (value,), (jnp.ones_like(value),))[1]
    )(coefficient)
    gradient = eqx.filter_jit(jax.grad(lambda value: jnp.sum(state(value))))(coefficient)
    assert bool(jnp.all(jnp.isnan(tangent)))
    assert bool(jnp.isnan(gradient))


def test_positive_metric_and_nominal_feature_geometry_are_required() -> None:
    recovery = prepare_recovery(size=9, dimension=2)
    original = recovery.prepared.problem
    signed = eqx.tree_at(
        lambda exterior: exterior.metric_result.weights,
        original.exterior,
        -original.exterior.metric_result.weights,
    )
    with pytest.raises(ValueError, match="positive metric"):
        MeshfreeConservationProblem(signed, original.law, features=original.features)
    other_geometry = EdgeFrameFeatures(
        original.exterior.points + 0.1,
        original.exterior.pairs,
        original.features.schema,
        (np.asarray(original.features.points[:, 0]),),
    )
    with pytest.raises(ValueError, match="nominal geometry"):
        MeshfreeConservationProblem(
            original.exterior, original.law, features=other_geometry
        )
    with pytest.raises(ValueError, match="nominal geometry"):
        MeshfreeConservationProblem(
            original.exterior, original.law, features=original.features.reoriented()
        )
    unrelated_source = EdgeFrameFeatures(
        original.exterior.points,
        original.exterior.pairs,
        original.features.schema,
        (np.zeros(original.source.size),),
        source_id="unrelated-scientific-cloud",
    )
    with pytest.raises(ValueError, match="nominal geometry"):
        MeshfreeConservationProblem(
            original.exterior, original.law, features=unrelated_source
        )


def test_equation_mask_prescribes_extra_rows_without_changing_conservation() -> None:
    recovery = prepare_recovery(size=15, dimension=2)
    original = recovery.prepared.problem
    mask = np.asarray(original.equation_mask).copy()
    indices = np.flatnonzero(mask)
    assert indices.size >= 2
    mask[indices[-1]] = False
    problem = MeshfreeConservationProblem(
        original.exterior,
        original.law,
        features=original.features,
        source=original.source / recovery.truth_scale,
        boundary_values=recovery.reference,
        equation_mask=mask,
    )
    prepared = prepare_meshfree_conservation_solve(problem)
    solved = prepared.solve()
    assert bool(solved.accepted)
    np.testing.assert_allclose(solved.state, recovery.reference, rtol=1e-8, atol=1e-10)
    assert abs(float(solved.ledger.balance_defect)) < 1e-10
    np.testing.assert_allclose(solved.state[~mask], recovery.reference[~mask], atol=0)


def test_coverage_refusal_is_preserved_as_unsuccessful_forward() -> None:
    recovery = prepare_recovery(size=9, dimension=2)
    original = recovery.prepared.problem
    outside = EdgeFrameFeatures(
        original.exterior.points,
        original.exterior.pairs,
        (EdgeFeatureField("external-material", "scalar"),),
        (np.full(9, 2.0),),
        source_id=original.exterior.incidence.source.space_id,
    )
    coverage = EdgeFeatureCoverage.fit(
        np.linspace(-1.0, 1.0, 31)[:, None],
        feature_names=outside.even_names,
        training_domain="bounded-material-training",
    )
    problem = MeshfreeConservationProblem(
        original.exterior,
        original.law,
        features=outside,
        source=0.0,
        boundary_values=0.0,
        coverage=coverage,
    )
    solved = prepare_meshfree_conservation_solve(problem).solve(implicit=False)
    assert not bool(solved.accepted)
    assert solved.coverage is not None and not bool(solved.coverage.admitted)
    assert bool(jnp.all(jnp.isnan(solved.state)))
    assert int(solved.primal_status) == int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE)


def test_background_solve_and_true_energy_contraction_distinguish_uncertified_bound() -> (
    None
):
    recovery = prepare_recovery(size=9, dimension=2)
    exterior = recovery.prepared.problem.exterior
    model = Linear(in_size=1, out_size="scalar", activation=jax.nn.tanh, rwf=False)
    model = eqx.tree_at(
        lambda layer: (layer.weight, layer.bias),
        model,
        (jnp.asarray([[0.2]]), jnp.asarray([0.4])),
    )
    law = LipschitzEdgeFlux(model, background_conductance=1.0)
    problem = MeshfreeConservationProblem(exterior, law, source=1.0, boundary_values=0.0)
    prepared = prepare_meshfree_conservation_solve(problem)
    solved = prepared.solve()
    assert bool(solved.accepted)
    assert solved.background is not None and bool(solved.background.successful)
    assert int(solved.constitutive_evidence.contraction_status) == int(
        MeshfreeContractionStatus.CERTIFIED
    )
    np.testing.assert_allclose(
        solved.constitutive_evidence.contraction_bound, 0.2, atol=1e-14
    )
    args = problem.runtime()
    free = prepared.residual.free_indices
    left, right = jnp.zeros(free.shape), jnp.full(free.shape, 0.3)
    step_left = prepared.background_solve(args, state=left)
    step_right = prepared.background_solve(args, state=right)
    background_operator = prepared.reduced.operator(exterior.metric_result.weights)
    norm_in = jnp.sqrt(jnp.vdot(right - left, background_operator.mv(right - left)))
    difference = step_right.value - step_left.value
    norm_out = jnp.sqrt(jnp.vdot(difference, background_operator.mv(difference)))
    assert float(norm_out) <= 0.2 * float(norm_in) + 1e-12
    high_model = eqx.tree_at(lambda layer: layer.weight, model, jnp.asarray([[2.0]]))
    high_law = LipschitzEdgeFlux(high_model, background_conductance=1.0)
    high = prepared.solve(law=high_law)
    assert int(high.constitutive_evidence.contraction_status) == int(
        MeshfreeContractionStatus.CONTRACT_UNCERTIFIED
    )
    assert not bool(high.constitutive_evidence.coercivity_certified)
