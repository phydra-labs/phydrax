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
    MeshfreeCoupledConservationProblem,
    prepare_meshfree_conservation_solve,
    prepare_meshfree_coupled_conservation_solve,
)
from phydrax.discretization.meshfree._constitutive import (
    AbstractCoupledEdgeConstitutiveLaw,
    EdgeFeatureField,
    EdgeFrameFeatures,
    LipschitzCoupledEdgeFlux,
    LipschitzEdgeFlux,
    MonotoneCoupledEdgeFlux,
    MonotoneEdgeConductance,
    O3EdgeInvariants,
    O3EdgeNetwork,
)
from phydrax.discretization.meshfree._coverage import EdgeFeatureCoverage
from phydrax.discretization.meshfree._exterior import (
    MeshfreeCoercivityPolicy,
    MeshfreeExteriorCalculusPlan,
    PreparedMeshfreeExteriorCalculus,
)
from phydrax.nn.layers import Linear
from phydrax.nn.models import PartiallyInputConvexNetwork
from phydrax.nn.operator.representations import O3Representation
from phydrax.nonlinear import NonlinearStatus, NonlinearTermination
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


def test_unassessed_background_keeps_converged_root_without_property_claims() -> None:
    side = np.arange(5, dtype=np.float64) / 4
    points = np.stack(np.meshgrid(side, side, indexing="ij"), axis=-1).reshape(-1, 2)
    ring = np.any((points == 0) | (points == 1), axis=1)
    exterior = MeshfreeExteriorCalculusPlan(
        points,
        1.01 / 4,
        80,
        node_volumes=np.full(25, 1 / 25),
        dirichlet=ring,
        coercivity_policy=MeshfreeCoercivityPolicy("unassessed"),
    ).prepare()
    model = Linear(in_size=1, out_size="scalar", activation=jax.nn.tanh, rwf=False)
    model = eqx.tree_at(
        lambda layer: (layer.weight, layer.bias),
        model,
        (jnp.asarray([[0.2]]), jnp.asarray([0.4])),
    )
    law = LipschitzEdgeFlux(model, background_conductance=1.0)
    problem = MeshfreeConservationProblem(exterior, law, source=1.0, boundary_values=0.0)
    solved = prepare_meshfree_conservation_solve(problem).solve()
    # The root and the background solve converge; no property audit was run.
    assert bool(solved.accepted)
    assert int(solved.primal_status) == int(NonlinearStatus.SUCCESS)
    assert float(solved.ledger.maximum_equation_defect) < 1e-9
    evidence = solved.constitutive_evidence
    assert not bool(evidence.background_assessed)
    assert int(evidence.background_factor_status) == -1
    assert bool(evidence.anchored)
    assert not bool(evidence.coercivity_certified)
    assert int(evidence.contraction_status) == int(
        MeshfreeContractionStatus.CONTRACT_UNCERTIFIED
    )


_COUPLED_STATE = O3Representation(scalars=1, vectors=1)
_TIGHT = NonlinearTermination(
    absolute_residual=1e-11, relative_residual=1e-11, maximum_steps=64
)


def _coupled_cloud() -> tuple[PreparedMeshfreeExteriorCalculus, EdgeFrameFeatures]:
    # 57 lattice nodes; the 19 nodes with complete axial stars are equations.
    exterior = prepare_recovery(size=57, dimension=3, seed=1).prepared.problem.exterior
    material = np.sin(np.asarray(exterior.points) @ np.asarray([1.0, 2.0, 3.0]))
    features = EdgeFrameFeatures(
        exterior.points,
        exterior.pairs,
        (EdgeFeatureField("external-material", "scalar"),),
        (material,),
        source_id=exterior.incidence.source.space_id,
    )
    return exterior, features


def _coupled_law(
    family: str, features: EdgeFrameFeatures
) -> AbstractCoupledEdgeConstitutiveLaw:
    even, odd = features.even.shape[1], features.odd.shape[1]
    if family == "monotone":
        invariants = O3EdgeInvariants(
            _COUPLED_STATE,
            quadratic_representation=O3Representation(scalars=2, vectors=1),
            linear_count=1,
            key=jax.random.key(0),
        )
        potential = PartiallyInputConvexNetwork(
            context_size=even + odd,
            convex_size=invariants.size,
            width_size=4,
            depth=1,
            input_monotonicity="nondecreasing",
            key=jax.random.key(1),
        )
        return MonotoneCoupledEdgeFlux(
            invariants, potential, background_conductance=0.7, odd_size=odd
        )
    network = O3EdgeNetwork(
        _COUPLED_STATE,
        O3Representation(scalars=2, vectors=1),
        context_size=even + odd,
        key=jax.random.key(2),
    )
    return LipschitzCoupledEdgeFlux(
        network, even_size=even, odd_size=odd, background_conductance=2.0
    )


def test_coupled_monotone_law_block_solve_ledgers_and_implicit_adjoint() -> None:
    exterior, features = _coupled_cloud()
    law = _coupled_law("monotone", features)
    count = exterior.points.shape[0]
    rng = np.random.default_rng(3)
    port = phx.ValuePort(
        "coupled-strength",
        event_shape=(),
        component_ids=("strength",),
        representation="positive-scalar",
        dimensions=(DIMENSIONLESS,),
    )
    binding = ParameterBinding(
        "strength",
        port,
        targets=(RuntimeInput("meshfree", "conductance"),),
        role="coefficient",
        derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
    )
    problem = MeshfreeCoupledConservationProblem(
        exterior,
        law,
        features=features,
        source=rng.normal(size=(count, 4)),
        boundary_values=rng.normal(size=(count, 4)),
        parameter_bindings=(binding,),
    )
    prepared = prepare_meshfree_coupled_conservation_solve(
        problem, parameters={"strength": jnp.asarray(1.0)}, termination=_TIGHT
    )
    free = prepared.residual.free_indices
    assert free.size == 19
    coefficient = jnp.asarray(1.3)
    solved = prepared.solve(parameters={"strength": coefficient})
    assert bool(solved.accepted)
    assert solved.state.shape == (count, 4)
    np.testing.assert_array_equal(
        np.asarray(solved.state)[np.asarray(problem.boundary_mask)],
        np.asarray(problem.boundary_values)[np.asarray(problem.boundary_mask)],
    )
    # Independent host ledger: integrated edge flux per component.
    pairs = np.asarray(exterior.pairs)
    flux = np.asarray(solved.edge_flux)
    integrated = np.zeros((count, 4))
    np.add.at(integrated, pairs[:, 0], flux)
    np.add.at(integrated, pairs[:, 1], -flux)
    sources = np.asarray(exterior.node_volumes)[:, None] * np.asarray(problem.source)
    np.testing.assert_allclose(
        integrated[np.asarray(free)], sources[np.asarray(free)], atol=1e-9
    )
    assert solved.ledger.balance_defect.shape == (4,)
    np.testing.assert_allclose(solved.ledger.balance_defect, 0.0, atol=1e-9)
    np.testing.assert_allclose(solved.ledger.internal_flux_sum, 0.0, atol=1e-10)
    evidence = solved.constitutive_evidence
    assert bool(evidence.background_assessed) and bool(evidence.coercivity_certified)
    np.testing.assert_allclose(evidence.strong_monotonicity_lower_bound, 0.7)
    cotangent = jnp.asarray(rng.normal(size=(count, 4)))

    def objective(value: jax.Array) -> jax.Array:
        return jnp.vdot(cotangent, prepared.solve(parameters={"strength": value}).state)

    implicit = jax.grad(objective)(coefficient)
    step = 1e-5
    central = (objective(coefficient + step) - objective(coefficient - step)) / (2 * step)
    np.testing.assert_allclose(implicit, central, rtol=1e-5, atol=1e-9)
    adjoint = prepared.adjoint(solved, cotangent, parameters={"strength": coefficient})
    assert bool(adjoint.accepted) and adjoint.value.shape == (19, 4)
    # dR/dc = (B^T kron I)(a F)/c at the root, so dJ/dc = -<lambda, dR/dc>.
    explicit = -jnp.vdot(adjoint.value, jnp.asarray(integrated)[free] / coefficient)
    np.testing.assert_allclose(implicit, explicit, rtol=1e-7, atol=1e-10)


def test_coupled_lipschitz_root_converges_without_contraction_certificate() -> None:
    exterior, features = _coupled_cloud()
    law = _coupled_law("lipschitz", features)
    count = exterior.points.shape[0]
    rng = np.random.default_rng(5)
    problem = MeshfreeCoupledConservationProblem(
        exterior,
        law,
        features=features,
        source=0.1 * rng.normal(size=(4,)),
        boundary_values=0.1 * rng.normal(size=(count, 4)),
    )
    # L/b > 1 and b - L < 0: no contraction or monotonicity is certified, yet
    # Newton converges to a root of the original equations for this data.
    solved = prepare_meshfree_coupled_conservation_solve(
        problem, termination=_TIGHT
    ).solve()
    assert bool(solved.accepted)
    assert float(jnp.max(solved.ledger.maximum_equation_defect)) < 1e-9
    evidence = solved.constitutive_evidence
    assert bool(evidence.background_assessed)
    assert float(evidence.contraction_bound) > 1.0
    assert int(evidence.contraction_status) == int(
        MeshfreeContractionStatus.CONTRACT_UNCERTIFIED
    )
    assert not bool(evidence.coercivity_certified)


def test_coupled_law_refuses_planar_frames_and_foreign_component_data() -> None:
    exterior, features = _coupled_cloud()
    law = _coupled_law("lipschitz", features)
    with pytest.raises(ValueError, match="packed component vector"):
        MeshfreeCoupledConservationProblem(
            exterior, law, features=features, source=np.zeros(3)
        )
    planar = prepare_recovery(size=9, dimension=2).prepared.problem
    planar_features = EdgeFrameFeatures(
        planar.exterior.points,
        planar.exterior.pairs,
        (EdgeFeatureField("external-material", "scalar"),),
        (np.zeros(9),),
        source_id=planar.exterior.incidence.source.space_id,
    )
    with pytest.raises(ValueError, match="three-dimensional"):
        MeshfreeCoupledConservationProblem(planar.exterior, law, features=planar_features)


def test_runtime_metric_weights_have_implicit_gradient_and_refuse_nonpositive() -> None:
    recovery = prepare_recovery(size=12, dimension=2, seed=5)
    original = recovery.prepared.problem
    exterior = original.exterior
    edges = exterior.lengths.size
    port = phx.ValuePort(
        "corrected-metric",
        event_shape=(edges,),
        component_ids=tuple(f"edge-{index}" for index in range(edges)),
        representation="positive-edge-metric",
    )
    binding = ParameterBinding(
        "metric",
        port,
        targets=(RuntimeInput("meshfree", "metric_weights"),),
        role="coefficient",
        derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
    )
    problem = MeshfreeConservationProblem(
        exterior,
        original.law,
        features=original.features,
        source=original.source,
        boundary_values=recovery.reference,
        parameter_bindings=(binding,),
        problem_id="corrected-metric-recovery",
    )
    base = exterior.metric_result.weights
    prepared = prepare_meshfree_conservation_solve(
        problem, parameters={"metric": base}, termination=_TIGHT
    )
    pattern = jnp.cos(jnp.arange(edges, dtype=jnp.float64))

    def state(theta: jax.Array) -> jax.Array:
        weights = base * jnp.exp(theta * pattern)
        return prepared.solve(parameters={"metric": weights}).state

    cotangent = jnp.linspace(0.2, 1.0, exterior.points.shape[0])
    point = jnp.asarray(0.1)
    gradient = jax.grad(lambda theta: jnp.vdot(cotangent, state(theta)))(point)
    step = 1e-5
    central = (
        jnp.vdot(cotangent, state(point + step))
        - jnp.vdot(cotangent, state(point - step))
    ) / (2 * step)
    np.testing.assert_allclose(gradient, central, rtol=1e-5, atol=1e-9)
    assert abs(float(gradient)) > 1e-8
    corrected = prepared.solve(parameters={"metric": base * jnp.exp(0.1 * pattern)})
    assert bool(corrected.accepted)
    # The coercivity factor is numerically refreshed at the runtime metric.
    evidence = corrected.constitutive_evidence
    assert bool(evidence.background_assessed) and bool(evidence.coercivity_certified)
    refused = prepared.solve(parameters={"metric": base.at[0].set(-1e-3)}, implicit=False)
    assert not bool(refused.accepted)
    assert not bool(refused.constitutive_evidence.positive_metric)
    assert bool(jnp.all(jnp.isnan(refused.state)))
