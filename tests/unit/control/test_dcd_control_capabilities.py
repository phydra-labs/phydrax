import jax
import jax.numpy as jnp

from phydrax.control._advanced_collocation import (
    audit_complementarity,
    ComplementarityConstraint,
    radau_collocation_defects,
)
from phydrax.control._continuous_certification import (
    AffineBernsteinPathEnvelope,
    CertifiedPathConstraint,
    certify_continuous_path_constraints,
    ControlSegmentInterpolant,
    LipschitzPathEnvelope,
)
from phydrax.control._global_certificate import (
    BoundedControlCertificatePlan,
    certify_bounded_control_optimum,
    ConvexTranscriptionRelaxation,
    LipschitzBoxControlRelaxation,
)
from phydrax.optim import BranchAndBoundPolicy, BranchAndBoundStatus
from phydrax.solver._radau_iia import RadauIIAMethod


def test_radau_tableau_and_linear_ode_defects_for_stages_one_through_four():
    for stages in range(1, 5):
        method = RadauIIAMethod(stages)
        assert method.order == 2 * stages - 1
        assert jnp.allclose(method.A[-1], method.b)

    method = RadauIIAMethod(2)
    times = jnp.asarray([0.0, 1.0])
    states = jnp.asarray([[0.0], [1.0]])
    rates = jnp.ones((1, 2, 1))
    controls = jnp.zeros((1, 1))
    defects = radau_collocation_defects(
        method,
        lambda time, state, control, args: jnp.ones_like(state),
        times,
        states,
        rates,
        controls,
    )
    assert defects.finite
    assert jnp.allclose(defects.stage_defects, 0.0)
    assert jnp.allclose(defects.endpoint_defects, 0.0)


def test_affine_and_lipschitz_continuous_path_certificates_are_interval_wide():
    interpolant = ControlSegmentInterpolant(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[[0.0], [0.8]]]),
        jnp.asarray([[[0.0]]]),
        case_shape=(),
        state_shape=(1,),
        control_shape=(1,),
        interpolant_id="linear-segment",
    )
    affine = CertifiedPathConstraint(
        lambda time, state, control, args: state[0] - 1.0,
        AffineBernsteinPathEnvelope(
            jnp.asarray([1.0]), jnp.asarray([0.0]), -1.0, envelope_id="affine"
        ),
        constraint_id="state-upper",
    )
    lipschitz = CertifiedPathConstraint(
        lambda time, state, control, args: state[0] - 1.0,
        LipschitzPathEnvelope(0.8, sample_count=3, envelope_id="lipschitz"),
        constraint_id="state-upper-lipschitz",
    )
    certificate = jax.jit(
        lambda: certify_continuous_path_constraints(interpolant, (affine, lipschitz))
    )()
    assert certificate.certified
    assert jnp.all(certificate.upper_bounds <= 0.0)


def test_complementarity_and_bounded_global_gap_evidence_fail_closed():
    constraint = ComplementarityConstraint(
        lambda value: (value, 1.0 - value),
        constraint_id="switch",
    )
    evidence = audit_complementarity(constraint, jnp.asarray(0.0))
    assert evidence.exact & evidence.valid

    objective = lambda value: value[0] ** 2
    relaxation = LipschitzBoxControlRelaxation(
        objective,
        2.0,
        relaxation_id="quadratic-lipschitz",
    )
    plan = BoundedControlCertificatePlan(
        objective,
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
        relaxation,
        BranchAndBoundPolicy(maximum_nodes=64, absolute_gap=0.2),
        lambda value: jnp.asarray(True),
        minimum_box_width=0.125,
        problem_id="bounded-quadratic",
    )
    certificate = certify_bounded_control_optimum(plan)
    assert certificate.relaxation_valid
    assert certificate.domain_covered
    assert certificate.epsilon_global
    assert certificate.status == int(BranchAndBoundStatus.GAP_REACHED)
    assert not certificate.exact
    assert certificate.global_lower_bound < certificate.objective
    assert jnp.allclose(
        certificate.absolute_gap,
        certificate.objective - certificate.global_lower_bound,
    )


def test_exact_convex_control_certificate_uses_the_certified_optimizer():
    objective = lambda value: (value[0] - 0.25) ** 2

    def solve(lower, upper):
        candidate = jnp.asarray([0.25])
        return (
            jnp.asarray(0.0),
            candidate,
            objective(candidate),
            jnp.asarray(0.0),
            jnp.asarray(True),
        )

    plan = BoundedControlCertificatePlan(
        objective,
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
        ConvexTranscriptionRelaxation(
            solve,
            gap_tolerance=1.0e-8,
            relaxation_id="exact-convex-quadratic",
        ),
        BranchAndBoundPolicy(),
        lambda value: jnp.asarray(True),
        problem_id="exact-convex-quadratic",
    )
    certificate = certify_bounded_control_optimum(plan)
    assert certificate.exact
    assert not certificate.epsilon_global
    assert jnp.allclose(certificate.incumbent, jnp.asarray([0.25]))
    assert jnp.allclose(certificate.objective, 0.0)
    assert jnp.allclose(certificate.global_lower_bound, 0.0)


def test_supplied_control_incumbent_closes_gap_without_a_tree_incumbent():
    objective = lambda value: value[0] ** 2
    plan = BoundedControlCertificatePlan(
        objective,
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
        LipschitzBoxControlRelaxation(
            objective,
            2.0,
            relaxation_id="supplied-incumbent-quadratic",
        ),
        BranchAndBoundPolicy(maximum_nodes=1, absolute_gap=0.8),
        lambda value: jnp.asarray(True),
        minimum_box_width=0.01,
        problem_id="supplied-incumbent-quadratic",
    )

    certificate = certify_bounded_control_optimum(plan, jnp.asarray([0.0]))

    assert certificate.epsilon_global
    assert certificate.domain_covered
    assert certificate.status == int(BranchAndBoundStatus.GAP_REACHED)
    assert jnp.allclose(certificate.incumbent, jnp.asarray([0.0]))
    assert jnp.isfinite(certificate.objective)
    assert jnp.allclose(certificate.objective, objective(certificate.incumbent))
    assert jnp.allclose(certificate.global_lower_bound, -0.75)
    assert jnp.allclose(certificate.absolute_gap, 0.75)


def test_better_supplied_control_incumbent_replaces_the_tree_upper_bound():
    objective = lambda value: (value[0] - 1.0) ** 2
    plan = BoundedControlCertificatePlan(
        objective,
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
        LipschitzBoxControlRelaxation(
            objective,
            4.0,
            relaxation_id="combined-incumbent-quadratic",
        ),
        BranchAndBoundPolicy(absolute_gap=3.5),
        lambda value: jnp.asarray(True),
        minimum_box_width=2.0,
        problem_id="combined-incumbent-quadratic",
    )

    certificate = certify_bounded_control_optimum(plan, jnp.asarray([1.0]))

    assert certificate.epsilon_global
    assert certificate.status == int(BranchAndBoundStatus.GAP_REACHED)
    assert jnp.allclose(certificate.incumbent, jnp.asarray([1.0]))
    assert jnp.allclose(certificate.objective, 0.0)
    assert jnp.allclose(certificate.global_lower_bound, -3.0)
    assert jnp.allclose(certificate.absolute_gap, 3.0)


def test_invalid_and_worse_supplied_control_incumbents_do_not_replace_tree_evidence():
    objective = lambda value: (value[0] - 0.25) ** 2
    plan = BoundedControlCertificatePlan(
        objective,
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
        LipschitzBoxControlRelaxation(
            objective,
            2.5,
            relaxation_id="incumbent-validation-quadratic",
        ),
        BranchAndBoundPolicy(),
        lambda value: ~jnp.isclose(value[0], 0.25),
        minimum_box_width=2.0,
        problem_id="incumbent-validation-quadratic",
    )

    for supplied in (
        jnp.asarray([jnp.nan]),
        jnp.asarray([2.0]),
        jnp.asarray([0.25]),
        jnp.asarray([0.75]),
        jnp.asarray([0.0, 0.0]),
    ):
        certificate = certify_bounded_control_optimum(plan, supplied)
        assert jnp.allclose(certificate.incumbent, jnp.asarray([0.0]))
        assert certificate.continuous_feasible
        assert jnp.allclose(certificate.objective, objective(certificate.incumbent))
