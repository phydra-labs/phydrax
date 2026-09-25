#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import equinox as eqx
import pytest

from phydrax import (
    AbstractConstructionCertificate,
    admit_regularity,
    authority_admits,
    branch_policy_contract,
    BranchDifferentiationPolicy,
    CapabilityEvidenceKind,
    CapabilityRequirement,
    ComponentAuthority,
    DERIVATIVE_UNSUPPORTED,
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    gradient_level_at_least,
    GradientLevel,
    ObjectiveKind,
    RegularityPolicy,
    resolve_gradient_level,
    SurfaceDerivative,
    weakest_level,
)


Surface = DerivativeSurface
Level = GradientLevel
Route = DerivativeRoute
Regularity = DerivativeRegularity

RELU = Regularity.piecewise_polynomial(continuity=0, degree_bound=1)
SQUARED_RELU = Regularity.piecewise_polynomial(continuity=1, degree_bound=2)
AFFINE = Regularity.smooth(degree_bound=1)
TANH = Regularity.smooth()
PERMISSIVE = RegularityPolicy(allow_almost_everywhere=True, allow_undeclared=True)


def _contract(*entries, route=Route.DIRECT, regularity=None, conditions=(), outputs=()):
    return DerivativeContract(
        tuple(SurfaceDerivative(surface, level) for surface, level in entries),
        route=route,
        regularity=regularity,
        conditions=conditions,
        nondifferentiable_outputs=outputs,
    )


def _request(*surfaces, order=1, authority=None):
    return DifferentiationRequest(surfaces, order=order, authority=authority)


def test_conditions_resolve_before_levels_are_compared():
    assert weakest_level((Level.SMOOTH, Level.ALMOST_EVERYWHERE)) is (
        Level.ALMOST_EVERYWHERE
    )
    assert weakest_level((Level.CONDITIONAL, Level.SMOOTH, Level.NONE)) is Level.NONE

    assert resolve_gradient_level(Level.SMOOTH, conditions=("unique-root",)) is (
        Level.CONDITIONAL
    )
    assert not gradient_level_at_least(
        Level.SMOOTH, Level.ALMOST_EVERYWHERE, conditions=("unique-root",)
    )
    assert gradient_level_at_least(
        Level.SMOOTH,
        Level.SMOOTH,
        conditions=("unique-root",),
        satisfied=("unique-root", "other"),
    )
    assert not gradient_level_at_least(
        Level.CONDITIONAL, Level.ALMOST_EVERYWHERE, satisfied=("unique-root",)
    )
    assert gradient_level_at_least(Level.CONDITIONAL, Level.CONDITIONAL)
    assert resolve_gradient_level(Level.NONE, conditions=("c",)) is Level.NONE

    with pytest.raises(TypeError):
        weakest_level(("smooth",))
    with pytest.raises(ValueError):
        weakest_level(())


def test_regularity_worked_cases():
    relu_linear = RELU.compose(AFFINE)
    assert relu_linear.admits_order(1) == (Level.ALMOST_EVERYWHERE, ())
    assert relu_linear.admits_order(2) == (Level.NONE, ())

    relu_tanh = RELU.compose(TANH)
    assert (relu_tanh.continuity, relu_tanh.pieces) == (0, "smooth")
    assert relu_tanh.admits_order(2) == (
        Level.ALMOST_EVERYWHERE,
        ("singular-part-ignored",),
    )

    squared = AFFINE.compose(SQUARED_RELU).compose(AFFINE).compose(SQUARED_RELU)
    assert (squared.continuity, squared.degree_bound) == (1, 4)
    assert squared.admits_order(1)[0] is Level.SMOOTH
    assert squared.admits_order(3) == (
        Level.ALMOST_EVERYWHERE,
        ("singular-part-ignored",),
    )
    assert squared.admits_order(5)[0] is Level.NONE

    elu = Regularity.piecewise_smooth(continuity=0)
    assert elu.admits_order(1) == (Level.ALMOST_EVERYWHERE, ())
    assert elu.admits_order(2) == (Level.ALMOST_EVERYWHERE, ("singular-part-ignored",))

    hard_tree = Regularity.piecewise_polynomial(continuity=-1, degree_bound=0)
    assert hard_tree.admits_order(1) == (Level.NONE, ())
    assert Regularity.discontinuous().admits_order(1) == (Level.NONE, ())


def test_differentiated_regularity_carries_the_admitted_conditions():
    elu = Regularity.piecewise_smooth(continuity=0, conditions=("interior",))
    assert elu.admits_order(2) == (
        Level.ALMOST_EVERYWHERE,
        ("interior", "singular-part-ignored"),
    )

    second = elu.differentiate(2)
    assert second.conditions == ("interior", "singular-part-ignored")
    assert elu.differentiate(1).conditions == ("interior",)
    assert elu.differentiate(1).differentiate(1).conditions == second.conditions
    assert TANH.differentiate(3).conditions == ()
    with pytest.raises(ValueError, match="degenerate"):
        RELU.differentiate(2)


def test_regularity_algebra_bounds_degree_and_continuity():
    cubic_c0 = Regularity.piecewise_polynomial(continuity=0, degree_bound=3)
    added = SQUARED_RELU.add(cubic_c0)
    assert (added.continuity, added.pieces, added.degree_bound) == (0, "polynomial", 3)
    product = SQUARED_RELU.multiply(cubic_c0)
    assert (product.continuity, product.degree_bound) == (0, 5)
    assert RELU.compose(SQUARED_RELU).degree_bound == 2

    smooth_stage = TANH.compose(RELU)
    assert (smooth_stage.continuity, smooth_stage.pieces) == (0, "smooth")
    assert smooth_stage.degree_bound is None
    assert RELU.add(Regularity.discontinuous()).continuity == -1
    assert RELU.add(Regularity.discontinuous()).admits_order(1)[0] is Level.NONE

    constrained = RELU.add(Regularity.smooth(conditions=("interior",), support="box"))
    assert constrained.conditions == ("interior",)
    assert constrained.support == "box"
    with pytest.raises(ValueError, match="incompatible"):
        constrained.add(Regularity.smooth(support="ball"))

    # A C^1 piecewise-linear map is globally linear, so its declaration is smooth.
    assert Regularity.piecewise_polynomial(continuity=1, degree_bound=1) == AFFINE
    assert Regularity.smooth(degree_bound=1).admits_order(3)[0] is Level.SMOOTH

    with pytest.raises(ValueError):
        Regularity(continuity=0, pieces="smooth", degree_bound=2)
    with pytest.raises(TypeError):
        Regularity.piecewise_polynomial(continuity=True, degree_bound=1)
    with pytest.raises(ValueError):
        RELU.admits_order(0)


def test_contract_declaration_is_canonical():
    first = _contract(
        (Surface.MODEL_PARAMETER, Level.SMOOTH),
        (Surface.INPUT, Level.SMOOTH),
        (Surface.FIT_TARGETS, Level.NONE),
        conditions=("b", "a", "a"),
    )
    second = _contract(
        (Surface.INPUT, Level.SMOOTH),
        (Surface.MODEL_PARAMETER, Level.SMOOTH),
        conditions=("a", "b"),
    )
    assert first == second
    assert first.contract_id == second.contract_id
    assert [entry.surface for entry in first.surfaces] == [
        Surface.INPUT,
        Surface.MODEL_PARAMETER,
    ]
    assert first.level(Surface.FIT_TARGETS) is Level.NONE

    owned_none = _contract((Surface.MODEL_PARAMETER, Level.NONE))
    assert owned_none.contract_id != _contract().contract_id
    assert (
        first.contract_id
        != _contract(
            (Surface.INPUT, Level.SMOOTH),
            (Surface.MODEL_PARAMETER, Level.SMOOTH),
            conditions=("a", "b"),
            route=Route.IMPLICIT,
        ).contract_id
    )

    with pytest.raises(ValueError, match="once"):
        _contract((Surface.INPUT, Level.SMOOTH), (Surface.INPUT, Level.NONE))
    with pytest.raises(TypeError):
        _contract((Surface.INPUT, "smooth"))


def test_meet_reproduces_artifact_owned_surface_union():
    native = _contract(
        (Surface.PHYSICAL_PARAMETER, Level.SMOOTH),
        (Surface.STORED_VALUES, Level.SMOOTH),
        (Surface.INPUT, Level.SMOOTH),
        (Surface.MODEL_PARAMETER, Level.SMOOTH),
        regularity=Regularity.smooth(),
    )
    constant = _contract()
    combined = native.meet(constant)

    assert combined.level(Surface.PHYSICAL_PARAMETER) is Level.NONE
    assert combined.level(Surface.STORED_VALUES) is Level.NONE
    assert combined.level(Surface.INPUT) is Level.NONE
    assert combined.level(Surface.MODEL_PARAMETER) is Level.SMOOTH
    assert combined.regularity is None
    assert constant.meet(constant).surfaces == ()


def test_meet_takes_weakest_capability_and_owner_levels():
    first = _contract(
        (Surface.INPUT, Level.SMOOTH),
        (Surface.MODEL_PARAMETER, Level.SMOOTH),
        regularity=TANH,
        outputs=("label",),
    )
    second = _contract(
        (Surface.INPUT, Level.ALMOST_EVERYWHERE),
        (Surface.MODEL_PARAMETER, Level.ALMOST_EVERYWHERE),
        (Surface.STOCHASTIC_REALIZATION, Level.SMOOTH),
        regularity=RELU,
        outputs=("index",),
    )
    combined = first.meet(second)

    assert combined.level(Surface.INPUT) is Level.ALMOST_EVERYWHERE
    assert combined.level(Surface.MODEL_PARAMETER) is Level.ALMOST_EVERYWHERE
    assert combined.level(Surface.STOCHASTIC_REALIZATION) is Level.SMOOTH
    assert combined.regularity == TANH.add(RELU)
    assert combined.nondifferentiable_outputs == ("index", "label")
    assert combined.route is Route.DIRECT


def test_route_mismatch_stops_unless_composition_route_is_declared():
    direct = _contract((Surface.MODEL_PARAMETER, Level.SMOOTH))
    implicit = _contract((Surface.MODEL_PARAMETER, Level.SMOOTH), route=Route.IMPLICIT)

    stopped = direct.meet(implicit)
    assert stopped.route is Route.STOPPED
    assert "mixed-derivative-routes:direct,implicit" in stopped.conditions
    assert stopped.level(Surface.MODEL_PARAMETER) is Level.SMOOTH
    assert direct.compose(implicit).route is Route.STOPPED

    request = _request(Surface.MODEL_PARAMETER)
    refused = stopped.admit(request, policy=PERMISSIVE)
    assert not refused.supported
    assert refused.status == DERIVATIVE_UNSUPPORTED
    assert refused.levels == (Level.NONE,)
    assert refused.reasons == ("route-stopped",)
    with pytest.raises(ValueError, match="route-stopped"):
        stopped.require(request)
    with pytest.raises(ValueError, match="route-stopped"):
        direct.compose(implicit).require(request)

    declared = direct.meet(implicit, composition_route=Route.UNROLLED)
    assert declared.route is Route.UNROLLED
    assert declared.conditions == ()
    assert declared.require(request).supported


def test_stopped_route_admits_no_request_whatever_its_declared_levels():
    stopped = DerivativeContract.smooth(
        (Surface.INPUT, Surface.MODEL_PARAMETER, Surface.FIT_TARGETS),
        route=Route.STOPPED,
    )
    for surfaces in ((Surface.MODEL_PARAMETER,), (Surface.INPUT, Surface.FIT_TARGETS)):
        request = _request(*surfaces, authority=ComponentAuthority.SURROGATE)
        admission = stopped.admit(request, policy=PERMISSIVE)
        assert admission.status == DERIVATIVE_UNSUPPORTED
        assert admission.levels == (Level.NONE,) * len(surfaces)
        assert admission.reasons == ("route-stopped",)

    bound = admit_regularity(
        TANH, _request(Surface.MODEL_PARAMETER), route=Route.STOPPED, policy=PERMISSIVE
    )
    assert bound.levels == (Level.NONE,)
    assert bound.reasons == ("route-stopped",)


def test_compose_passes_upstream_derivatives_through_downstream_input():
    upstream = _contract(
        (Surface.INPUT, Level.SMOOTH),
        (Surface.MODEL_PARAMETER, Level.SMOOTH),
        (Surface.PHYSICAL_PARAMETER, Level.SMOOTH),
        regularity=RELU,
    )
    smooth_downstream = _contract(
        (Surface.INPUT, Level.SMOOTH),
        (Surface.PHYSICAL_PARAMETER, Level.SMOOTH),
        (Surface.MODEL_STATE, Level.ALMOST_EVERYWHERE),
        regularity=AFFINE,
    )
    pipeline = upstream.compose(smooth_downstream)
    assert pipeline.level(Surface.INPUT) is Level.SMOOTH
    assert pipeline.level(Surface.MODEL_PARAMETER) is Level.SMOOTH
    assert pipeline.level(Surface.PHYSICAL_PARAMETER) is Level.SMOOTH
    assert pipeline.level(Surface.MODEL_STATE) is Level.ALMOST_EVERYWHERE
    assert pipeline.regularity == RELU.compose(AFFINE)

    blocking = _contract((Surface.MODEL_STATE, Level.SMOOTH))
    blocked = upstream.compose(blocking)
    assert blocked.level(Surface.INPUT) is Level.NONE
    assert blocked.level(Surface.PHYSICAL_PARAMETER) is Level.NONE
    # Upstream still owns its parameters; their derivatives cannot cross the
    # downstream input, while downstream-owned state keeps its own level.
    assert Surface.MODEL_PARAMETER in {entry.surface for entry in blocked.surfaces}
    assert blocked.level(Surface.MODEL_PARAMETER) is Level.NONE
    assert blocked.level(Surface.MODEL_STATE) is Level.SMOOTH

    # Parallel combination has no passage, so the upstream parameters survive.
    assert upstream.meet(blocking).level(Surface.MODEL_PARAMETER) is Level.SMOOTH


def test_authority_admission_table():
    expected = {
        ComponentAuthority.ACCELERATOR: {
            (Route.UNROLLED, ObjectiveKind.ALGORITHMIC_WORK),
            (Route.DIRECT, ObjectiveKind.SUPERVISED_PROXY),
        },
        ComponentAuthority.DISCRETIZATION: {
            (Route.UNROLLED, ObjectiveKind.ROLLOUT),
            (Route.DIRECT, ObjectiveKind.ROLLOUT),
            (Route.IMPLICIT, ObjectiveKind.SOLUTION_MAP),
            (Route.DIRECT, ObjectiveKind.SUPERVISED_PROXY),
            (Route.DIRECT, ObjectiveKind.PHYSICAL_RESIDUAL),
        },
        ComponentAuthority.MODEL: {
            (Route.DIRECT, ObjectiveKind.DATA_FIT),
            (Route.DIRECT, ObjectiveKind.PHYSICAL_RESIDUAL),
            (Route.DIRECT, ObjectiveKind.SUPERVISED_PROXY),
            (Route.IMPLICIT, ObjectiveKind.SOLUTION_MAP),
            (Route.UNROLLED, ObjectiveKind.ROLLOUT),
            (Route.EXTERNAL_ADJOINT, ObjectiveKind.SOLUTION_MAP),
            (Route.EXTERNAL_ADJOINT, ObjectiveKind.ROLLOUT),
        },
        ComponentAuthority.SURROGATE: {
            (Route.DIRECT, ObjectiveKind.PHYSICAL_RESIDUAL),
            (Route.DIRECT, ObjectiveKind.DATA_FIT),
            (Route.UNROLLED, ObjectiveKind.ROLLOUT),
        },
        ComponentAuthority.DECISION: {
            (Route.DIRECT, ObjectiveKind.ROLLOUT),
            (Route.UNROLLED, ObjectiveKind.ROLLOUT),
            (Route.RELAXED, ObjectiveKind.ROLLOUT),
            (Route.DIRECT, ObjectiveKind.SUPERVISED_PROXY),
        },
    }
    for authority, route, kind in itertools.product(
        ComponentAuthority, Route, ObjectiveKind
    ):
        assert authority_admits(authority, route, kind) is (
            (route, kind) in expected[authority]
        )
    assert not any(
        authority_admits(ComponentAuthority.ACCELERATOR, Route.IMPLICIT, kind)
        for kind in ObjectiveKind
    )
    with pytest.raises(TypeError):
        authority_admits("model", Route.DIRECT, ObjectiveKind.DATA_FIT)


def test_proven_degeneracy_is_always_rejected():
    request = _request(Surface.INPUT, order=2)
    admission = admit_regularity(
        RELU.compose(AFFINE), request, route=Route.DIRECT, policy=PERMISSIVE
    )
    assert not admission.supported
    assert admission.reasons == ("regularity-degenerate",)
    assert admission.levels == (Level.NONE,)


def test_almost_everywhere_regularity_needs_owner_permission():
    request = _request(Surface.INPUT, Surface.MODEL_PARAMETER, order=2)
    regularity = RELU.compose(TANH)

    refused = admit_regularity(
        regularity, request, route=Route.DIRECT, policy=RegularityPolicy()
    )
    assert refused.reasons == ("almost-everywhere-not-allowed",)
    assert refused.level(Surface.INPUT) is Level.NONE
    assert refused.level(Surface.MODEL_PARAMETER) is Level.SMOOTH

    allowed = admit_regularity(
        regularity,
        request,
        route=Route.DIRECT,
        policy=RegularityPolicy(allow_almost_everywhere=True),
    )
    assert allowed.supported
    assert allowed.level(Surface.INPUT) is Level.ALMOST_EVERYWHERE
    assert allowed.conditions == ("singular-part-ignored",)


def test_undeclared_regularity_depends_on_authority_and_route():
    def admit(authority, policy=RegularityPolicy(), route=Route.DIRECT):
        return admit_regularity(
            None, _request(Surface.INPUT, authority=authority), route=route, policy=policy
        )

    eager = admit(None)
    assert eager.supported
    assert eager.conditions == ("regularity-undeclared",)
    assert not gradient_level_at_least(
        eager.level(Surface.INPUT), Level.SMOOTH, conditions=eager.conditions
    )
    assert admit(None, route=Route.IMPLICIT).supported

    for authority in (ComponentAuthority.MODEL, ComponentAuthority.DISCRETIZATION):
        rejected = admit(authority, PERMISSIVE)
        assert not rejected.supported
        assert rejected.reasons == ("regularity-undeclared",)

    for authority in (
        ComponentAuthority.SURROGATE,
        ComponentAuthority.ACCELERATOR,
        ComponentAuthority.DECISION,
    ):
        assert not admit(authority).supported
        assert admit(authority, PERMISSIVE).supported
        assert not admit(authority, PERMISSIVE, Route.IMPLICIT).supported

    parameters_only = admit_regularity(
        None,
        _request(Surface.MODEL_PARAMETER, authority=ComponentAuthority.MODEL),
        route=Route.DIRECT,
        policy=RegularityPolicy(),
    )
    assert parameters_only.supported
    assert parameters_only.conditions == ()


def test_implicit_routes_need_classical_c1_or_branch_margin():
    request = _request(Surface.MODEL_PARAMETER, authority=ComponentAuthority.MODEL)
    policy = RegularityPolicy(allow_almost_everywhere=True)

    kinked = admit_regularity(RELU, request, route=Route.IMPLICIT, policy=policy)
    assert kinked.reasons == ("implicit-requires-c1",)
    assert kinked.levels == (Level.NONE,)

    margin = Regularity.piecewise_polynomial(
        continuity=0, degree_bound=1, conditions=("branch-margin",)
    )
    assert admit_regularity(
        margin, request, route=Route.IMPLICIT, policy=policy
    ).supported
    assert admit_regularity(
        SQUARED_RELU, request, route=Route.IMPLICIT, policy=RegularityPolicy()
    ).supported


def test_contract_admission_combines_levels_and_regularity():
    contract = _contract(
        (Surface.INPUT, Level.SMOOTH),
        (Surface.MODEL_PARAMETER, Level.ALMOST_EVERYWHERE),
        (Surface.FIT_FEATURES, Level.CONDITIONAL),
        route=Route.DIRECT,
        regularity=RELU,
        conditions=("fixed-active-set",),
        outputs=("argmax",),
    )
    fit = contract.admit(_request(Surface.FIT_FEATURES))
    assert fit.supported
    assert fit.levels == (Level.CONDITIONAL,)
    assert fit.conditions == ("fixed-active-set",)
    assert fit.nondifferentiable_outputs == ("argmax",)

    permissive = contract.admit(
        _request(Surface.INPUT, Surface.MODEL_PARAMETER), policy=PERMISSIVE
    )
    assert permissive.levels == (Level.ALMOST_EVERYWHERE, Level.ALMOST_EVERYWHERE)
    assert permissive.status == "derivative-supported"

    with pytest.raises(ValueError, match=DERIVATIVE_UNSUPPORTED) as error:
        contract.require(_request(Surface.INPUT, Surface.FIT_TARGETS), policy=PERMISSIVE)
    assert "fit-targets" in str(error.value)
    unsupported = contract.admit(_request(Surface.INPUT, Surface.FIT_TARGETS))
    assert unsupported.status == DERIVATIVE_UNSUPPORTED
    assert unsupported.reasons == (
        "almost-everywhere-not-allowed",
        "surface-unsupported:fit-targets",
    )
    with pytest.raises(ValueError, match="regularity-degenerate"):
        contract.require(_request(Surface.INPUT, order=2), policy=PERMISSIVE)


def test_branch_policies_map_to_distinct_canonical_contracts():
    surfaces = (Surface.PRIMAL_STATE, Surface.PHYSICAL_PARAMETER)
    contracts = {
        policy: branch_policy_contract(policy, surfaces=surfaces)
        for policy in BranchDifferentiationPolicy
    }
    assert len({contract.contract_id for contract in contracts.values()}) == len(
        BranchDifferentiationPolicy
    )

    request = _request(*surfaces, authority=ComponentAuthority.DISCRETIZATION)
    smooth = contracts[BranchDifferentiationPolicy.SMOOTH].admit(request)
    assert smooth.levels == (Level.SMOOTH, Level.SMOOTH)
    assert smooth.conditions == ()

    branchwise = contracts[BranchDifferentiationPolicy.BRANCHWISE]
    assert not branchwise.admit(request).supported
    assert branchwise.admit(request, policy=PERMISSIVE).conditions == (
        "executed-branch",
        "singular-part-ignored",
    )

    frozen = contracts[BranchDifferentiationPolicy.FROZEN_DECISION].admit(request)
    assert frozen.supported
    assert not gradient_level_at_least(
        frozen.level(Surface.PRIMAL_STATE), Level.SMOOTH, conditions=frozen.conditions
    )

    assert contracts[BranchDifferentiationPolicy.SMOOTH_SURROGATE].route is (
        Route.RELAXED
    )
    unsupported = contracts[BranchDifferentiationPolicy.UNSUPPORTED]
    assert unsupported.route is Route.STOPPED
    assert not unsupported.admit(request).supported


def test_capability_requirement_alternatives_and_safety():
    Kind = CapabilityEvidenceKind
    requirement = CapabilityRequirement(
        "holomorphic-map",
        ({Kind.CONSTRUCTED}, (Kind.DECLARED, Kind.RUNTIME_CHECKED)),
        safety_critical=True,
    )
    assert requirement.is_satisfied_by({Kind.CONSTRUCTED})
    assert requirement.is_satisfied_by((Kind.RUNTIME_CHECKED, Kind.DECLARED))
    assert not requirement.is_satisfied_by({Kind.DECLARED})
    assert not requirement.is_satisfied_by({Kind.RUNTIME_CHECKED})
    assert not requirement.is_satisfied_by(())

    declared = CapabilityRequirement("activation-regularity", ({Kind.DECLARED},))
    assert declared.is_satisfied_by({Kind.DECLARED})
    with pytest.raises(ValueError, match="declaration alone"):
        CapabilityRequirement(
            "activation-regularity", ({Kind.DECLARED},), safety_critical=True
        )
    with pytest.raises(ValueError):
        CapabilityRequirement("activation-regularity", ())
    with pytest.raises(TypeError):
        requirement.is_satisfied_by({"constructed"})


def test_construction_certificates_declare_capability_and_identity():
    class _Certificate(AbstractConstructionCertificate):
        certificate_id: str = eqx.field(static=True)

        def __init__(self, certificate_id):
            self.certificate_id = certificate_id

        @property
        def capability_id(self):
            return "input-convex"

    certificate = _Certificate("abc")
    assert (certificate.capability_id, certificate.certificate_id) == (
        "input-convex",
        "abc",
    )
    with pytest.raises(TypeError):
        AbstractConstructionCertificate()


def test_requests_are_validated_surface_sets():
    request = _request(Surface.MODEL_PARAMETER, Surface.INPUT)
    assert request.surfaces == (Surface.INPUT, Surface.MODEL_PARAMETER)
    assert request == _request(Surface.INPUT, Surface.MODEL_PARAMETER)
    with pytest.raises(ValueError):
        _request()
    with pytest.raises(ValueError):
        _request(Surface.INPUT, Surface.INPUT)
    with pytest.raises(TypeError):
        _request("input")
    with pytest.raises(TypeError):
        _request(Surface.INPUT, order=1.0)
    with pytest.raises(TypeError):
        _request(Surface.INPUT, authority="model")
