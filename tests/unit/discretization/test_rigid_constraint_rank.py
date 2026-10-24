#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from numpy.typing import NDArray

import phydrax as phx
from phydrax.discretization.particle._rigid_constraint_dynamics import (
    _constraint_operator,
    PreparedRigidConstraintDynamics,
    RigidConstraintDynamicsPlan,
    RigidConstraintRejectionReason,
    RigidConstraintSolverPlan,
)
from phydrax.discretization.particle._rigid_constraint_rank import (
    RigidConstraintRankAudit,
    RigidConstraintRankRoute,
    RigidConstraintRankStatus,
)
from phydrax.discretization.particle._rigid_joints import (
    _RigidMobileIncrement,
    RigidJointGraphPlan,
)
from phydrax.linalg.svd import SVDResourcePolicy


_FOREST = RigidConstraintRankAudit("certified-forest")


def _bodies(
    count: int, fixed: list[bool], dimension: int = 3
) -> tuple[jax.Array, phx.discretization.PreparedRigidBodySet]:
    ids = jnp.arange(100, 100 + count, dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        ids, jnp.ones((count,)), ambient_dimension=dimension
    ).prepare()
    inertia = (
        jnp.ones((count,))
        if dimension == 2
        else jnp.stack(tuple(jnp.eye(3) for _ in range(count)))
    )
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((count,), dtype=jnp.int32), inertia, fixed_mask=jnp.asarray(fixed)
    ).prepare(particles)
    return ids, bodies


def _reference(
    bodies: phx.discretization.PreparedRigidBodySet, position: NDArray[np.float64]
) -> phx.discretization.RigidBodyKinematics:
    position_ = jnp.asarray(position)
    if bodies.ambient_dimension == 2:
        orientation = jnp.zeros((bodies.capacity, 1))
        angular = jnp.zeros((bodies.capacity, 1))
    else:
        orientation = jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * bodies.capacity)
        angular = jnp.zeros_like(position_)
    return bodies.kinematics(
        position_, jnp.zeros_like(position_), orientation, angular
    )


def _line(count: int, dimension: int = 3) -> NDArray[np.float64]:
    position = np.zeros((count, dimension))
    position[:, 0] = np.arange(count, dtype=np.float64)
    return position


def _mixed_chain() -> tuple[
    phx.discretization.PreparedRigidBodySet,
    phx.discretization.RigidBodyKinematics,
    RigidJointGraphPlan,
]:
    """Fixed base, then ball, hinge, prismatic and distance joints in a chain."""
    ids, bodies = _bodies(5, [True, False, False, False, False])
    reference = _reference(bodies, _line(5))
    graph = RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            jnp.asarray([10]), ids[:1], ids[1:2], jnp.asarray([[0.5, 0.0, 0.0]])
        ),
        hinge=phx.discretization.HingeJointSetPlan(
            jnp.asarray([11]),
            ids[1:2],
            ids[2:3],
            jnp.asarray([[1.5, 0.0, 0.0]]),
            jnp.asarray([[0.0, 0.0, 1.0]]),
        ),
        prismatic=phx.discretization.PrismaticJointSetPlan(
            jnp.asarray([12]),
            ids[3:4],
            ids[2:3],
            jnp.asarray([[2.5, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0]]),
        ),
        distance=phx.discretization.DistanceJointSetPlan(
            jnp.asarray([13]),
            ids[3:4],
            ids[4:5],
            jnp.asarray([[3.0, 0.2, 0.0]]),
            jnp.asarray([[4.0, 0.0, 0.0]]),
        ),
    )
    return bodies, reference, graph


def _prepare(
    graph: RigidJointGraphPlan,
    bodies: phx.discretization.PreparedRigidBodySet,
    reference: phx.discretization.RigidBodyKinematics,
    audit: RigidConstraintRankAudit | None = None,
    *,
    rank_tolerance: float = 1.0e-10,
) -> PreparedRigidConstraintDynamics:
    solver = RigidConstraintSolverPlan(rank_audit=audit, rank_tolerance=rank_tolerance)
    return RigidConstraintDynamicsPlan(graph, solver=solver).prepare(bodies, reference)


def _perturbed(
    bodies: phx.discretization.PreparedRigidBodySet,
    reference: phx.discretization.RigidBodyKinematics,
    seed: int,
) -> phx.discretization.RigidBodyKinematics:
    rng = np.random.default_rng(seed)
    orientation = np.asarray(reference.orientation) + 0.1 * rng.normal(
        size=reference.orientation.shape
    )
    if bodies.ambient_dimension == 3:
        orientation /= np.linalg.norm(orientation, axis=-1, keepdims=True)
    return bodies.kinematics(
        reference.position + 0.05 * jnp.asarray(rng.normal(size=reference.position.shape)),
        reference.velocity,
        jnp.asarray(orientation),
        reference.angular_velocity,
    )


def _singular_extremes(matrix: NDArray[np.float64]) -> tuple[mp.mpf, mp.mpf]:
    with mp.workdps(50):
        values = mp.svd_r(mp.matrix(matrix.tolist()), compute_uv=False)
        spectrum = [values[index] for index in range(values.rows)]
        return min(spectrum), max(spectrum)


def _dense_jacobian(
    dynamics: PreparedRigidConstraintDynamics,
    kinematics: phx.discretization.RigidBodyKinematics,
) -> NDArray[np.float64]:
    """Forward-mode reference of the audited operator in packed row order."""
    joints = dynamics.joints
    operator = _constraint_operator(
        joints, dynamics.solver.characteristic_length, kinematics, dynamics.prepared_id
    )
    dimension = dynamics.bodies.ambient_dimension
    width = dimension + dynamics.bodies.angular_dimension

    def packed(flat: jax.Array) -> jax.Array:
        coordinates = flat.reshape(joints.mobile_count, width)
        increment = _RigidMobileIncrement(
            coordinates[:, :dimension], coordinates[:, dimension:]
        )
        return joints.pack_residuals(operator.mv(increment))

    zero = jnp.zeros((joints.mobile_count * width,))
    return np.asarray(jax.jacfwd(packed)(zero))


def test_forest_route_certifies_and_preserves_step() -> None:
    bodies, reference, graph = _mixed_chain()

    def gravity(
        time: jax.Array, kinematics: phx.discretization.RigidBodyKinematics, args: None
    ) -> phx.discretization.RigidBodyLoad:
        del time, args
        return phx.discretization.RigidBodyLoad(
            jnp.broadcast_to(jnp.asarray([0.0, -9.81, 0.0]), kinematics.position.shape),
            jnp.zeros_like(kinematics.angular_velocity),
        )

    results = []
    for audit in (None, _FOREST):
        solver = RigidConstraintSolverPlan(rank_audit=audit)
        dynamics = RigidConstraintDynamicsPlan(graph, solver=solver).prepare(
            bodies, reference, external_load=gravity, external_load_id="gravity"
        )
        state = dynamics.initialize_state(
            reference.position,
            reference.velocity,
            reference.orientation,
            reference.angular_velocity,
        )
        results.append(dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3)))
    general, forest = results
    rows = graph.constraint_count_for_dimension(3)
    assert bool(general.successful) and bool(forest.successful)
    np.testing.assert_array_equal(
        np.asarray(general.accepted_state.kinematics.position),
        np.asarray(forest.accepted_state.kinematics.position),
    )
    dense = general.evaluation.diagnostics.rank
    assert int(dense.status) == RigidConstraintRankStatus.FULL_ROW_RANK_MEASURED
    assert int(dense.executed_route) == RigidConstraintRankRoute.DENSE_SVD
    assert dense.certificate is None
    certified = forest.evaluation.diagnostics.rank
    assert int(certified.status) == RigidConstraintRankStatus.FULL_ROW_RANK_CERTIFIED
    assert int(certified.executed_route) == RigidConstraintRankRoute.FOREST_CERTIFICATE
    assert int(certified.rank_lower_bound) == int(certified.rank_upper_bound) == rows
    assert np.isnan(float(certified.measured_condition))
    assert certified.certificate is not None
    assert float(certified.certificate.condition_upper_bound) >= float(
        dense.measured_condition
    ) * (1.0 - 1.0e-12)


def test_endpoint_blocks_and_bounds_match_dense_jacobian() -> None:
    bodies, reference, graph = _mixed_chain()
    dynamics = _prepare(graph, bodies, reference, _FOREST)
    kinematics = _perturbed(bodies, reference, seed=4)
    joints = dynamics.joints
    operator = _constraint_operator(
        joints, dynamics.solver.characteristic_length, kinematics, dynamics.prepared_id
    )
    if dynamics.rank.forest is None:
        raise AssertionError("forest route was not admitted")
    left, right = joints.endpoint_jacobian_blocks(
        operator.mv, dynamics.rank.forest.body_colors, kinematics.position.dtype
    )
    width = 6
    mobile = np.asarray(joints.mobile_indices)
    vertex_of = {int(body): index for index, body in enumerate(mobile)}
    layout = joints.row_layout
    slots = np.asarray(layout.row_joint_slots)
    local = np.asarray(layout.row_local_indices)
    left_bodies, right_bodies = (np.asarray(value) for value in joints.layout_endpoints())
    assembled = np.zeros((layout.row_count, joints.mobile_count * width))
    for row in range(layout.row_count):
        slot = slots[row]
        for bodies_, blocks in ((left_bodies, left), (right_bodies, right)):
            body = int(bodies_[slot])
            if body in vertex_of:
                start = vertex_of[body] * width
                assembled[row, start : start + width] = np.asarray(
                    blocks[slot, local[row]]
                )
    dense = _dense_jacobian(dynamics, kinematics)
    np.testing.assert_allclose(assembled, dense, rtol=1.0e-13, atol=1.0e-14)
    evidence = dynamics.constraint_rank_evidence(kinematics)
    smallest, largest = _singular_extremes(dense)
    certificate = evidence.certificate
    assert certificate is not None and bool(certificate.certified)
    assert float(certificate.minimum_singular_value_lower_bound) <= smallest
    assert largest <= float(certificate.maximum_singular_value_upper_bound)
    assert int(certificate.numeric_version) == 1


def test_free_planar_forest_is_certified() -> None:
    ids, bodies = _bodies(5, [False] * 5, dimension=2)
    reference = _reference(bodies, _line(5, 2))
    graph = RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            jnp.asarray([1, 2]),
            ids[:2],
            ids[1:3],
            jnp.asarray([[0.5, 0.0], [1.5, 0.0]]),
        ),
        prismatic=phx.discretization.PrismaticJointSetPlan(
            jnp.asarray([3]), ids[2:3], ids[3:4], jnp.asarray([[2.5, 0.0]]),
            jnp.asarray([[1.0, 0.0]]),
        ),
    )
    dynamics = _prepare(graph, bodies, reference, _FOREST)
    evidence = dynamics.constraint_rank_evidence(_perturbed(bodies, reference, seed=7))
    assert int(evidence.status) == RigidConstraintRankStatus.FULL_ROW_RANK_CERTIFIED
    assert int(evidence.rank_lower_bound) == 6


@pytest.mark.parametrize("topology", ["duplicate-edge", "cycle", "two-fixed"])
def test_forest_admission_refuses_unsupported_topology(topology: str) -> None:
    match topology:
        case "duplicate-edge":
            ids, bodies = _bodies(3, [True, False, False])
            ball = phx.discretization.BallJointSetPlan(
                jnp.asarray([50, 51]),
                jnp.asarray([ids[0], ids[0]]),
                jnp.asarray([ids[1], ids[1]]),
                jnp.asarray([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            )
            message = "acyclic"
        case "cycle":
            ids, bodies = _bodies(3, [False, False, False])
            ball = phx.discretization.BallJointSetPlan(
                jnp.asarray([50, 51, 52]),
                ids,
                jnp.roll(ids, -1),
                jnp.asarray([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            )
            message = "acyclic"
        case "two-fixed":
            ids, bodies = _bodies(3, [True, False, True])
            ball = phx.discretization.BallJointSetPlan(
                jnp.asarray([50, 51]),
                ids[:2],
                ids[1:],
                jnp.asarray([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            )
            message = "fixed body"
        case _:
            raise AssertionError(topology)
    graph = RigidJointGraphPlan(ball=ball)
    reference = _reference(bodies, _line(3))
    with pytest.raises(ValueError, match=message):
        _prepare(graph, bodies, reference, _FOREST)
    _prepare(graph, bodies, reference)


def _ball_chain(
    count: int,
) -> tuple[
    phx.discretization.PreparedRigidBodySet,
    phx.discretization.RigidBodyKinematics,
    RigidJointGraphPlan,
]:
    ids, bodies = _bodies(count, [True] + [False] * (count - 1))
    reference = _reference(bodies, _line(count))
    anchors = np.zeros((count - 1, 3))
    anchors[:, 0] = np.arange(count - 1) + 0.5
    graph = RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            jnp.arange(10, 9 + count), ids[:-1], ids[1:], jnp.asarray(anchors)
        )
    )
    return bodies, reference, graph


def test_inconclusive_certificate_refuses_transactionally() -> None:
    bodies, reference, graph = _ball_chain(4)
    dynamics = _prepare(graph, bodies, reference, _FOREST, rank_tolerance=0.9)
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    evidence = result.evaluation.diagnostics.rank
    reasons = int(result.rejection_reasons)
    assert not bool(result.successful)
    assert int(evidence.status) == RigidConstraintRankStatus.INCONCLUSIVE
    assert int(evidence.rank_lower_bound) == 0
    assert int(evidence.rank_upper_bound) == graph.constraint_count_for_dimension(3)
    assert reasons & int(RigidConstraintRejectionReason.RANK_INCONCLUSIVE)
    assert not reasons & int(RigidConstraintRejectionReason.RANK_OR_CONDITION)
    np.testing.assert_array_equal(
        np.asarray(result.accepted_state.kinematics.position),
        np.asarray(state.kinematics.position),
    )


def test_admitted_dense_fallback_runs_only_after_inconclusive_certificate() -> None:
    bodies, reference, graph = _ball_chain(4)
    audit = RigidConstraintRankAudit(
        "certified-forest", dense_fallback=SVDResourcePolicy()
    )
    strict = _prepare(graph, bodies, reference, audit, rank_tolerance=0.9)
    measured = strict.constraint_rank_evidence(reference)
    assert int(measured.executed_route) == RigidConstraintRankRoute.FOREST_DENSE_FALLBACK
    assert int(measured.status) == RigidConstraintRankStatus.RANK_DEFICIENT_MEASURED
    assert int(measured.rank_lower_bound) == int(measured.rank_upper_bound) < 9
    assert np.isfinite(float(measured.measured_condition))
    assert not bool(measured.full_row_rank)
    ordinary = _prepare(graph, bodies, reference, audit)
    certified = ordinary.constraint_rank_evidence(reference)
    assert int(certified.executed_route) == RigidConstraintRankRoute.FOREST_CERTIFICATE
    assert int(certified.status) == RigidConstraintRankStatus.FULL_ROW_RANK_CERTIFIED
    assert np.isnan(float(certified.measured_condition))


def test_unaffordable_dense_fallback_is_refused_at_preparation() -> None:
    bodies, reference, graph = _ball_chain(4)
    audit = RigidConstraintRankAudit(
        "certified-forest",
        dense_fallback=SVDResourcePolicy(preparation_bytes=1, workspace_bytes=1),
    )
    with pytest.raises(ValueError, match="infeasible"):
        _prepare(graph, bodies, reference, audit)
    with pytest.raises(ValueError, match="workspace"):
        _prepare(
            graph,
            bodies,
            reference,
            RigidConstraintRankAudit("certified-forest", workspace_bytes=1),
        )


_FALLBACK = RigidConstraintRankAudit("certified-forest", dense_fallback=SVDResourcePolicy())


def _nonfinite(
    bodies: phx.discretization.PreparedRigidBodySet,
    reference: phx.discretization.RigidBodyKinematics,
) -> phx.discretization.RigidBodyKinematics:
    """A NaN orientation; ball rows depend on rotation, so the Jacobian is NaN.

    (A NaN position would not do: ball-joint Jacobians are position-independent.)
    """
    return bodies.kinematics(
        reference.position,
        reference.velocity,
        reference.orientation.at[2, 0].set(jnp.nan),
        reference.angular_velocity,
    )


def test_failed_dense_decomposition_is_an_inspectable_status() -> None:
    bodies, reference, graph = _ball_chain(4)
    rows = graph.constraint_count_for_dimension(3)
    broken = _nonfinite(bodies, reference)
    forest = _prepare(graph, bodies, reference, _FALLBACK)
    fallback = forest.constraint_rank_evidence(broken)
    assert fallback.certificate is not None
    assert not bool(fallback.certificate.certified)
    assert int(fallback.executed_route) == RigidConstraintRankRoute.FOREST_DENSE_FALLBACK
    assert int(fallback.status) == RigidConstraintRankStatus.DENSE_SVD_FAILED
    assert (int(fallback.rank_lower_bound), int(fallback.rank_upper_bound)) == (0, rows)
    assert not bool(fallback.full_row_rank)
    general = _prepare(graph, bodies, reference).constraint_rank_evidence(broken)
    assert int(general.status) == RigidConstraintRankStatus.DENSE_SVD_FAILED
    assert (int(general.rank_lower_bound), int(general.rank_upper_bound)) == (0, rows)


def test_batched_fallback_isolates_failed_cases() -> None:
    bodies, reference, graph = _ball_chain(4)
    rows = graph.constraint_count_for_dimension(3)
    dynamics = _prepare(graph, bodies, reference, _FALLBACK)
    cases = jax.tree.map(
        lambda good, bad: jnp.stack((good, bad)),
        reference,
        _nonfinite(bodies, reference),
    )
    evidence = jax.jit(jax.vmap(dynamics.constraint_rank_evidence))(cases)
    np.testing.assert_array_equal(
        np.asarray(evidence.status),
        [
            RigidConstraintRankStatus.FULL_ROW_RANK_CERTIFIED,
            RigidConstraintRankStatus.DENSE_SVD_FAILED,
        ],
    )
    np.testing.assert_array_equal(
        np.asarray(evidence.executed_route),
        [
            RigidConstraintRankRoute.FOREST_CERTIFICATE,
            RigidConstraintRankRoute.FOREST_DENSE_FALLBACK,
        ],
    )
    np.testing.assert_array_equal(np.asarray(evidence.rank_lower_bound), [rows, 0])
    np.testing.assert_array_equal(np.asarray(evidence.rank_upper_bound), [rows, rows])
    np.testing.assert_array_equal(np.asarray(evidence.full_row_rank), [True, False])


def test_nonfinite_step_with_fallback_rolls_back() -> None:
    bodies, reference, graph = _ball_chain(4)
    dynamics = _prepare(graph, bodies, reference, _FALLBACK)
    broken = _nonfinite(bodies, reference)
    state = dynamics.initialize_state(
        broken.position, broken.velocity, broken.orientation, broken.angular_velocity
    )
    result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-3))
    evidence = result.evaluation.diagnostics.rank
    reasons = int(result.rejection_reasons)
    assert not bool(result.successful)
    assert not bool(evidence.full_row_rank)
    assert int(evidence.status) == RigidConstraintRankStatus.DENSE_SVD_FAILED
    assert int(evidence.executed_route) == RigidConstraintRankRoute.FOREST_DENSE_FALLBACK
    assert reasons & int(RigidConstraintRejectionReason.INVALID_STATE)
    assert reasons & int(RigidConstraintRejectionReason.RANK_OR_CONDITION)
    assert not reasons & int(RigidConstraintRejectionReason.RANK_INCONCLUSIVE)
    np.testing.assert_array_equal(
        np.asarray(result.accepted_state.kinematics.position),
        np.asarray(state.kinematics.position),
    )


def test_evidence_uses_current_numeric_state() -> None:
    ids, bodies = _bodies(2, [True, False])
    reference = _reference(bodies, _line(2))
    graph = RigidJointGraphPlan(
        distance=phx.discretization.DistanceJointSetPlan(
            jnp.asarray([7]),
            ids[:1],
            ids[1:],
            jnp.asarray([[0.0, 0.0, 0.0]]),
            jnp.asarray([[1.0, 0.0, 0.0]]),
        )
    )
    forest = _prepare(graph, bodies, reference, _FOREST)
    general = _prepare(graph, bodies, reference)
    collapsed = bodies.kinematics(
        jnp.zeros_like(reference.position),
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    assert bool(forest.constraint_rank_evidence(reference).full_row_rank)
    stale = forest.constraint_rank_evidence(collapsed)
    assert int(stale.status) == RigidConstraintRankStatus.INCONCLUSIVE
    assert stale.certificate is not None and int(stale.certificate.numeric_version) == 1
    dense = general.constraint_rank_evidence(collapsed)
    assert int(dense.status) == RigidConstraintRankStatus.RANK_DEFICIENT_MEASURED
    assert int(dense.rank_upper_bound) == 0


def test_empty_graph_keeps_rank_zero_condition_one() -> None:
    _, bodies = _bodies(2, [True, False])
    reference = _reference(bodies, _line(2))
    dynamics = _prepare(RigidJointGraphPlan(), bodies, reference, _FOREST)
    state = dynamics.initialize_state(
        reference.position,
        reference.velocity,
        reference.orientation,
        reference.angular_velocity,
    )
    evidence = dynamics.step(
        state, jnp.asarray(0.0), jnp.asarray(1.0e-3)
    ).evaluation.diagnostics.rank
    assert int(evidence.status) == RigidConstraintRankStatus.EMPTY
    assert int(evidence.executed_route) == RigidConstraintRankRoute.NONE
    assert int(evidence.rank_upper_bound) == 0
    assert float(evidence.measured_condition) == 1.0


def test_rank_audit_options_belong_to_their_route() -> None:
    with pytest.raises(ValueError, match="certified-forest"):
        RigidConstraintRankAudit("general-dense-svd", dense_fallback=SVDResourcePolicy())
    with pytest.raises(ValueError, match="certified-forest"):
        RigidConstraintRankAudit("general-dense-svd", workspace_bytes=1024)
    default = RigidConstraintSolverPlan()
    assert default.rank_audit.kind == "general-dense-svd"
    assert RigidConstraintSolverPlan(rank_audit=_FOREST).plan_id != default.plan_id
