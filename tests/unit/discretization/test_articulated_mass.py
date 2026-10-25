#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.particle._articulated_mass import (
    ArticulatedInverseMassOperator,
    ArticulatedMassOperator,
    ArticulatedMassPolicy,
    ArticulatedMassStatus,
    prepare_articulated_mass,
    PreparedArticulatedMass,
)
from phydrax.discretization.particle._reduced_articulation import (
    PreparedReducedArticulation,
)
from phydrax.discretization.particle._reduced_dynamics import (
    reduced_energy,
    reduced_forward_dynamics,
    reduced_mass_matrix,
    ReducedDynamicsStatus,
)


_ROOT = 9001
_BODY_IDS = (9001, 42, 7, 1_000_003, 513, 77)
_POSITIONS = (
    (0.0, 0.0, 0.0),
    (0.3, 0.0, 0.1),
    (0.9, 0.2, 0.1),
    (0.9, 0.7, 0.3),
    (0.3, -0.6, 0.2),
    (0.5, -1.1, 0.0),
)
_MASSES = (1.0, 1.5, 0.8, 0.6, 1.1, 0.4)
_INERTIA_PARAMETERS = (
    (0.2, 0.25, 0.3, 0.02),
    (0.1, 0.12, 0.09, 0.01),
    (0.05, 0.04, 0.06, 0.005),
    (0.08, 0.1, 0.07, -0.01),
    (0.03, 0.02, 0.025, 0.003),
    (0.15, 0.15, 0.15, 0.0),
)
_ORIENTATIONS = (
    (1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.95, 0.1, 0.2, 0.05),
    (1.0, 0.0, 0.0, 0.0),
    (0.9, 0.0, 0.3, 0.3),
    (1.0, 0.0, 0.0, 0.0),
)
# (joint ID, kind, parent ID, child ID, world anchor, world axis)
_EDGES = (
    (300, "fixed", 9001, 42, None, None),
    (301, "hinge", 42, 7, (0.6, 0.1, 0.1), (0.0, 0.0, 1.0)),
    (302, "prismatic", 7, 1_000_003, (0.9, 0.2, 0.1), (0.2, 1.0, 0.4)),
    (303, "hinge", 42, 513, (0.3, -0.3, 0.15), (1.0, 1.0, 0.0)),
    (304, "hinge", 513, 77, (0.4, -0.85, 0.1), (1.0, 0.0, 0.3)),
)


def _inertia(a: float, b: float, c: float, d: float) -> np.ndarray:
    return np.asarray([[a, d, 0.0], [d, b, 0.5 * d], [0.0, 0.5 * d, c]])


def _joint_set(
    kind: str, edges: Sequence[tuple[int, str, int, int, object, object]]
) -> tuple[np.ndarray, ...] | None:
    selected = [edge for edge in edges if edge[1] == kind]
    if not selected:
        return None
    identifiers = np.asarray([edge[0] for edge in selected], dtype=np.int64)
    parents = np.asarray([edge[2] for edge in selected], dtype=np.int64)
    children = np.asarray([edge[3] for edge in selected], dtype=np.int64)
    if kind == "fixed":
        return identifiers, parents, children
    anchors = np.asarray([edge[4] for edge in selected], dtype=np.float64)
    axes = np.asarray([edge[5] for edge in selected], dtype=np.float64)
    return identifiers, parents, children, anchors, axes


def _mixed_tree(
    *,
    body_order: Sequence[int] = (0, 1, 2, 3, 4, 5),
    edge_order: Sequence[int] = (0, 1, 2, 3, 4),
) -> PreparedReducedArticulation:
    """Branching fixed/hinge/prismatic tree with large, unsorted stable IDs."""

    order = np.asarray(body_order)
    body_ids = np.asarray(_BODY_IDS, dtype=np.int64)[order]
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray(body_ids),
        jnp.asarray(np.asarray(_MASSES)[order]),
        ambient_dimension=3,
    ).prepare()
    inertia = np.stack([_inertia(*_INERTIA_PARAMETERS[index]) for index in order])
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((order.size,), dtype=jnp.int32),
        jnp.asarray(inertia),
        fixed_mask=jnp.asarray(body_ids == _ROOT),
    ).prepare(particles)
    quaternions = np.asarray(_ORIENTATIONS, dtype=np.float64)[order]
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    positions = np.asarray(_POSITIONS, dtype=np.float64)[order]
    reference = bodies.kinematics(
        jnp.asarray(positions),
        jnp.zeros((order.size, 3)),
        jnp.asarray(quaternions),
        jnp.zeros((order.size, 3)),
    )
    edges = [_EDGES[index] for index in edge_order]
    fixed = _joint_set("fixed", edges)
    hinge = _joint_set("hinge", edges)
    prismatic = _joint_set("prismatic", edges)
    graph = phx.discretization.RigidJointGraphPlan(
        fixed=None
        if fixed is None
        else phx.discretization.FixedJointSetPlan(*map(jnp.asarray, fixed)),
        hinge=None
        if hinge is None
        else phx.discretization.HingeJointSetPlan(*map(jnp.asarray, hinge)),
        prismatic=None
        if prismatic is None
        else phx.discretization.PrismaticJointSetPlan(*map(jnp.asarray, prismatic)),
    ).prepare(bodies, reference)
    plan = phx.discretization.ReducedArticulationPlan(
        _ROOT,
        jnp.asarray([edge[0] for edge in edges], dtype=jnp.int64),
        jnp.asarray([edge[2] for edge in edges], dtype=jnp.int64),
        jnp.asarray([edge[3] for edge in edges], dtype=jnp.int64),
    )
    return plan.prepare(graph, reference)


def _single_axis(kind: str) -> PreparedReducedArticulation:
    """Root plus one body: mass 2 and I_zz 3 at unit hinge arm, or a slider."""

    body_ids = jnp.asarray([10, 11], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids, jnp.asarray([1.0, 2.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.stack((jnp.eye(3), jnp.diag(jnp.asarray([1.0, 1.0, 3.0])))),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    position = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    reference = bodies.kinematics(
        position,
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        jnp.zeros((2, 3)),
    )
    arguments = (
        jnp.asarray([20]),
        body_ids[:1],
        body_ids[1:],
        jnp.asarray([[0.0, 0.0, 0.0]]),
        jnp.asarray([[0.0, 0.0, 1.0]] if kind == "hinge" else [[1.0, 0.0, 0.0]]),
    )
    graph = (
        phx.discretization.RigidJointGraphPlan(
            hinge=phx.discretization.HingeJointSetPlan(*arguments)
        )
        if kind == "hinge"
        else phx.discretization.RigidJointGraphPlan(
            prismatic=phx.discretization.PrismaticJointSetPlan(*arguments)
        )
    ).prepare(bodies, reference)
    return phx.discretization.ReducedArticulationPlan(
        10, jnp.asarray([20]), jnp.asarray([10]), jnp.asarray([11])
    ).prepare(graph, reference)


def _fixed_only() -> PreparedReducedArticulation:
    body_ids = jnp.asarray([5, 6], dtype=jnp.int64)
    particles = phx.discretization.ParticleSetPlan(
        body_ids, jnp.asarray([1.0, 2.0]), ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        fixed_mask=jnp.asarray([True, False]),
    ).prepare(particles)
    reference = bodies.kinematics(
        jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        jnp.zeros((2, 3)),
    )
    graph = phx.discretization.RigidJointGraphPlan(
        fixed=phx.discretization.FixedJointSetPlan(
            jnp.asarray([8]), body_ids[:1], body_ids[1:]
        )
    ).prepare(bodies, reference)
    return phx.discretization.ReducedArticulationPlan(
        5, jnp.asarray([8]), jnp.asarray([5]), jnp.asarray([6])
    ).prepare(graph, reference)


def _configuration() -> Array:
    return jnp.asarray([0.41, -0.23, 0.77, -1.3], dtype=jnp.float64)


def _dense_mass(articulation: PreparedReducedArticulation, q: Array) -> Array:
    """Independent JᵀGJ reference from forward-mode body Jacobians."""

    return reduced_mass_matrix(articulation, q).matrix


def _dof_joint_ids(articulation: PreparedReducedArticulation) -> np.ndarray:
    return np.asarray(articulation.joint_ids)[np.asarray(articulation.dof_joint_indices)]


def _with_leaf(
    articulation: PreparedReducedArticulation,
    where: Callable[[PreparedReducedArticulation], Array],
    value: Array,
) -> PreparedReducedArticulation:
    return eqx.tree_at(where, articulation, value)


def _masses(articulation: PreparedReducedArticulation) -> Array:
    return articulation.graph.bodies.mass_properties.masses


def _inertias(articulation: PreparedReducedArticulation) -> Array:
    return articulation.graph.bodies.mass_properties.inertia_com


def _com_offsets(articulation: PreparedReducedArticulation) -> Array:
    return articulation.parent_reference_translation


def _inverse_value(
    articulation: PreparedReducedArticulation, q: Array, effort: Array
) -> Array:
    return prepare_articulated_mass(articulation, q).apply_inverse(
        articulation, q, effort
    ).value


def _dense_inverse(
    articulation: PreparedReducedArticulation, q: Array, effort: Array
) -> Array:
    return jnp.linalg.solve(_dense_mass(articulation, q), effort)


def test_single_axis_analytic_inverse_and_mass() -> None:
    for kind, expected in (("hinge", 5.0), ("prismatic", 2.0)):
        articulation = _single_axis(kind)
        q = jnp.asarray([0.3])
        mass = prepare_articulated_mass(articulation, q)

        assert isinstance(mass, PreparedArticulatedMass)
        assert mass.successful
        assert mass.status == int(ArticulatedMassStatus.SUCCESS)
        assert jnp.allclose(mass.evidence.pivots, jnp.asarray([expected]), rtol=1e-12)
        assert jnp.allclose(mass.evidence.minimum_pivot, expected, rtol=1e-12)
        assert bool(mass.evidence.moving[0])

        inverse = mass.apply_inverse(articulation, q, jnp.asarray([2.0]))
        product = mass.apply_mass(articulation, q, jnp.asarray([2.0]))
        assert inverse.successful and product.successful
        assert inverse.status == int(ArticulatedMassStatus.SUCCESS)
        assert jnp.allclose(inverse.value, jnp.asarray([2.0 / expected]), rtol=1e-12)
        assert jnp.allclose(product.value, jnp.asarray([2.0 * expected]), rtol=1e-12)
        assert inverse.relative_residual.shape == ()
        assert inverse.relative_residual <= 1.0e-12


def test_operators_map_true_dual_spaces_without_dense_materialization() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    mass = prepare_articulated_mass(articulation, q)
    inverse = mass.inverse_mass_operator
    forward = mass.mass_operator

    assert isinstance(inverse, ArticulatedInverseMassOperator)
    assert isinstance(forward, ArticulatedMassOperator)
    assert inverse.source.compatible(articulation.generalized_effort_space)
    assert inverse.source.compatible(phx.linalg.DualSpace(inverse.target))
    assert inverse.target.compatible(articulation.generalized_velocity_space)
    assert forward.source.compatible(inverse.target)
    assert forward.target.compatible(inverse.source)
    jacobian = articulation.body_jacobian_operator(q)
    assert jacobian.source.compatible(inverse.target)
    composed = jacobian @ inverse
    assert composed.source.compatible(articulation.generalized_effort_space)
    with pytest.raises(phx.linalg.LinearCapabilityError):
        inverse._materialize()
    with pytest.raises(phx.linalg.LinearCapabilityError):
        forward._materialize()


def test_mixed_branching_tree_matches_independent_dense_reference() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    mass = prepare_articulated_mass(articulation, q)
    dense = _dense_mass(articulation, q)
    effort = jnp.asarray([0.7, -1.2, 0.35, 2.1])
    velocity = jnp.asarray([-0.4, 0.9, 1.3, -0.2])

    assert articulation.nv == 4
    assert mass.successful
    kinds = np.asarray(articulation.joint_kinds)
    assert np.array_equal(
        np.asarray(mass.evidence.moving),
        kinds != int(phx.discretization.RigidJointKind.FIXED),
    )
    assert jnp.all(jnp.where(mass.evidence.moving, mass.evidence.pivots, 1.0) > 0.0)
    fixed_edges = ~np.asarray(mass.evidence.moving)
    assert np.all(np.asarray(mass.factor.reciprocal_pivots)[fixed_edges] == 0.0)

    inverse = mass.apply_inverse(articulation, q, effort)
    assert inverse.successful
    assert jnp.allclose(inverse.value, jnp.linalg.solve(dense, effort), rtol=1e-10)
    assert inverse.relative_residual <= 1.0e-12
    assert jnp.allclose(
        mass.apply_mass(articulation, q, velocity).value, dense @ velocity, rtol=1e-10
    )
    assert jnp.allclose(mass.inverse_mass_operator.mv(effort), inverse.value)
    assert jnp.allclose(mass.mass_operator.mv(velocity), dense @ velocity, rtol=1e-10)


def test_multiple_rhs_and_prepared_versus_cold_actions() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    mass = prepare_articulated_mass(articulation, q)
    columns = jnp.asarray(
        [
            [1.0, 0.0, 0.3, -2.0, 0.0],
            [0.0, 1.0, -0.7, 0.4, 0.0],
            [0.2, 0.0, 1.1, 0.9, 0.0],
            [-0.5, 0.3, 0.0, 1.6, 0.0],
        ]
    )
    vectors = jnp.stack(
        [mass.apply_inverse(articulation, q, column).value for column in columns.T],
        axis=1,
    )
    cold = jnp.stack(
        [_inverse_value(articulation, q, column) for column in columns.T], axis=1
    )
    block = mass.apply_inverse(articulation, q, columns, rhs_axis=1)
    transposed = mass.apply_inverse(articulation, q, columns.T, rhs_axis=0)
    negative_axis = mass.apply_inverse(articulation, q, columns, rhs_axis=-1)

    assert block.successful and transposed.successful
    assert block.value.shape == columns.shape
    assert transposed.value.shape == columns.T.shape
    assert block.relative_residual.shape == (columns.shape[1],)
    assert jnp.all(block.relative_residual <= 1.0e-12)
    assert block.relative_residual[-1] == 0.0
    assert jnp.allclose(block.value, vectors, rtol=1e-13, atol=1e-15)
    assert jnp.allclose(vectors, cold, rtol=1e-13, atol=1e-15)
    assert jnp.allclose(transposed.value, vectors.T, rtol=1e-13, atol=1e-15)
    assert jnp.allclose(negative_axis.value, block.value)
    assert jnp.allclose(
        mass.inverse_mass_operator.mv_block(columns), vectors, rtol=1e-13, atol=1e-15
    )
    products = mass.apply_mass(articulation, q, vectors, rhs_axis=1)
    assert products.successful
    assert jnp.allclose(products.value, columns, rtol=1e-10, atol=1e-12)


def test_permuted_stable_ids_and_edge_order_preserve_generalized_response() -> None:
    baseline = _mixed_tree()
    permuted = _mixed_tree(body_order=(3, 5, 0, 4, 1, 2), edge_order=(4, 2, 0, 3, 1))
    baseline_ids = _dof_joint_ids(baseline)
    permuted_ids = _dof_joint_ids(permuted)
    to_permuted = np.asarray(
        [int(np.flatnonzero(baseline_ids == joint)[0]) for joint in permuted_ids]
    )
    q = _configuration()
    effort = jnp.asarray([0.7, -1.2, 0.35, 2.1])

    first = _inverse_value(baseline, q, effort)
    second = _inverse_value(permuted, q[to_permuted], effort[to_permuted])

    assert sorted(baseline_ids.tolist()) == sorted(permuted_ids.tolist())
    assert jnp.allclose(second, first[to_permuted], rtol=1e-11, atol=1e-13)


def test_empty_generalized_space_is_exact() -> None:
    articulation = _fixed_only()
    q = jnp.zeros((0,))
    mass = prepare_articulated_mass(articulation, q)

    assert articulation.nv == 0
    assert mass.successful
    assert jnp.isinf(mass.evidence.minimum_pivot)
    vector = mass.apply_inverse(articulation, q, jnp.zeros((0,)))
    block = mass.apply_inverse(articulation, q, jnp.zeros((0, 3)), rhs_axis=1)
    assert vector.successful and block.successful
    assert vector.value.shape == (0,)
    assert block.value.shape == (0, 3)
    assert jnp.all(block.relative_residual == 0.0)


def test_adjoint_symmetry_virtual_work_and_kinetic_energy() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    mass = prepare_articulated_mass(articulation, q)
    first = jnp.asarray([0.7, -1.2, 0.35, 2.1])
    second = jnp.asarray([-0.3, 0.5, 1.4, -0.8])

    evidence = mass.adjoint_evidence(first, second)
    assert evidence.finite and evidence.valid
    assert jnp.abs(evidence.residual) <= 1.0e-13 * evidence.scale
    assert jnp.allclose(
        mass.inverse_mass_operator.transpose_mv(first),
        mass.inverse_mass_operator.mv(first),
    )
    assert jnp.allclose(
        mass.inverse_mass_operator.adjoint_mv(first),
        mass.inverse_mass_operator.mv(first),
    )
    response = mass.apply_inverse(articulation, q, first).value
    effort_work = mass.effort_space.pair(first, response)
    kinetic = reduced_energy(articulation, q, response, jnp.zeros((3,))).kinetic
    product = mass.apply_mass(articulation, q, response).value
    assert jnp.allclose(effort_work, 2.0 * kinetic, rtol=1e-11)
    assert jnp.allclose(jnp.vdot(response, product), 2.0 * kinetic, rtol=1e-11)
    assert effort_work > 0.0


def test_configuration_jvp_vjp_match_dense_reference_and_finite_difference() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    effort = jnp.asarray([0.7, -1.2, 0.35, 2.1])
    direction = jnp.asarray([0.3, -0.8, 0.5, 0.1])
    cotangent = jnp.asarray([1.1, 0.2, -0.6, 0.9])

    def native(point: Array) -> Array:
        return _inverse_value(articulation, point, effort)

    def dense(point: Array) -> Array:
        return _dense_inverse(articulation, point, effort)

    _, native_tangent = jax.jvp(native, (q,), (direction,))
    _, dense_tangent = jax.jvp(dense, (q,), (direction,))
    step = 1.0e-6
    central = (native(q + step * direction) - native(q - step * direction)) / (2 * step)
    native_pullback = jax.vjp(native, q)[1](cotangent)[0]
    dense_pullback = jax.vjp(dense, q)[1](cotangent)[0]

    assert jnp.allclose(native_tangent, dense_tangent, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(native_tangent, central, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(native_pullback, dense_pullback, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(jnp.vdot(cotangent, native_tangent), jnp.vdot(native_pullback, direction))


def _identity(value: Array) -> Array:
    return value


def _symmetric(value: Array) -> Array:
    """Admissible COM inertia coordinates: only the symmetric part is physical."""

    return 0.5 * (value + jnp.swapaxes(value, -1, -2))


@pytest.mark.parametrize("parameter", ["mass", "inertia", "com"])
def test_mass_inertia_and_com_derivatives_match_dense_reference(parameter: str) -> None:
    articulation = _mixed_tree()
    q = _configuration()
    effort = jnp.asarray([0.7, -1.2, 0.35, 2.1])
    cotangent = jnp.asarray([0.4, -1.0, 0.25, 0.6])
    capacity = articulation.graph.bodies.capacity
    match parameter:
        case "mass":
            where, realize = _masses, _identity
            direction = jnp.linspace(0.2, -0.3, capacity)
        case "inertia":
            # Raw, nonsymmetric coordinates are mapped onto the symmetric
            # tensors the mechanics admits; derivatives are compared in them.
            where, realize = _inertias, _symmetric
            direction = (
                jnp.arange(capacity * 9, dtype=jnp.float64).reshape((capacity, 3, 3))
                / 50.0
            )
        case "com":
            where, realize = _com_offsets, _identity
            direction = jnp.sin(
                jnp.arange(articulation.edge_count * 3, dtype=jnp.float64)
            ).reshape((articulation.edge_count, 3))
        case _:
            raise AssertionError(parameter)
    value = where(articulation)

    def native(coordinates: Array) -> Array:
        changed = _with_leaf(articulation, where, realize(coordinates))
        return _inverse_value(changed, q, effort)

    def dense(coordinates: Array) -> Array:
        changed = _with_leaf(articulation, where, realize(coordinates))
        return _dense_inverse(changed, q, effort)

    _, native_tangent = jax.jvp(native, (value,), (direction,))
    _, dense_tangent = jax.jvp(dense, (value,), (direction,))
    native_pullback = jax.vjp(native, value)[1](cotangent)[0]
    dense_pullback = jax.vjp(dense, value)[1](cotangent)[0]
    step = 1.0e-6
    central = (native(value + step * direction) - native(value - step * direction)) / (
        2.0 * step
    )

    assert jnp.linalg.norm(dense_tangent) > 0.0
    assert jnp.allclose(native_tangent, dense_tangent, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(native_tangent, central, rtol=1e-6, atol=1e-8)
    assert jnp.allclose(native_pullback, dense_pullback, rtol=1e-9, atol=1e-11)
    assert jnp.allclose(
        jnp.vdot(native_pullback, direction),
        jnp.vdot(cotangent, central),
        rtol=1e-6,
        atol=1e-8,
    )


def test_stale_state_frame_and_material_changes_are_never_accepted() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    effort = jnp.asarray([0.7, -1.2, 0.35, 2.1])
    mass = prepare_articulated_mass(articulation, q)
    moved = q.at[2].add(1.0e-12)
    heavier = _with_leaf(articulation, _masses, _masses(articulation) * 1.01)
    reframed = _with_leaf(
        articulation,
        lambda value: value.parent_axes,
        articulation.parent_axes.at[1, 0].add(1.0e-9),
    )

    assert mass.is_current(articulation, q)
    for current_articulation, point in (
        (articulation, moved),
        (heavier, q),
        (reframed, q),
    ):
        assert not mass.is_current(current_articulation, point)
        result = mass.apply_inverse(current_articulation, point, effort)
        product = mass.apply_mass(current_articulation, point, effort)
        assert not result.current and not result.successful
        assert result.status == int(ArticulatedMassStatus.STALE_BINDING)
        assert jnp.all(result.value == 0.0)
        assert product.status == int(ArticulatedMassStatus.STALE_BINDING)
        assert jnp.all(product.value == 0.0)
    with pytest.raises(ValueError, match="different identities"):
        mass.is_current(_single_axis("hinge"), jnp.asarray([0.0]))
    refreshed = prepare_articulated_mass(heavier, q)
    assert refreshed.apply_inverse(heavier, q, effort).successful
    assert refreshed.mass_id == mass.mass_id


@pytest.mark.parametrize("value", [jnp.inf, -jnp.inf, jnp.nan])
def test_nonfinite_bound_data_never_qualifies_as_current(value: float) -> None:
    articulation = _mixed_tree()
    q = _configuration()
    effort = jnp.asarray([0.7, -1.2, 0.35, 2.1])
    cases = (
        (articulation, q.at[1].set(value)),
        (_with_leaf(articulation, _masses, _masses(articulation).at[3].set(value)), q),
        (
            _with_leaf(
                articulation,
                _com_offsets,
                _com_offsets(articulation).at[2, 1].set(value),
            ),
            q,
        ),
    )
    for current_articulation, point in cases:
        mass = prepare_articulated_mass(current_articulation, point)
        result = mass.apply_inverse(current_articulation, point, effort)

        assert not mass.binding.matches(mass.binding)
        assert not mass.is_current(current_articulation, point)
        assert not result.current and not result.successful
        assert result.status == int(ArticulatedMassStatus.STALE_BINDING)
        assert jnp.all(result.value == 0.0)


def test_zero_dof_infinite_minimum_is_diagnostic_not_bound_data() -> None:
    articulation = _fixed_only()
    q = jnp.zeros((0,))
    mass = prepare_articulated_mass(articulation, q)

    assert jnp.isinf(mass.evidence.minimum_pivot)
    assert jnp.isinf(mass.evidence.minimum_relative_pivot)
    assert mass.binding.matches(mass.binding)
    assert mass.is_current(articulation, q)
    assert mass.apply_inverse(articulation, q, jnp.zeros((0,))).successful


def test_nonpositive_pivots_are_refused_without_repair() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    effort = jnp.asarray([0.7, -1.2, 0.35, 2.1])
    leaf = int(np.flatnonzero(np.asarray(articulation.graph.bodies.particles.particle_ids) == 77)[0])
    inertia = _inertias(articulation).at[leaf].set(jnp.zeros((3, 3)))
    for leaf_mass in (0.0, -0.4):
        changed = _with_leaf(
            _with_leaf(articulation, _inertias, inertia),
            _masses,
            _masses(articulation).at[leaf].set(leaf_mass),
        )
        mass = prepare_articulated_mass(changed, q)
        result = mass.apply_inverse(changed, q, effort)
        forward = reduced_forward_dynamics(
            changed, q, jnp.zeros((4,)), effort, jnp.zeros((3,))
        )

        assert not mass.successful
        assert not mass.evidence.pivots_positive
        assert mass.status == int(ArticulatedMassStatus.PIVOT_NOT_POSITIVE)
        assert mass.evidence.minimum_pivot <= 0.0
        assert not result.successful
        assert result.status == int(ArticulatedMassStatus.PREPARATION_FAILED)
        assert jnp.all(result.value == 0.0)
        assert not forward.successful
        assert forward.status == int(ReducedDynamicsStatus.MASS_NOT_POSITIVE_DEFINITE)


def test_nonfinite_inputs_are_reported() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    nan_q = prepare_articulated_mass(articulation, q.at[0].set(jnp.nan))
    nan_inertia = _with_leaf(
        articulation, _inertias, _inertias(articulation).at[2, 0, 0].set(jnp.nan)
    )
    mass = prepare_articulated_mass(articulation, q)
    nan_effort = mass.apply_inverse(
        articulation, q, jnp.asarray([0.0, jnp.nan, 1.0, 0.0])
    )

    assert nan_q.status == int(ArticulatedMassStatus.NONFINITE_INPUT)
    assert not nan_q.successful
    assert (
        prepare_articulated_mass(nan_inertia, q).status
        == int(ArticulatedMassStatus.NONFINITE_INPUT)
    )
    assert not nan_effort.successful
    assert nan_effort.status == int(ArticulatedMassStatus.NONFINITE_INPUT)
    assert jnp.all(nan_effort.value == 0.0)


def test_resource_and_layout_refusals() -> None:
    articulation = _mixed_tree()
    q = _configuration()
    mass = prepare_articulated_mass(
        articulation, q, policy=ArticulatedMassPolicy(maximum_rhs=2)
    )

    assert mass.apply_inverse(articulation, q, jnp.ones((4, 2)), rhs_axis=1).successful
    with pytest.raises(ValueError, match="maximum_rhs"):
        mass.apply_inverse(articulation, q, jnp.ones((4, 3)), rhs_axis=1)
    with pytest.raises(ValueError, match="maximum_rhs"):
        mass.apply_mass(articulation, q, jnp.ones((3, 4)), rhs_axis=0)
    with pytest.raises(ValueError):
        mass.apply_inverse(articulation, q, jnp.ones((4, 2)), rhs_axis=2)
    with pytest.raises(TypeError):
        mass.apply_inverse(articulation, q, jnp.ones((4, 2)), rhs_axis=True)
    with pytest.raises(ValueError):
        mass.apply_inverse(articulation, q, jnp.ones((3,)))
    with pytest.raises(ValueError):
        mass.apply_inverse(articulation, q, jnp.ones((4, 2)))
    with pytest.raises(ValueError):
        ArticulatedMassPolicy(maximum_rhs=0)
    with pytest.raises(TypeError):
        ArticulatedMassPolicy(maximum_rhs=2.0)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        ArticulatedMassPolicy(relative_pivot_tolerance=-1.0)
    with pytest.raises(ValueError):
        ArticulatedMassPolicy(residual_tolerance=float("nan"))
    with pytest.raises(TypeError):
        prepare_articulated_mass(articulation, q, policy=object())  # ty: ignore[invalid-argument-type]


def test_eager_jit_and_vmap_agree() -> None:
    articulation = _mixed_tree()
    points = jnp.stack((_configuration(), _configuration()[::-1], -_configuration()))
    efforts = jnp.asarray(
        [[0.7, -1.2, 0.35, 2.1], [1.0, 0.0, -0.5, 0.2], [-0.3, 0.5, 1.4, -0.8]]
    )

    def action(point: Array, effort: Array) -> tuple[Array, Array]:
        result = prepare_articulated_mass(articulation, point).apply_inverse(
            articulation, point, effort
        )
        return result.value, result.successful

    eager = [action(point, effort) for point, effort in zip(points, efforts)]
    compiled = jax.jit(action)
    batched_value, batched_success = jax.vmap(action)(points, efforts)
    compiled_batch = jax.jit(jax.vmap(action))(points, efforts)

    for index, (value, successful) in enumerate(eager):
        assert successful
        jitted_value, jitted_success = compiled(points[index], efforts[index])
        assert jitted_success
        assert jnp.allclose(jitted_value, value, rtol=1e-13, atol=1e-15)
        assert jnp.allclose(batched_value[index], value, rtol=1e-13, atol=1e-15)
        assert jnp.allclose(compiled_batch[0][index], value, rtol=1e-13, atol=1e-15)
    assert jnp.all(batched_success)

    batched_masses = jax.vmap(lambda point: prepare_articulated_mass(articulation, point))(
        points
    )
    assert batched_masses.factor.pivots.shape == (3, articulation.edge_count)
    assert jnp.all(batched_masses.successful)


def _device_action(device: jax.Device) -> None:
    articulation = _mixed_tree()
    placed = jax.tree.map(
        lambda leaf: jax.device_put(leaf, device) if isinstance(leaf, jax.Array) else leaf,
        articulation,
    )
    q = jax.device_put(_configuration(), device)
    effort = jax.device_put(jnp.asarray([0.7, -1.2, 0.35, 2.1]), device)

    @jax.jit
    def action(point: Array, value: Array) -> tuple[Array, Array]:
        result = prepare_articulated_mass(placed, point).apply_inverse(
            placed, point, value
        )
        return result.value, result.successful

    value, successful = action(q, effort)
    assert value.devices() == {device}
    assert successful
    reference = np.linalg.solve(
        np.asarray(_dense_mass(articulation, _configuration())),
        np.asarray([0.7, -1.2, 0.35, 2.1]),
    )
    assert np.allclose(np.asarray(value), reference, rtol=1e-10)


def test_explicit_cpu_execution() -> None:
    _device_action(jax.devices("cpu")[0])


def test_actual_gpu_execution() -> None:
    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        pytest.skip("No GPU backend is visible to JAX.")
    _device_action(devices[0])
