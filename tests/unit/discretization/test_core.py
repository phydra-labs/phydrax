#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


def _interval_topology() -> Any:
    # ty: ignore[invalid-argument-type]
    vertices = phx.discretization.EntitySet("vertices", 0, [0, 1, 2])
    # ty: ignore[invalid-argument-type]
    edges = phx.discretization.EntitySet("edges", 1, [0, 1])
    relation = phx.sparse.EdgeRelation(
        # ty: ignore[invalid-argument-type]
        [0, 1, 1, 2],
        # ty: ignore[invalid-argument-type]
        [0, 0, 1, 1],
        source_size=3,
        target_size=2,
    )
    incidence = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edges,
        relation,
        # ty: ignore[invalid-argument-type]
        [-1.0, 1.0, -1.0, 1.0],
    )
    return phx.discretization.CellComplexTopology((vertices, edges), (incidence,))


def _triangle_topology() -> Any:
    # ty: ignore[invalid-argument-type]
    vertices = phx.discretization.EntitySet("vertices", 0, [0, 1, 2])
    # ty: ignore[invalid-argument-type]
    edges = phx.discretization.EntitySet("edges", 1, [0, 1, 2])
    # ty: ignore[invalid-argument-type]
    faces = phx.discretization.EntitySet("faces", 2, [0])
    edge_relation = phx.sparse.EdgeRelation(
        # ty: ignore[invalid-argument-type]
        [0, 1, 1, 2, 2, 0],
        # ty: ignore[invalid-argument-type]
        [0, 0, 1, 1, 2, 2],
        source_size=3,
        target_size=3,
    )
    face_relation = phx.sparse.EdgeRelation(
        # ty: ignore[invalid-argument-type]
        [0, 1, 2],
        # ty: ignore[invalid-argument-type]
        [0, 0, 0],
        source_size=3,
        target_size=1,
    )
    return phx.discretization.CellComplexTopology(
        (vertices, edges, faces),
        (
            phx.discretization.OrientedIncidence(
                1,
                vertices,
                edges,
                edge_relation,
                # ty: ignore[invalid-argument-type]
                [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            ),
            phx.discretization.OrientedIncidence(
                2,
                edges,
                faces,
                face_relation,
                # ty: ignore[invalid-argument-type]
                [1.0, 1.0, 1.0],
            ),
        ),
    )


def _field_space(name: Any = "u") -> Any:
    topology = phx.discretization.TensorTopology(("x",), (3,))
    support = phx.discretization.DiscreteSupport(topology, 1, "line")
    layout = phx.discretization.TensorDofLayout(("x",), (3,))
    vector_space = phx.linalg.ArraySpace((3,), space_id=f"{name}-vectors")
    return phx.discretization.DiscreteFieldSpace(
        name,
        support.support_id,
        layout,
        vector_space,
        representation="point_value",
    )


def test_core_scenario_1() -> None:
    left = phx.discretization.DiscretizationKey(
        "space", "physical", domain_labels=("x", "y")
    )
    right = phx.discretization.DiscretizationKey(
        "space", phx.discretization.DiscretizationRole.PHYSICAL, domain_labels=("x", "y")
    )
    temporal = phx.discretization.DiscretizationKey(
        "space", "temporal", domain_labels=("x", "y")
    )

    assert left.key_id == right.key_id
    assert temporal.key_id != left.key_id
    with pytest.raises(ValueError, match="unique"):
        phx.discretization.DiscretizationKey(
            "space", "physical", domain_labels=("x", "x")
        )
    topology = _triangle_topology()

    assert topology.dimension == 2
    assert topology.incidences[0].scipy_boundary().shape == (3, 3)
    assert np.array_equal(
        (
            topology.incidences[0].scipy_boundary()
            @ topology.incidences[1].scipy_boundary()
        ).toarray(),
        np.zeros((3, 1)),
    )

    bad = phx.discretization.OrientedIncidence(
        2,
        topology.entity_sets[1],
        topology.entity_sets[2],
        topology.incidences[1].relation,
        # ty: ignore[invalid-argument-type]
        [1.0, -1.0, 1.0],
    )
    with pytest.raises(ValueError, match="boundary-of-boundary"):
        phx.discretization.CellComplexTopology(
            topology.entity_sets,
            (topology.incidences[0], bad),
        )
    # ty: ignore[invalid-argument-type]
    subset = phx.discretization.EntitySubset("boundary", [True, False, True])
    with pytest.raises(ValueError, match="inactive"):
        phx.discretization.EntitySet(
            "vertices",
            0,
            # ty: ignore[invalid-argument-type]
            [0, 1, -1],
            # ty: ignore[invalid-argument-type]
            active_mask=[True, True, False],
            subsets=(subset,),
        )


def test_core_scenario_2() -> None:
    topology = phx.discretization.PointTopology(
        phx.discretization.EntitySet(
            "points",
            0,
            # ty: ignore[invalid-argument-type]
            [0, 1, -1],
            # ty: ignore[invalid-argument-type]
            active_mask=[True, True, False],
        )
    )
    support = phx.discretization.DiscreteSupport(topology, 1, "points")
    measure = phx.discretization.DiscreteMeasure(
        "physical",
        support.support_id,
        topology.points.entity_set_id,
        # ty: ignore[invalid-argument-type]
        [0.25, 0.75, np.nan],
        # ty: ignore[invalid-argument-type]
        active_mask=[True, True, False],
        normalization="probability",
    )

    value = measure.integrate(jnp.asarray([2.0, 4.0, jnp.inf]))

    assert jnp.isfinite(value)
    assert jnp.allclose(value, 3.5)
    topology = phx.discretization.TensorTopology(("x",), (3,))
    support = phx.discretization.DiscreteSupport(topology, 1, "line")
    layout = phx.discretization.TensorDofLayout(("x",), (3,))

    with pytest.raises(ValueError, match="does not match"):
        phx.discretization.DiscreteFieldSpace(
            "u",
            support.support_id,
            layout,
            phx.linalg.ArraySpace((4,)),
            representation="point_value",
        )
    topology = phx.discretization.TensorTopology(("x",), (3,))
    support = phx.discretization.DiscreteSupport(topology, 1, "line")
    layout = phx.discretization.TensorDofLayout(("x",), (3,))

    space = phx.discretization.DiscreteFieldSpace(
        "u",
        support.support_id,
        layout,
        phx.linalg.ArraySpace((3,)),
        representation="basis_coefficient",
        conformity="H1",
    )

    assert space.representation == "basis_coefficient"
    source = _field_space("source")
    target = _field_space("target")
    operator = phx.linalg.DenseLinearOperator(
        jnp.eye(3),
        source=source.vector_space,
        target=target.vector_space,
    )
    transfer = phx.discretization.FieldTransfer(
        source,
        target,
        operator,
        properties=phx.discretization.TransferProperties(
            constant_preserving=True,
            exact_on=("constants",),
        ),
    )
    space_key = phx.discretization.DiscretizationKey(
        "space", "physical", domain_labels=("x",)
    )
    time_key = phx.discretization.DiscretizationKey(
        "time", "temporal", domain_labels=("t",)
    )
    bundle = phx.discretization.DiscretizationBundle(
        (
            phx.discretization.DiscretizationRecord(
                space_key, "field-space", source.field_space_id
            ),
            phx.discretization.DiscretizationRecord(
                time_key,
                "temporal-mesh",
                "time-mesh",
                dependency_key_ids=(space_key.key_id,),
            ),
        ),
        transfers=(transfer,),
    )

    assert bundle.record(time_key).dependency_key_ids == (space_key.key_id,)
    assert bundle.transfers[0].properties.constant_preserving

    cyclic_space = phx.discretization.DiscretizationRecord(
        space_key,
        "field-space",
        source.field_space_id,
        dependency_key_ids=(time_key.key_id,),
    )
    with pytest.raises(ValueError, match="acyclic"):
        phx.discretization.DiscretizationBundle((cyclic_space, bundle.record(time_key)))
    plan = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 4, role="driver")
    realized = phx.discretization.TemporalMesh(
        # ty: ignore[invalid-argument-type]
        [0.0, 0.1, 0.4, 1.0],
        role="driver",
        realized=True,
        source_plan_id=plan.mesh_id,
    )

    assert plan.interval_count == 4
    assert realized.source_plan_id == plan.mesh_id
    assert not plan.realized
    with pytest.raises(ValueError, match="source_plan_id"):
        # ty: ignore[invalid-argument-type]
        phx.discretization.TemporalMesh([0.0, 1.0], role="internal", realized=True)


def test_triangle_mesh_exposes_one_canonical_oriented_support() -> None:
    mesh = phx.geometry.TriangleMesh(
        jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        source_id="triangle-support",
    )

    support = mesh.discrete_support()
    topology = support.topology
    # ty: ignore[unresolved-attribute]
    boundary = topology.incidences[0].scipy_boundary()
    # ty: ignore[unresolved-attribute]
    face_boundary = topology.incidences[1].scipy_boundary()

    assert isinstance(topology, phx.discretization.CellComplexTopology)
    assert np.array_equal(
        (boundary @ face_boundary).toarray(),
        np.zeros((3, 1)),
    )
    assert topology.entities(1).subset("boundary").mask.sum() == 3


def test_traced_measure_updates_mass_without_changing_binding_identity() -> None:
    @jax.jit
    def prepare(weights: Array) -> phx.discretization.DiscreteMeasure:
        return phx.discretization.DiscreteMeasure(
            "volume", "mesh", "cells", weights, measure_id="geometry-binding"
        )

    first = prepare(jnp.asarray([2.0, 3.0]))
    second = prepare(jnp.asarray([4.0, 5.0]))
    assert first.measure_id == second.measure_id == "geometry-binding"
    assert first.total_mass == pytest.approx(5.0)
    assert second.total_mass == pytest.approx(9.0)
    assert second.integrate(jnp.asarray([5.0, 7.0])) == pytest.approx(55.0)


def test_measure_integral_differentiates_dynamic_weights() -> None:
    def integral(weights: Array) -> Array:
        measure = phx.discretization.DiscreteMeasure(
            "volume", "mesh", "cells", weights, measure_id="geometry-binding"
        )
        return measure.integrate(jnp.asarray([5.0, 7.0]))

    gradient = jax.jit(jax.grad(integral))(jnp.asarray([2.0, 3.0]))
    np.testing.assert_allclose(gradient, [5.0, 7.0], rtol=0.0, atol=0.0)


def test_traced_measure_refuses_undeclared_binding_identity() -> None:
    @jax.jit
    def mass(weights: Array) -> Array:
        return phx.discretization.DiscreteMeasure(
            "volume", "mesh", "cells", weights
        ).total_mass

    with pytest.raises(ValueError, match="explicit measure_id"):
        mass(jnp.asarray([2.0, 3.0]))


def test_traced_invalid_measure_exposes_failed_admission() -> None:
    @jax.jit
    def prepare(weights: Array) -> phx.discretization.DiscreteMeasure:
        return phx.discretization.DiscreteMeasure(
            "volume", "mesh", "cells", weights, measure_id="geometry-binding"
        )

    measure = prepare(jnp.asarray([-1.0, 3.0]))
    assert not bool(measure.valid)
    with pytest.raises(ValueError, match="non-negative"):
        measure.admit()
