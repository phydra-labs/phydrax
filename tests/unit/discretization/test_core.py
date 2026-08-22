#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _interval_topology():
    vertices = phx.discretization.EntitySet("vertices", 0, [0, 1, 2])
    edges = phx.discretization.EntitySet("edges", 1, [0, 1])
    relation = phx.sparse.EdgeRelation(
        [0, 1, 1, 2],
        [0, 0, 1, 1],
        source_size=3,
        target_size=2,
    )
    incidence = phx.discretization.OrientedIncidence(
        1,
        vertices,
        edges,
        relation,
        [-1.0, 1.0, -1.0, 1.0],
    )
    return phx.discretization.CellComplexTopology((vertices, edges), (incidence,))


def _triangle_topology():
    vertices = phx.discretization.EntitySet("vertices", 0, [0, 1, 2])
    edges = phx.discretization.EntitySet("edges", 1, [0, 1, 2])
    faces = phx.discretization.EntitySet("faces", 2, [0])
    edge_relation = phx.sparse.EdgeRelation(
        [0, 1, 1, 2, 2, 0],
        [0, 0, 1, 1, 2, 2],
        source_size=3,
        target_size=3,
    )
    face_relation = phx.sparse.EdgeRelation(
        [0, 1, 2],
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
                [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            ),
            phx.discretization.OrientedIncidence(
                2,
                edges,
                faces,
                face_relation,
                [1.0, 1.0, 1.0],
            ),
        ),
    )


def _field_space(name="u"):
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


def test_discretization_keys_are_semantic_and_stable():
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


def test_cell_complex_validates_boundary_of_boundary_without_dense_state():
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
        [1.0, -1.0, 1.0],
    )
    with pytest.raises(ValueError, match="boundary-of-boundary"):
        phx.discretization.CellComplexTopology(
            topology.entity_sets,
            (topology.incidences[0], bad),
        )


def test_entity_subsets_cannot_activate_padding():
    subset = phx.discretization.EntitySubset("boundary", [True, False, True])
    with pytest.raises(ValueError, match="inactive"):
        phx.discretization.EntitySet(
            "vertices",
            0,
            [0, 1, -1],
            active_mask=[True, True, False],
            subsets=(subset,),
        )


def test_discrete_measure_masks_nonfinite_padding_before_multiplication():
    topology = phx.discretization.PointTopology(
        phx.discretization.EntitySet(
            "points", 0, [0, 1, -1], active_mask=[True, True, False]
        )
    )
    support = phx.discretization.DiscreteSupport(topology, 1, "points-v1")
    measure = phx.discretization.DiscreteMeasure(
        "physical",
        support.support_id,
        topology.points.entity_set_id,
        [0.25, 0.75, np.nan],
        active_mask=[True, True, False],
        normalization="probability",
    )

    value = measure.integrate(jnp.asarray([2.0, 4.0, jnp.inf]))

    assert jnp.isfinite(value)
    assert jnp.allclose(value, 3.5)


def test_field_space_requires_exact_vector_coordinates():
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


def test_field_transfer_validates_spaces_and_bundle_dependencies():
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


def test_temporal_mesh_distinguishes_plan_from_realization():
    plan = phx.discretization.TemporalMesh.uniform(0.0, 1.0, 4, role="driver")
    realized = phx.discretization.TemporalMesh(
        [0.0, 0.1, 0.4, 1.0],
        role="driver",
        realized=True,
        source_plan_id=plan.mesh_id,
    )

    assert plan.interval_count == 4
    assert realized.source_plan_id == plan.mesh_id
    assert not plan.realized
    with pytest.raises(ValueError, match="source_plan_id"):
        phx.discretization.TemporalMesh([0.0, 1.0], role="internal", realized=True)


def test_triangle_mesh_exposes_one_canonical_oriented_support():
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
    boundary = topology.incidences[0].scipy_boundary()
    face_boundary = topology.incidences[1].scipy_boundary()

    assert isinstance(topology, phx.discretization.CellComplexTopology)
    assert np.array_equal(
        (boundary @ face_boundary).toarray(),
        np.zeros((3, 1)),
    )
    assert topology.entities(1).subset("boundary").mask.sum() == 3
