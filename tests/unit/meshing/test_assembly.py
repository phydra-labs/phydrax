import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing._assembly import MeshAssembly, MeshPart


def test_assembly_preserves_compact_carriers_and_binds_spline_geometry_revision():
    contract = phx.SpatialCoordinateContract.si()
    tensor = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2),
            phx.discretization.UniformCellAxisSpec(2),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    points = np.asarray([(x, y) for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])
    cloud = phx.discretization.PointCloudPlan(points, np.ones(9), neighbor_count=9)
    iga = phx.discretization.iga
    axis = iga.BSplineGrid.open_uniform(2, 1)
    xx, yy = jnp.meshgrid(axis.greville_abscissae, axis.greville_abscissae, indexing="ij")
    spline = iga.IsogeometricPlan.isoparametric(
        (axis, axis),
        iga.NURBSGeometryState(jnp.stack((xx, yy), axis=-1), jnp.ones(xx.shape)),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    parts = tuple(
        MeshPart(name, carrier, coordinate_contract=contract)
        for name, carrier in (("grid", tensor), ("cloud", cloud), ("spline", spline))
    )
    assembly = MeshAssembly(parts)
    assert assembly.assembly_id == MeshAssembly(tuple(reversed(parts))).assembly_id
    assert isinstance(
        assembly.part("grid").carrier, phx.discretization.PreparedTensorGrid
    )
    assert isinstance(assembly.part("cloud").carrier, phx.discretization.PointCloudPlan)
    assert isinstance(assembly.part("spline").carrier, iga.IsogeometricPlan)
    np.testing.assert_allclose(
        assembly.part("cloud").point_coordinates(parts[1].scope(0, [2, 4])),
        points[[2, 4]],
    )
    spline_scope = parts[2].scope(2, [0])
    changed_geometry = iga.NURBSGeometryState(
        spline.geometry.control_points * 2, spline.geometry.weights
    )
    changed_spline = eqx.tree_at(lambda value: value.geometry, spline, changed_geometry)
    changed_part = MeshPart("spline", changed_spline, coordinate_contract=contract)
    assert changed_spline.plan_id == spline.plan_id
    with pytest.raises(ValueError, match="stale"):
        changed_part.require_scope(spline_scope)
    with pytest.raises(ValueError, match="coefficient vertices"):
        parts[2].scope(0, [0])


def test_assembly_rejects_duplicate_part_ownership_and_coordinate_frame_mismatch():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    part = MeshPart("fluid", grid, coordinate_contract=phx.SpatialCoordinateContract.si())
    with pytest.raises(ValueError, match="unique name"):
        MeshAssembly((part, part))
    other = MeshPart(
        "solid",
        grid,
        coordinate_contract=phx.SpatialCoordinateContract(
            phx.SpatialCoordinateContract.si().length_unit, reference_frame="body"
        ),
    )
    with pytest.raises(ValueError, match="coordinate contract"):
        MeshAssembly((part, other))
    with pytest.raises(ValueError, match="unknown or inactive"):
        part.scope(1, [9])
