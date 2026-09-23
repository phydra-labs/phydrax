#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _topology_2d():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(1),
            phx.discretization.UniformCellAxisSpec(1),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((1, 1), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(signature, 1),),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0), (1, 1)),),
    )
    return phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()


def test_high_order_mapped_metrics_close_and_satisfy_gcl():
    def moving_map(point, time, args):
        del args
        return jnp.asarray(((1.0 + 0.2 * time) * point[0], 3.0 * point[1]))

    geometry = phx.discretization.CanonicalMappedGeometryPlan(
        _topology_2d(),
        phx.discretization.PatchCoordinateMapSet(moving_map, "moving-affine"),
        quadrature_order=3,
        tolerance=2.0e-6,
    ).evaluate(0.4, revision=3)

    assert bool(geometry.evidence.valid)
    volume = geometry.cell_volumes[0][0][0, 0, 0]
    center = geometry.cell_centers[0][0][0, 0, 0]
    rate = geometry.mesh_volume_rates[0][0][0, 0, 0]
    np.testing.assert_allclose(volume, 3.0 * 1.08, rtol=2.0e-6)
    np.testing.assert_allclose(center, jnp.asarray((0.54, 1.5)), rtol=2.0e-6)
    np.testing.assert_allclose(rate, 0.6, rtol=2.0e-6)
    np.testing.assert_allclose(
        geometry.evidence.face_closure_defect[0][0][0, 0, 0],
        0.0,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        geometry.evidence.gcl_defect[0][0][0, 0, 0],
        0.0,
        atol=2.0e-6,
    )


def test_canonical_mapped_geometry_rejects_invalid_revision_values():
    def identity(point, time, args):
        del time, args
        return point

    plan = phx.discretization.CanonicalMappedGeometryPlan(
        _topology_2d(),
        phx.discretization.PatchCoordinateMapSet(identity, "identity-revision-test"),
    )

    for revision in (-1, 1.5, 2**31):
        with pytest.raises(ValueError, match="revision"):
            plan.evaluate(0.0, revision=revision)


def test_nonconforming_mortar_uses_one_common_physical_surface():
    def identity(point, time, args):
        del time, args
        return point

    def reverse_y(point, time, args):
        del time, args
        return jnp.asarray((point[0], 1.0 - point[1]))

    maps = phx.discretization.PatchCoordinateMapSet(
        identity,
        "identity",
        patch_maps={"right": (reverse_y, "reverse-y")},
    )
    mortar = phx.discretization.MappedMortarPlan(
        "left",
        "right",
        lambda parameter, time, args: jnp.asarray((0.5, parameter[0])),
        lambda parameter, time, args: jnp.asarray((0.5, 1.0 - parameter[0])),
        lambda parameter, time, args: jnp.asarray((0.5, parameter[0])),
        ((0.0, 1.0),),
        quadrature_order=4,
    ).prepare(maps)

    assert bool(mortar.evidence.valid)
    np.testing.assert_allclose(jnp.sum(mortar.quadrature_weights), 1.0)
    np.testing.assert_allclose(
        jnp.sum(mortar.weighted_area_vectors, axis=0),
        jnp.asarray((1.0, 0.0)),
    )


def test_nonconforming_mortar_flux_scatter_is_exactly_conservative():
    def identity(point, time, args):
        del time, args
        return point

    maps = phx.discretization.PatchCoordinateMapSet(identity, "identity")
    geometry = phx.discretization.MappedMortarPlan(
        "left",
        "right",
        lambda parameter, time, args: jnp.asarray((0.5, parameter[0])),
        lambda parameter, time, args: jnp.asarray((0.5, parameter[0])),
        lambda parameter, time, args: jnp.asarray((0.5, parameter[0])),
        ((0.0, 1.0),),
        quadrature_order=4,
    ).prepare(maps)
    system = phx.equations.EulerSystem(2)
    state = system.primitive_to_conserved(jnp.asarray((1.0, 0.0, 0.0, 1.0)))

    result = phx.discretization.MappedMortarFluxPlan(
        geometry,
        0,
        1,
        2,
    ).evaluate(
        system,
        phx.discretization.RusanovFluxPlan(),
        state,
        state,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.conservation_defect, 0.0)
    np.testing.assert_allclose(jnp.sum(result.content_rate, axis=0), 0.0)
