#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _topology():
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


def _complex():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: (points[:, 0] - 0.3) * (points[:, 0] - 0.7),
        "two-dimensional-slab",
        5,
    )
    resources = phx.discretization.BlockAMRResourcePlan(
        maximum_components_per_cell=4,
        maximum_apertures_per_face=64,
        maximum_embedded_faces_per_cell=128,
    )
    return phx.discretization.MultivaluedCutCell2DPlan(
        _topology(),
        lambda points, time, args: points,
        "identity-2d",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        resources,
        subdivision=4,
    ).prepare()


def test_two_dimensional_subcell_geometry_keeps_disconnected_components():
    complex_ = _complex()

    assert complex_.evidence.valid
    assert complex_.evidence.multivalued_cell_count == 1
    assert complex_.component_count == 2
    areas = np.sort(
        np.asarray(complex_.component_areas)[np.asarray(complex_.component_active)]
    )
    centers = np.sort(
        np.asarray(complex_.component_centers)[np.asarray(complex_.component_active), 0]
    )
    np.testing.assert_allclose(areas, (0.34, 0.34), atol=2.0e-6)
    np.testing.assert_allclose(centers, (0.17, 0.83), atol=2.0e-6)


def test_two_dimensional_multivalued_fv_preserves_uniform_wall_state():
    complex_ = _complex()
    system = phx.equations.EulerSystem(2)
    primitive = jnp.asarray((1.0, 0.0, 0.0, 1.0))
    conserved = system.primitive_to_conserved(primitive)
    state = jnp.zeros((complex_.component_capacity, system.component_count))
    state = state.at[: complex_.component_count].set(conserved)
    boundaries = {}
    for face in range(complex_.face_count):
        if int(complex_.face_neighbour_components[face]) >= 0:
            continue
        if int(complex_.face_kinds[face]) == 2:
            name = f"embedded-{int(complex_.face_body_tags[face])}"
            boundaries[name] = phx.discretization.SlipWallBoundary()
        else:
            name = (
                f"physical-{int(complex_.face_axes[face])}-"
                f"{int(complex_.face_sides[face])}"
            )
            boundaries[name] = phx.discretization.ExtrapolationBoundary()

    advanced = complex_.ssprk33_step(
        system,
        phx.discretization.RusanovFluxPlan(),
        boundaries,
        state,
        0.0,
        1.0e-4,
    )
    compiled = eqx.filter_jit(
        lambda value: complex_.ssprk33_step(
            system,
            phx.discretization.RusanovFluxPlan(),
            boundaries,
            value,
            0.0,
            1.0e-4,
        )
    )(state)

    np.testing.assert_allclose(
        advanced[: complex_.component_count],
        state[: complex_.component_count],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(compiled, advanced, rtol=2.0e-6, atol=2.0e-7)
