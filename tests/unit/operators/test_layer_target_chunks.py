#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

import phydrax as phx
from phydrax.operators.integral import layer_potential as layer
from phydrax.units import METER


type Family2D = Literal["helmholtz", "combined", "modified", "stokes", "elasticity"]
type Family3D = Literal["laplace", "helmholtz", "combined", "stokes", "elasticity"]


@pytest.fixture(scope="module")
def circle() -> layer.BoundaryPanelization2D:
    geometry = phx.geometry.Circle((0.0, 0.0), 1.0).compile(coordinate_unit=METER)
    return layer.BoundaryPanelization2D(
        geometry.boundary_atlas, panels_per_chart=2, quadrature_order=2, geometry=geometry
    )


@pytest.fixture(scope="module")
def surface() -> layer.SurfacePanelization3D:
    vertices = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )
    faces = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)
    geometry = phx.geometry.MeshRegion(vertices, faces).compile(coordinate_unit=METER)
    return layer.SurfacePanelization3D(
        geometry.boundary_atlas, quadrature_order=2, geometry=geometry
    )


@pytest.mark.parametrize(
    "family", ["helmholtz", "combined", "modified", "stokes", "elasticity"]
)
@pytest.mark.parametrize("chunk_size", [1, 3, 8], ids=["one", "ragged", "oversized"])
def test_planar_direct_kernels_preserve_target_order(
    circle: layer.BoundaryPanelization2D, family: Family2D, chunk_size: int
) -> None:
    density = jnp.linspace(-0.7, 1.1, circle.node_count)
    vector_density = jnp.stack((density, density**2), axis=-1)
    targets = jnp.asarray([[2.0, 0.1], [1.8, -0.3], [2.1, 0.4], [1.7, 0.2], [2.2, -0.1]])
    direct: Callable[..., Array]
    field: Callable[[Array], Array]
    match family:
        case "helmholtz":
            scalar = layer.HelmholtzLayerPotential2D(circle, 1.3, density=density)
            direct, field = scalar._evaluate_direct, scalar
        case "combined":
            combined = layer.HelmholtzCombinedField2D(circle, 1.3, density, eta=0.8)
            direct, field = combined._evaluate_direct, combined
        case "modified":
            modified = layer.ModifiedHelmholtzLayerPotential2D(
                circle, density, decay=1.3, minimum_clearance=0.1
            )
            direct, field = modified.evaluate, modified
        case "stokes":
            stokes = layer.StokesLayerPotential2D(
                circle, vector_density, viscosity=0.8, minimum_clearance=0.1
            )
            direct, field = stokes.evaluate, stokes
        case "elasticity":
            elastic = layer.ElasticityLayerPotential2D(
                circle,
                vector_density,
                young_modulus=10.0,
                poisson_ratio=0.25,
                reduction="plane_strain",
                minimum_clearance=0.1,
            )
            direct, field = elastic.evaluate, elastic
    expected = jax.vmap(field)(targets)
    tiled = direct(targets, target_execution="chunked", target_chunk_size=chunk_size)
    assert tiled.shape == expected.shape
    assert jnp.allclose(tiled, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "family", ["laplace", "helmholtz", "combined", "stokes", "elasticity"]
)
@pytest.mark.parametrize("chunk_size", [1, 3, 8], ids=["one", "ragged", "oversized"])
def test_surface_direct_kernels_preserve_target_order(
    surface: layer.SurfacePanelization3D, family: Family3D, chunk_size: int
) -> None:
    density = jnp.linspace(-0.7, 1.1, surface.node_count)
    vector_density = jnp.stack((density, density**2, density**3), axis=-1)
    targets = jnp.asarray(
        [
            [2.0, 0.1, 0.2],
            [1.8, -0.3, 0.4],
            [2.1, 0.4, -0.2],
            [1.7, 0.2, 0.3],
            [2.2, -0.1, 0.5],
        ]
    )
    direct: Callable[..., Array]
    field: Callable[[Array], Array]
    match family:
        case "laplace":
            scalar = layer.LaplaceLayerPotential3D(surface, density=density)
            direct, field = scalar._evaluate_direct, scalar
        case "helmholtz":
            wave = layer.HelmholtzLayerPotential3D(surface, 1.3, density=density)
            direct, field = wave._evaluate_direct, wave
        case "combined":
            combined = layer.HelmholtzCombinedField3D(surface, 1.3, density, eta=0.8)
            direct, field = combined._evaluate_direct, combined
        case "stokes":
            stokes = layer.StokesLayerPotential3D(
                surface, viscosity=0.8, density=vector_density
            )
            direct, field = stokes._evaluate_direct, stokes
        case "elasticity":
            elastic = layer.ElasticityLayerPotential3D(
                surface, shear_modulus=2.0, poisson_ratio=0.25, density=vector_density
            )
            direct, field = elastic._evaluate_direct, elastic
    expected = jax.vmap(field)(targets)
    tiled = direct(targets, target_execution="chunked", target_chunk_size=chunk_size)
    assert tiled.shape == expected.shape
    assert jnp.allclose(tiled, expected, rtol=2e-12, atol=2e-12)


def test_chunked_direct_density_jvp(circle: layer.BoundaryPanelization2D) -> None:
    density = jnp.linspace(-0.7, 1.1, circle.node_count)
    targets = jnp.asarray(
        [[0.1, 0.2], [-0.2, 0.3], [0.3, -0.1], [0.4, 0.0], [-0.3, -0.2]]
    )
    potential = layer.LaplaceLayerPotential2D(circle, kind="single", density=density)

    def evaluate(values: Array) -> Array:
        return potential.with_density(values)._evaluate_direct(
            targets, target_execution="chunked", target_chunk_size=3
        )

    result, tangent = jax.jvp(evaluate, (density,), (jnp.ones_like(density),))
    squared_distance = jnp.sum(
        (targets[:, None, :] - circle.points[None, :, :]) ** 2, axis=-1
    )
    kernels = -jnp.log(squared_distance) / (4.0 * jnp.pi)
    assert jnp.allclose(
        result,
        jnp.sum(kernels * (circle.weights * density)[None, :], axis=-1),
        rtol=2e-12,
        atol=2e-12,
    )
    assert jnp.allclose(
        tangent,
        jnp.sum(kernels * circle.weights[None, :], axis=-1),
        rtol=2e-12,
        atol=2e-12,
    )


def test_chunked_direct_preserves_singular_support_refusal(
    circle: layer.BoundaryPanelization2D,
) -> None:
    potential = layer.LaplaceLayerPotential2D(circle, density=jnp.ones(circle.node_count))
    targets = jnp.asarray([[0.1, 0.2], [0.2, -0.1], [0.3, 0.0]])
    targets = targets.at[2].set(circle.points[0])
    with pytest.raises(eqx.EquinoxRuntimeError, match="intersects its singular support"):
        potential._evaluate_direct(
            targets, target_execution="chunked", target_chunk_size=2
        ).block_until_ready()
