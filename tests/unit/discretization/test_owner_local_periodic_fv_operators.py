#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Translation-aware FV operators across owner-local periodic seams.

Every operator is compared with the same serial operator on a 5×5 tiling of
the single-owner periodic patch: the tiled copies are welded by quotient vertex
identity, and the central copy's rows are the periodic rows.
"""

from __future__ import annotations

import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._unstructured import (
    _owner_local_periodic_vertex_frames,
    periodic_image_key,
    periodic_image_scale,
)


_COPIES = 2


def _single_owner_periodic_geometry(component_names: tuple[str, ...], /) -> Any:
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from tests.unit.meshing.test_native_overset import _owner_local_periodic_assembly

    if jax.process_count() != 1:
        pytest.skip("The tiled serial reference uses one periodic owner.")
    device = jax.local_devices()[0]
    group = ExecutionGroup(
        ExecutionGroupSpec(
            "owner-local-periodic-operators",
            (jax.process_index(),),
            ((device.process_index, device.id),),
            mesh_axes=(("owner_local_periodic_operators", 1),),
        ),
        (device,),
    )
    view = (
        _owner_local_periodic_assembly(
            "periodic-operators",
            np.asarray(((0.11, 0.13), (0.58, 0.17), (0.31, 0.62), (0.84, 0.77))),
            np.zeros((4,), dtype=np.int32),
            group,
        )
        .parts[0]
        .carrier
    )
    assert isinstance(view, OwnerLocalCellGeometry)
    geometry = phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        view, component_names=component_names
    ).prepare()
    assert geometry.periodic_face_translations is not None
    assert np.any(np.asarray(geometry.periodic_face_translations) != 0.0)
    return view, geometry


class _TiledReference:
    """Serial 5×5 tiling of a single-owner periodic FV patch.

    Copies are ordered by `periodic_image_key` of their translation (the
    identity first), cell-major within a copy, so tiled row `k * n + c` is
    cell `c` at the `k`-th image and rows `0..n-1` are the periodic rows.
    """

    def __init__(self, view: Any, geometry: Any, /) -> None:
        cell_count = geometry.cell_count
        dimension = geometry.cell_dimension
        coordinates = np.asarray(geometry.vertices, dtype=np.float64)
        frames = _owner_local_periodic_vertex_frames(
            view, np.asarray(geometry.vertex_global_ids, dtype=np.int64)
        )
        assert frames is not None
        lift = np.asarray(view.periodic_lift_matrix, dtype=np.float64)[:, :dimension]
        scale = periodic_image_scale(np.asarray(geometry.cell_centers))
        shifts = sorted(
            (
                np.asarray(shift, dtype=np.int64)
                for shift in itertools.product(
                    range(-_COPIES, _COPIES + 1), repeat=dimension
                )
            ),
            key=lambda shift: periodic_image_key(shift @ lift, scale),
        )
        welded: dict[tuple[int, ...], int] = {}
        points: list[np.ndarray] = []
        cells: list[np.ndarray] = []
        triangles = np.asarray(geometry.triangles, dtype=np.int64)
        assert triangles.shape[0] == cell_count
        for shift in shifts:
            translation = shift @ lift
            rows = np.empty((coordinates.shape[0],), dtype=np.int64)
            for vertex in range(coordinates.shape[0]):
                winding = frames.windings[vertex]
                assert winding is not None
                key = (
                    int(frames.roots[vertex]),
                    *(int(value) for value in np.asarray(winding) + shift),
                )
                point = coordinates[vertex] + translation
                row = welded.get(key)
                if row is None:
                    row = len(points)
                    welded[key] = row
                    points.append(point)
                else:
                    np.testing.assert_allclose(points[row], point, atol=1.0e-12)
                rows[vertex] = row
            cells.append(rows[triangles])
        self.cell_count = cell_count
        self.translations = np.stack([shift @ lift for shift in shifts])
        self.scale = scale
        self.copy_by_image = {
            periodic_image_key(translation, scale): copy
            for copy, translation in enumerate(self.translations)
        }
        self.geometry = phx.discretization.UnstructuredFiniteVolumePlan(
            np.asarray(points),
            triangles=np.concatenate(cells, axis=0),
            component_names=geometry.component_names,
        ).prepare()
        np.testing.assert_allclose(
            np.asarray(self.geometry.cell_centers)[:cell_count],
            np.asarray(geometry.cell_centers),
            atol=1.0e-12,
        )

    def tile(self, values: np.ndarray, /) -> np.ndarray:
        return np.concatenate([values] * self.translations.shape[0], axis=0)

    def row(self, cell: int, image: np.ndarray, /) -> int:
        copy = self.copy_by_image[periodic_image_key(image, self.scale)]
        return copy * self.cell_count + int(cell)


def _euler_state(system: Any, centers: np.ndarray, lift: np.ndarray, /) -> np.ndarray:
    """Smooth nonconstant periodic primitive field at `centers`."""
    phase = 2.0 * np.pi * (centers @ np.linalg.inv(lift))
    primitive = np.zeros((centers.shape[0], 4), dtype=np.float64)
    primitive[:, 0] = 1.0 + 0.2 * np.sin(phase[:, 0]) + 0.1 * np.cos(phase[:, 1])
    primitive[:, 1] = 0.3 + 0.1 * np.cos(phase[:, 0] + phase[:, 1])
    primitive[:, 2] = -0.2 + 0.15 * np.sin(phase[:, 1])
    primitive[:, 3] = 1.0 + 0.25 * np.cos(phase[:, 0]) * np.sin(phase[:, 1])
    return np.asarray(system.primitive_to_conserved(jnp.asarray(primitive)))


def _extrapolated_dynamics(system: Any, geometry: Any, reconstruction: Any, /) -> Any:
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in geometry.boundary_patch_names
        },
    )
    return phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "owner-local-periodic-operators", "state", system, boundaries
        ),
        geometry,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            reconstruction, phx.discretization.RusanovFluxPlan()
        ),
    ).dynamics


@pytest.mark.parametrize("kind", ("polynomial", "weno"))
def test_owner_local_periodic_reconstruction_matches_tiled_serial(kind: str) -> None:
    """Stencils are cell images; neighbor traces live in the neighbor's frame."""
    from phydrax.discretization.finite_volume._geometry_protocol import (
        lower_static_unstructured_stage_metrics,
    )

    system = phx.equations.EulerSystem(2)
    view, geometry = _single_owner_periodic_geometry(system.component_names)
    reference = _TiledReference(view, geometry)

    def prepare(discretization: Any) -> Any:
        if kind == "polynomial":
            return phx.discretization.CellPolynomialReconstructionPlan(2).prepare(
                discretization
            )
        return phx.discretization.UnstructuredWENOZReconstructionPlan(2).prepare(
            discretization
        )

    periodic = prepare(geometry)
    tiled = prepare(reference.geometry)
    optimal = periodic if kind == "polynomial" else periodic.optimal
    tiled_optimal = tiled if kind == "polynomial" else tiled.optimal
    n = geometry.cell_count
    # The stencil of every periodic cell is the tiled stencil of its central copy.
    images = np.asarray(optimal.stencil_images)
    cells = np.asarray(optimal.stencil_cells)
    valid = np.asarray(optimal.stencil_valid)
    assert np.any(np.any(images != 0.0, axis=-1) & valid)
    tiled_cells = np.asarray(tiled_optimal.stencil_cells)[:n]
    tiled_valid = np.asarray(tiled_optimal.stencil_valid)[:n]
    np.testing.assert_array_equal(valid, tiled_valid)
    for cell in range(n):
        expected = [
            reference.row(member, image)
            for member, image in zip(
                cells[cell][valid[cell]], images[cell][valid[cell]], strict=True
            )
        ]
        np.testing.assert_array_equal(tiled_cells[cell][tiled_valid[cell]], expected)
    np.testing.assert_allclose(
        np.asarray(optimal.factors),
        np.asarray(tiled_optimal.factors)[:n],
        rtol=1.0e-9,
        atol=1.0e-9,
    )

    lift = np.asarray(view.periodic_lift_matrix, dtype=np.float64)[:, :2]
    state = _euler_state(system, np.asarray(geometry.cell_centers), lift)
    tiled_state = jnp.asarray(reference.tile(state))

    def central(routes: np.ndarray, points: np.ndarray, /) -> np.ndarray:
        """Serial tiled traces of central rows `routes` at their own points."""
        if kind == "polynomial":
            return np.asarray(
                tiled.evaluate(tiled_state, jnp.asarray(routes), jnp.asarray(points))
            )
        return np.asarray(
            tiled.evaluate_cells(tiled_state, jnp.asarray(routes), jnp.asarray(points))
        )

    # Periodic face points are in the owner's frame; the neighbor's central
    # copy sees them translated by the face's lattice image.
    points = np.asarray(geometry.face_quadrature_points)
    owner = np.asarray(geometry.owner_cells)
    neighbor = np.asarray(geometry.neighbor_cells)
    translations = np.asarray(geometry.periodic_face_translations)
    assert np.all(neighbor >= 0)
    seams = np.any(translations != 0.0, axis=1)
    expected_left = central(owner, points)
    expected_right = central(neighbor, points + translations[:, None, :])
    left, right = periodic.reconstruct_at(jnp.asarray(state), jnp.asarray(points))
    np.testing.assert_allclose(np.asarray(left), expected_left, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(np.asarray(right), expected_right, rtol=1e-9, atol=1e-10)
    untranslated = central(neighbor, points)
    assert float(np.max(np.abs(untranslated[seams] - expected_right[seams]))) > 1.0e-3

    # The residual is the serial flux balance of those traces.
    dynamics = _extrapolated_dynamics(system, geometry, periodic)
    residual = np.asarray(dynamics(jnp.asarray(0.0), jnp.asarray(state)))
    normal = (
        np.asarray(geometry.area_vectors) / np.asarray(geometry.face_measures)[:, None]
    )
    normal_flux, _ = dynamics._static_normal_flux(
        jnp.asarray(expected_left),
        jnp.asarray(expected_right),
        jnp.asarray(np.broadcast_to(normal[:, None, :], points.shape)),
        None,
    )
    average_flux = (
        np.sum(
            np.asarray(geometry.face_quadrature_weights)[..., None]
            * np.asarray(normal_flux),
            axis=1,
        )
        / np.asarray(geometry.face_measures)[:, None]
    )
    expected_residual = np.asarray(
        dynamics.residual_from_fluxes(jnp.asarray(average_flux))
    )
    np.testing.assert_allclose(residual, expected_residual, rtol=1e-9, atol=1e-10)
    assert float(np.max(np.abs(residual))) > 1.0e-3
    # A closed periodic patch conserves content exactly.
    volumes = np.asarray(geometry.cell_volumes)[:, None]
    np.testing.assert_allclose(np.sum(volumes * residual, axis=0), 0.0, atol=1.0e-12)

    # Stage evaluation sees the same seam frames as the static path.
    metrics = lower_static_unstructured_stage_metrics(geometry)
    stage_left, stage_right, _ = dynamics._stage_face_states(
        jnp.asarray(state), metrics, None
    )[0]
    static_left, static_right = periodic.reconstruct_at(
        jnp.asarray(state), geometry.face_quadrature_points
    )
    np.testing.assert_allclose(
        np.asarray(stage_left), np.asarray(static_left), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(stage_right), np.asarray(static_right), atol=1e-12
    )
    if kind == "polynomial":
        degree_one = phx.discretization.CellPolynomialReconstructionPlan(1)
        moving = degree_one.prepare(geometry)
        tiled_moving = degree_one.prepare(reference.geometry)
        coefficients, lengths = moving.stage_coefficients(jnp.asarray(state), metrics)
        tiled_coefficients, tiled_lengths = tiled_moving.stage_coefficients(
            jnp.asarray(tiled_state),
            lower_static_unstructured_stage_metrics(reference.geometry),
        )
        np.testing.assert_allclose(
            np.asarray(coefficients),
            np.asarray(tiled_coefficients)[:n],
            rtol=1.0e-9,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.asarray(lengths), np.asarray(tiled_lengths)[:n], rtol=1.0e-12
        )


def _stability_rates(geometry: Any, diffusivity: np.ndarray, /) -> np.ndarray:
    """Serial graph diffusion rates of a seam-free unstructured FV mesh."""
    owner = np.asarray(geometry.owner_cells)
    neighbor = np.asarray(geometry.neighbor_cells)
    centers = np.asarray(geometry.cell_centers)
    face_centers = np.asarray(geometry.face_centers)
    measures = np.asarray(geometry.face_measures)
    normal = np.asarray(geometry.area_vectors) / measures[:, None]
    interior = neighbor >= 0
    safe = np.maximum(neighbor, 0)
    owner_distance = np.abs(np.sum((face_centers - centers[owner]) * normal, axis=-1))
    neighbor_distance = np.where(
        interior,
        np.abs(np.sum((centers[safe] - face_centers) * normal, axis=-1)),
        owner_distance,
    )
    face_diffusivity = np.where(
        interior, 0.5 * (diffusivity[owner] + diffusivity[safe]), diffusivity[owner]
    )
    weight = 2.0 * measures * face_diffusivity / (owner_distance + neighbor_distance)
    rate = np.zeros((geometry.cell_count,))
    np.add.at(rate, owner, weight)
    np.add.at(rate, neighbor[interior], weight[interior])
    return rate / np.asarray(geometry.cell_volumes)


def test_owner_local_periodic_viscous_operator_matches_tiled_serial() -> None:
    """Seam neighbors enter gradients and face distances at their lattice image."""
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.2), 2
    )
    view, geometry = _single_owner_periodic_geometry(system.component_names)
    reference = _TiledReference(view, geometry)
    n = geometry.cell_count
    lift = np.asarray(view.periodic_lift_matrix, dtype=np.float64)[:, :2]
    centers = np.asarray(geometry.cell_centers)
    state = jnp.asarray(_euler_state(system, centers, lift))
    tiled_state = jnp.asarray(reference.tile(np.asarray(state)))
    plan = phx.discretization.ViscousFluxPlan()
    time = jnp.asarray(0.0)

    gradient = np.asarray(plan.unstructured_conserved_gradient(system, state, geometry))
    tiled_gradient = np.asarray(
        plan.unstructured_conserved_gradient(system, tiled_state, reference.geometry)
    )
    np.testing.assert_allclose(gradient, tiled_gradient[:n], rtol=1e-9, atol=1e-10)
    assert float(np.max(np.abs(gradient))) > 1.0e-2

    # Face fluxes average the two central-row diffusive tensors.
    evaluation = plan.evaluate_unstructured(system, time, state, geometry)
    tensor = np.asarray(
        system.diffusion_evaluation(tiled_state, jnp.asarray(tiled_gradient), None).flux
    )[:n]
    owner = np.asarray(geometry.owner_cells)
    neighbor = np.asarray(geometry.neighbor_cells)
    assert np.all(neighbor >= 0)
    normal = (
        np.asarray(geometry.area_vectors) / np.asarray(geometry.face_measures)[:, None]
    )
    expected_flux = np.einsum(
        "fcd,fd->fc", 0.5 * (tensor[owner] + tensor[neighbor]), normal
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.face_fluxes[0]), expected_flux, rtol=1e-9, atol=1e-10
    )

    # The gradient is a face sum, so every interior tiled copy is a translate and
    # the central rows of the serial residual are the periodic residual.
    residual = np.asarray(plan.unstructured_residual(system, time, state, geometry))
    tiled_residual = np.asarray(
        plan.unstructured_residual(system, time, tiled_state, reference.geometry)
    )
    np.testing.assert_allclose(residual, tiled_residual[:n], rtol=1e-9, atol=1e-10)
    assert float(np.max(np.abs(residual))) > 1.0e-3
    volumes = np.asarray(geometry.cell_volumes)[:, None]
    np.testing.assert_allclose(
        np.sum(volumes * (residual - np.asarray(evaluation.cell_source)), axis=0),
        0.0,
        atol=1.0e-12,
    )

    report = plan.unstructured_stability_report(system, state, geometry)
    diffusivity = np.broadcast_to(
        np.asarray(system.maximum_diffusivity(tiled_state, None)),
        (reference.geometry.cell_count,),
    )
    rates = _stability_rates(reference.geometry, diffusivity)[:n]
    np.testing.assert_allclose(
        float(report.maximum_diffusive_rate), float(np.max(rates)), rtol=1e-12
    )
    assert int(report.limiting_cell_flat_index) == int(np.argmax(rates))
