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


def _single_owner_periodic_plan(component_names: tuple[str, ...], /) -> Any:
    from phydrax._execution_resources import ExecutionGroupSpec
    from phydrax._execution_runtime import ExecutionGroup
    from phydrax.discretization._view_support import OwnerLocalCellGeometry
    from tools.meshing_qualification.scenarios.cold_owner_local_fv import (
        _owner_local_periodic_assembly,
    )

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
    plan = phx.discretization.UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        view, component_names=component_names
    )
    assert plan.periodic_face_joins is not None
    return view, plan, group


def _single_owner_periodic_geometry(component_names: tuple[str, ...], /) -> Any:
    view, plan, _ = _single_owner_periodic_plan(component_names)
    geometry = plan.prepare()
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
        self.plan = phx.discretization.UnstructuredFiniteVolumePlan(
            np.asarray(points),
            triangles=np.concatenate(cells, axis=0),
            component_names=geometry.component_names,
        )
        self.geometry = self.plan.prepare()
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


@pytest.mark.parametrize("kind", ("polynomial", "weno"))
def test_owner_local_periodic_side_traces_match_tiled_serial(kind: str) -> None:
    """A seam's neighbor trace reads the neighbor's twin facet in its own frame."""
    from phydrax.discretization import FacetTraceRule

    system = phx.equations.EulerSystem(2)
    view, geometry = _single_owner_periodic_geometry(system.component_names)
    reference = _TiledReference(view, geometry)
    lift = np.asarray(view.periodic_lift_matrix, dtype=np.float64)[:, :2]
    state = jnp.asarray(_euler_state(system, np.asarray(geometry.cell_centers), lift))
    tiled_state = jnp.asarray(reference.tile(np.asarray(state)))
    if kind == "polynomial":
        plan: Any = phx.discretization.CellPolynomialReconstructionPlan(2)
    else:
        plan = phx.discretization.UnstructuredWENOZReconstructionPlan(2)
    reconstruction: Any = plan.prepare(geometry)
    tiled: Any = plan.prepare(reference.geometry)
    domain = geometry.integration_domain("interior_facet")
    facets = np.asarray(domain.entity_indices)
    translations = np.asarray(geometry.periodic_face_translations)[facets]
    seams = np.any(translations != 0.0, axis=1)
    assert np.any(seams)
    np.testing.assert_array_equal(np.asarray(domain.periodic_face_mask), seams)
    rule = FacetTraceRule(points=3)
    traces = {}
    for side in ("owner", "neighbor"):
        if kind == "polynomial":
            trace: Any = geometry.prepare_side_trace(
                geometry.cell_space.name,
                domain,
                rule=rule,
                reconstruction=reconstruction,
                side=side,
            )
        else:
            trace = geometry.prepare_nonlinear_face_trace(
                geometry.cell_space.name,
                domain,
                rule=rule,
                reconstruction=reconstruction,
                side=side,
            )
        assert trace.descriptor.representation == "face-state"
        sites = np.asarray(trace.sites)
        cells = np.asarray(
            (geometry.owner_cells if side == "owner" else geometry.neighbor_cells)
        )[facets]
        # Both sides publish the facet's shared sites in the owner's frame;
        # the neighbor's central copy evaluates them at its lattice image.
        evaluation = sites if side == "owner" else sites + translations[:, None, :]
        if kind == "polynomial":
            expected = tiled.evaluate(
                tiled_state, jnp.asarray(cells), jnp.asarray(evaluation)
            )
        else:
            expected = tiled.evaluate_cells(
                tiled_state, jnp.asarray(cells), jnp.asarray(evaluation)
            )
        traces[side] = np.asarray(trace.apply(state))
        np.testing.assert_allclose(
            traces[side], np.asarray(expected), rtol=1e-9, atol=1e-10
        )
    # Untranslated neighbor sites would read the neighbor's polynomial far
    # outside its cell.
    untranslated = tiled.evaluate_cells if kind == "weno" else tiled.evaluate
    wrong = np.asarray(
        untranslated(
            tiled_state,
            jnp.asarray(np.asarray(geometry.neighbor_cells)[facets]),
            jnp.asarray(trace.sites),
        )
    )
    assert float(np.max(np.abs(wrong[seams] - traces["neighbor"][seams]))) > 1.0e-3
    # Sites are shared and normals opposite on both sides of every facet.
    owner_trace = geometry.prepare_side_trace(
        geometry.cell_space.name, domain, rule=rule, side="owner"
    )
    neighbor_trace = geometry.prepare_side_trace(
        geometry.cell_space.name, domain, rule=rule, side="neighbor"
    )
    np.testing.assert_array_equal(
        np.asarray(owner_trace.sites), np.asarray(neighbor_trace.sites)
    )
    np.testing.assert_array_equal(
        np.asarray(owner_trace.normals), -np.asarray(neighbor_trace.normals)
    )
    np.testing.assert_allclose(
        np.sum(
            np.asarray(owner_trace.sites) * np.asarray(owner_trace.weights)[..., None],
            axis=1,
        )
        / np.sum(np.asarray(owner_trace.weights), axis=1)[:, None],
        np.asarray(geometry.face_centers)[facets],
        atol=1e-12,
    )


def _equivariant_motion(lift: np.ndarray, /) -> Any:
    """Lattice-periodic vertex displacement: every seam twin moves as a translate."""
    inverse = jnp.asarray(np.linalg.inv(lift))

    def motion(time: Any, vertices: Any, args: Any) -> Any:
        del args
        phase = 2.0 * jnp.pi * (vertices @ inverse)
        displacement = jnp.stack(
            (jnp.sin(phase[:, 0]) * jnp.cos(phase[:, 1]), jnp.sin(phase[:, 1])),
            axis=-1,
        )
        return vertices + 0.02 * time * displacement + jnp.asarray((0.05, -0.03)) * time

    return motion


def test_owner_local_periodic_ale_geometry_matches_tiled_serial() -> None:
    """Continuous ALE evaluates lifted faces, reduces them to the seam joins."""
    system = phx.equations.EulerSystem(2)
    view, plan, group = _single_owner_periodic_plan(system.component_names)
    geometry = plan.prepare()
    reference = _TiledReference(view, geometry)
    n = geometry.cell_count
    lift = np.asarray(view.periodic_lift_matrix, dtype=np.float64)[:, :2]
    motion = _equivariant_motion(lift)
    periodic = phx.discretization.FixedConnectivityMotionPlan(
        plan, motion, mapping_id="periodic-ale"
    )
    tiled = phx.discretization.FixedConnectivityMotionPlan(
        reference.plan, motion, mapping_id="tiled-ale"
    )
    np.testing.assert_array_equal(
        np.asarray(periodic.owner_cells), np.asarray(geometry.owner_cells)
    )
    step = periodic.prepare_ssprk33_step(
        0.0,
        5.0e-4,
        "periodic-ale-epoch",
        0,
        0,
        prior_effective_cell_volumes=geometry.cell_volumes,
    )
    tiled_step = tiled.prepare_ssprk33_step(
        0.0,
        5.0e-4,
        "tiled-ale-epoch",
        0,
        0,
        prior_effective_cell_volumes=reference.geometry.cell_volumes,
    )
    assert bool(step.passed) and bool(tiled_step.passed)
    for name in ("stage_1", "stage_2", "stage_3", "accepted_geometry"):
        metrics, tiled_metrics = getattr(step, name), getattr(tiled_step, name)
        for field in (
            "effective_cell_volumes",
            "coordinate_effective_cell_volumes",
            "mesh_volume_rate",
            "cell_centers",
        ):
            np.testing.assert_allclose(
                np.asarray(getattr(metrics, field)),
                np.asarray(getattr(tiled_metrics, field))[:n],
                rtol=1e-11,
                atol=1e-12,
            )
        # Every joined face carries one quotient face's measure and normal.
        np.testing.assert_allclose(
            float(np.sum(np.asarray(metrics.effective_cell_volumes))),
            float(np.sum(np.asarray(geometry.cell_volumes))),
            rtol=1e-12,
        )
    for name in ("g1", "g2", "g3"):
        np.testing.assert_allclose(
            np.asarray(getattr(step, name)),
            np.asarray(getattr(tiled_step, name))[:n],
            rtol=1e-11,
            atol=1e-12,
        )
    assert float(np.max(np.abs(np.asarray(step.g2)))) > 1.0e-3
    # The joined faces conserve volume: a periodic patch's mesh rates sum to zero.
    np.testing.assert_allclose(np.sum(np.asarray(step.g1)), 0.0, atol=1e-13)

    # Host epochs rejoin the seams on the moved vertices.
    moved = periodic.evaluate_plan(jnp.asarray(0.25)).prepare()
    np.testing.assert_allclose(
        np.asarray(moved.cell_volumes),
        np.asarray(tiled.evaluate_plan(jnp.asarray(0.25)).prepare().cell_volumes)[:n],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        np.asarray(moved.periodic_face_translations),
        np.asarray(geometry.periodic_face_translations),
    )

    # A motion that tears a seam twin from its partner is not a periodic motion.
    assert periodic.seam_vertex_pairs is not None
    twin = periodic.seam_vertex_pairs[1][0]

    def tearing(time: Any, vertices: Any, args: Any) -> Any:
        return motion(time, vertices, args).at[twin].add(0.01 * time)

    torn = phx.discretization.FixedConnectivityMotionPlan(
        plan, tearing, mapping_id="periodic-ale-torn"
    )
    torn_step = torn.prepare_ssprk33_step(
        0.0, 0.25, "torn", 0, 0, prior_effective_cell_volumes=geometry.cell_volumes
    )
    assert not bool(torn_step.passed)
    with pytest.raises(ValueError, match="lattice translates"):
        torn.evaluate_plan(jnp.asarray(0.25))


@pytest.mark.parametrize("kind", ("constant", "polynomial"))
def test_owner_local_periodic_moving_runtime_conserves_and_preserves_free_stream(
    kind: str,
) -> None:
    """A moving owner-local periodic patch advances with seams as interior faces."""
    from phydrax.solver._finite_volume_runtime import FiniteVolumeStepPolicy

    system = phx.equations.EulerSystem(2)
    view, plan, group = _single_owner_periodic_plan(system.component_names)
    geometry = plan.prepare()
    lift = np.asarray(view.periodic_lift_matrix, dtype=np.float64)[:, :2]
    reconstruction: Any = (
        phx.discretization.PiecewiseConstantReconstruction()
        if kind == "constant"
        else phx.discretization.CellPolynomialReconstructionPlan(1).prepare(geometry)
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in geometry.boundary_patch_names
        },
    )
    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan, _equivariant_motion(lift), mapping_id=f"periodic-moving-{kind}"
    )
    dynamics: Any = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            f"periodic-moving-{kind}", "state", system, boundaries
        ),
        geometry,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            reconstruction, phx.discretization.RusanovFluxPlan()
        ),
        coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(motion=motion),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        FiniteVolumeStepPolicy(maximum_retries=2),
        execution_group=group,
        owner_local_geometry=view,
        owner_local_axis_name=f"periodic_moving_{kind}",
    )
    uniform = np.asarray(
        system.primitive_to_conserved(
            jnp.broadcast_to(jnp.asarray((1.0, 0.3, -0.2, 1.0)), geometry.state_shape)
        )
    )
    smooth = _euler_state(system, np.asarray(geometry.cell_centers), lift)
    for state in (uniform, smooth):
        initial = runtime.initialize_state(jnp.asarray(state), 0.0, 5.0e-4)
        result = runtime.advance(initial)
        assert bool(result.accepted)
        before = np.asarray(initial.content_state.conservative_content)
        content = result.runtime_state.content_state
        after = np.asarray(content.conservative_content)
        volumes = np.asarray(content.effective_cell_volumes)
        assert float(np.max(np.abs(volumes - np.asarray(geometry.cell_volumes)))) > 1e-6
        # No content leaves through a seam of the closed periodic patch.
        np.testing.assert_allclose(
            np.sum(after, axis=0), np.sum(before, axis=0), rtol=0.0, atol=1e-13
        )
        if state is uniform:
            # Seam mesh fluxes close every cell's volume law: free stream holds.
            np.testing.assert_allclose(
                after / volumes[:, None], uniform, rtol=0.0, atol=1e-13
            )
        else:
            assert float(np.max(np.abs(after - before))) > 1e-5


def _periodic_box_with_obstacle(component_names: tuple[str, ...], /) -> Any:
    """One serial owner holding a lifted unit periodic box with a square hole.

    The x = 1 and y = 1 seam faces are unwelded lattice copies of the x = 0
    and y = 0 faces; joining them leaves the obstacle as the only boundary.
    """
    from phydrax.discretization.finite_volume._unstructured import (
        _PERIODIC_SEAM_PATCH,
    )

    n = 6
    grid = np.asarray(
        [(i / n, j / n) for j in range(n + 1) for i in range(n + 1)], dtype=np.float64
    )

    def vertex(i: int, j: int, /) -> int:
        return j * (n + 1) + i

    def hole(i: int, j: int, /) -> bool:
        return 2 <= i < 4 and 2 <= j < 4

    triangles = []
    for j in range(n):
        for i in range(n):
            if hole(i, j):
                continue
            ll, lr = vertex(i, j), vertex(i + 1, j)
            ul, ur = vertex(i, j + 1), vertex(i + 1, j + 1)
            triangles.extend(((ll, lr, ur), (ll, ur, ul)))
    used = np.unique(np.asarray(triangles))
    rows = np.full((grid.shape[0],), -1, dtype=np.int64)
    rows[used] = np.arange(used.size)
    cells = rows[np.asarray(triangles)]
    seams = [
        (
            rows[[vertex(0, j), vertex(0, j + 1)]],
            rows[[vertex(n, j), vertex(n, j + 1)]],
            (1.0, 0.0),
        )
        for j in range(n)
    ] + [
        (
            rows[[vertex(i, 0), vertex(i + 1, 0)]],
            rows[[vertex(i, n), vertex(i + 1, n)]],
            (0.0, 1.0),
        )
        for i in range(n)
    ]
    obstacle = [
        rows[[vertex(*a), vertex(*b)]]
        for a, b in (((2, 2 + k), (2, 3 + k)) for k in range(2))
    ]
    obstacle += [rows[[vertex(4, 2 + k), vertex(4, 3 + k)]] for k in range(2)]
    obstacle += [rows[[vertex(2 + k, 2), vertex(3 + k, 2)]] for k in range(2)]
    obstacle += [rows[[vertex(2 + k, 4), vertex(3 + k, 4)]] for k in range(2)]
    lifted = phx.discretization.UnstructuredFiniteVolumePlan(
        grid[used],
        triangles=cells,
        # Seam faces are labelled last, as the owner-local lowering does, so a
        # host epoch rebuilds the same topology.
        boundary_patches={
            "obstacle": np.asarray(obstacle),
            _PERIODIC_SEAM_PATCH: np.concatenate(
                [np.stack((kept, twin)) for kept, twin, _ in seams]
            ),
        },
        component_names=component_names,
    )
    connectivity: Any = lifted.prepare().connectivity
    edges = np.sort(np.asarray(connectivity.edges, dtype=np.int64), axis=1)
    edge_ids = {tuple(edge): face for face, edge in enumerate(edges.tolist())}
    joins = np.asarray(
        [
            (
                edge_ids[tuple(sorted(kept.tolist()))],
                edge_ids[tuple(sorted(twin.tolist()))],
            )
            for kept, twin, _ in seams
        ],
        dtype=np.int32,
    )
    translations = np.asarray([shift for _, _, shift in seams], dtype=np.float64)
    # Relabel with the canonical face rows a host epoch rebuilds them from.
    face_vertices = np.asarray(connectivity.edges, dtype=np.int64)
    lifted = phx.discretization.UnstructuredFiniteVolumePlan(
        grid[used],
        triangles=cells,
        boundary_patches={
            "obstacle": face_vertices[
                np.asarray(lifted.patch_faces[lifted.patch_names.index("obstacle")])
            ],
            _PERIODIC_SEAM_PATCH: face_vertices[np.sort(joins.reshape(-1))],
        },
        component_names=component_names,
    )
    plan = phx.discretization.UnstructuredFiniteVolumePlan._with_owner_local_joins(
        lifted, None, joins, translations
    )
    assert plan.prepare().boundary_patch_names == ("obstacle",)
    return plan


def _box_with_obstacle(component_names: tuple[str, ...], /) -> Any:
    """The same square-hole mesh as a nonperiodic box with fixed outer walls."""
    n = 6
    grid = np.asarray(
        [(i / n, j / n) for j in range(n + 1) for i in range(n + 1)], dtype=np.float64
    )

    def vertex(i: int, j: int, /) -> int:
        return j * (n + 1) + i

    triangles = []
    for j in range(n):
        for i in range(n):
            if 2 <= i < 4 and 2 <= j < 4:
                continue
            ll, lr = vertex(i, j), vertex(i + 1, j)
            ul, ur = vertex(i, j + 1), vertex(i + 1, j + 1)
            triangles.extend(((ll, lr, ur), (ll, ur, ul)))
    used = np.unique(np.asarray(triangles))
    rows = np.full((grid.shape[0],), -1, dtype=np.int64)
    rows[used] = np.arange(used.size)
    outer = [rows[[vertex(k, j), vertex(k + 1, j)]] for k in range(n) for j in (0, n)]
    outer += [rows[[vertex(i, k), vertex(i, k + 1)]] for k in range(n) for i in (0, n)]
    obstacle = [
        rows[[vertex(i, 2 + k), vertex(i, 3 + k)]] for k in range(2) for i in (2, 4)
    ]
    obstacle += [
        rows[[vertex(2 + k, j), vertex(3 + k, j)]] for k in range(2) for j in (2, 4)
    ]
    return phx.discretization.UnstructuredFiniteVolumePlan(
        grid[used],
        triangles=rows[np.asarray(triangles)],
        boundary_patches={"obstacle": np.asarray(obstacle), "outer": np.asarray(outer)},
        component_names=component_names,
    )


_OBSTACLE_CENTER = np.asarray((0.5, 0.5))
_OBSTACLE_RATE = 2.0


def _rotating_obstacle(time: Any, vertices: Any, args: Any) -> Any:
    """The obstacle rotates rigidly about its centre; outer walls stay fixed.

    Vertex trajectories are circular, so cell volumes are not polynomial in
    time: stage volumes match the coordinates only through swept volumes.
    """
    del args
    center = jnp.asarray(_OBSTACLE_CENTER)
    angle = _OBSTACLE_RATE * time
    rotation = jnp.asarray(
        ((jnp.cos(angle), -jnp.sin(angle)), (jnp.sin(angle), jnp.cos(angle)))
    )
    on_obstacle = jnp.max(jnp.abs(vertices - center), axis=-1) < 0.25
    return jnp.where(
        on_obstacle[:, None], center + (vertices - center) @ rotation.T, vertices
    )


def _obstacle_wall_velocity(time: Any, points: Any, normal: Any, args: Any) -> Any:
    """Rigid rotation velocity ``omega x (x - c)`` at the current wall points."""
    del time, normal, args
    relative = points - jnp.asarray(_OBSTACLE_CENTER)
    return _OBSTACLE_RATE * jnp.stack((-relative[..., 1], relative[..., 0]), axis=-1)


def _obstacle_plan(topology: str, component_names: tuple[str, ...], /) -> Any:
    if topology == "periodic":
        return _periodic_box_with_obstacle(component_names)
    return _box_with_obstacle(component_names)


def test_periodic_fe_extension_moves_seam_twins_as_lattice_copies() -> None:
    """Interior motion around a moving obstacle is solved on the periodic quotient.

    The harmonic route is used here; the other routes' quotient coupling is
    covered in `test_mesh_motion_routes.py`.
    """
    system = phx.equations.EulerSystem(2)
    plan = _periodic_box_with_obstacle(system.component_names)
    geometry = plan.prepare()
    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan,
        _rotating_obstacle,
        mapping_id="periodic-obstacle-harmonic",
        motion_policy=phx.discretization.FiniteElementMeshMotionPolicy(),
    )
    assert motion.interior_motion is not None and motion.seam_vertex_pairs is not None
    kept, twin, translations = (np.asarray(part) for part in motion.seam_vertex_pairs)
    reference = np.asarray(plan.vertices)
    for time in (0.05, 0.2):
        vertices, successful = motion._vertices(jnp.asarray(time), None)
        vertices = np.asarray(vertices)
        assert bool(successful)
        # Twins share one quotient unknown: exact lattice copies after motion.
        np.testing.assert_allclose(
            vertices[twin] - vertices[kept], translations, rtol=0.0, atol=1e-15
        )
        # The seams really move: the coupling is not a trivial rigid shift.
        displacement = vertices - reference
        assert float(np.max(np.abs(displacement[twin]))) > 1.0e-4
        assert float(np.ptp(displacement[twin], axis=0).max()) > 1.0e-4
        # The host epoch rejoins the moved seams.
        moved = motion.evaluate_plan(jnp.asarray(time)).prepare()
        np.testing.assert_allclose(
            float(np.sum(np.asarray(moved.cell_volumes))),
            float(np.sum(np.asarray(geometry.cell_volumes))),
            rtol=1e-12,
        )


@pytest.mark.parametrize("route", ("harmonic", "linear_elasticity"))
@pytest.mark.parametrize("topology", ("periodic", "nonperiodic"))
def test_rotating_obstacle_stage_volumes_are_coordinate_volumes(
    topology: str, route: str
) -> None:
    """The discrete GCL holds at every SSPRK stage for a non-affine trajectory.

    Stage face rates are swept volumes between the realized stage positions,
    so the stage volume recurrence reproduces the coordinate volumes of those
    positions at rounding, for a rotating obstacle and for interior positions
    produced by an iterative elasticity solve alike.
    """
    system = phx.equations.EulerSystem(2)
    plan = _obstacle_plan(topology, system.component_names)
    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan,
        _rotating_obstacle,
        mapping_id=f"{topology}-rotating-obstacle-{route}",
        motion_policy=phx.discretization.FiniteElementMeshMotionPolicy(
            route=phx.discretization.FiniteElementMeshMotionRoute(route)
        ),
    )
    start = motion.evaluate_plan(jnp.asarray(0.3)).prepare()
    step = motion.prepare_ssprk33_step(
        0.3,
        5.0e-3,
        f"{topology}-rotating-obstacle-epoch",
        0,
        0,
        prior_effective_cell_volumes=start.cell_volumes,
    )
    assert bool(step.passed)
    scale = float(np.max(np.asarray(start.cell_volumes)))
    for name in ("stage_1", "stage_2", "stage_3", "accepted_geometry"):
        metrics = getattr(step, name)
        np.testing.assert_allclose(
            np.asarray(metrics.effective_cell_volumes),
            np.asarray(metrics.coordinate_effective_cell_volumes),
            rtol=0.0,
            atol=1e-13 * scale,
        )
        assert float(np.max(np.asarray(metrics.evidence.gcl_identity_defect))) == 0.0
    # The interior really deforms: cell volumes change along the step.
    change = np.asarray(step.accepted_geometry.coordinate_effective_cell_volumes) - (
        np.asarray(start.cell_volumes)
    )
    assert float(np.max(np.abs(change))) > 1.0e-6 * scale
    for name in ("g1", "g2", "g3"):
        # The rigid obstacle keeps the fluid volume of the box.
        np.testing.assert_allclose(
            np.sum(np.asarray(getattr(step, name))), 0.0, atol=1e-13
        )


@pytest.mark.parametrize("route", ("harmonic", "linear_elasticity"))
@pytest.mark.parametrize("topology", ("periodic", "nonperiodic"))
def test_rotating_obstacle_runtime_holds_gcl_and_conserves_mass(
    topology: str, route: str
) -> None:
    """The ALE runtime advances a rotating obstacle conservatively.

    The runtime's effective volumes are the coordinate volumes of the moved
    plan (the discrete GCL), and Euler with a moving slip wall keeps its mass:
    the wall reflects about the swept-volume grid velocity its flux uses.
    """
    system = phx.equations.EulerSystem(2)
    plan = _obstacle_plan(topology, system.component_names)
    geometry = plan.prepare()
    policies: dict[str, Any] = {
        "obstacle": phx.discretization.MovingSlipWallBoundary(
            _obstacle_wall_velocity,
            wall_velocity_provider_id="rotating-obstacle-wall",
        )
    }
    if topology == "nonperiodic":
        policies["outer"] = phx.discretization.SlipWallBoundary()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        geometry.boundary_patch_names, policies
    )
    motion = phx.discretization.FixedConnectivityMotionPlan(
        plan,
        _rotating_obstacle,
        mapping_id=f"{topology}-rotating-obstacle-runtime-{route}",
        motion_policy=phx.discretization.FiniteElementMeshMotionPolicy(
            route=phx.discretization.FiniteElementMeshMotionRoute(route)
        ),
    )
    dynamics: Any = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "rotating-obstacle-runtime", "state", system, boundaries
        ),
        geometry,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
        coupling=phx.discretization.UnstructuredFiniteVolumeCouplingPlan(motion=motion),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics, phx.discretization.FluxPositivityPlan()
    )
    state = _euler_state(system, np.asarray(geometry.cell_centers), np.eye(2))
    current = runtime.initialize_state(jnp.asarray(state), 0.0, 5.0e-3)
    before = np.asarray(current.content_state.conservative_content)
    for _ in range(2):
        result = runtime.advance(current)
        assert bool(result.accepted)
        current = result.runtime_state
    content = current.content_state
    after = np.asarray(content.conservative_content)
    volumes = np.asarray(content.effective_cell_volumes)
    assert float(np.max(np.abs(volumes - np.asarray(geometry.cell_volumes)))) > 1e-6
    # Discrete GCL: rate-advanced volumes are the moved plan's coordinate volumes.
    moved = motion.evaluate_plan(jnp.asarray(content.time)).prepare()
    np.testing.assert_allclose(
        volumes, np.asarray(moved.cell_volumes), rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        np.sum(volumes), np.sum(np.asarray(geometry.cell_volumes)), rtol=1e-13
    )
    # Seams are interior faces and both walls carry no mass.
    np.testing.assert_allclose(np.sum(after[:, 0]), np.sum(before[:, 0]), rtol=1e-13)
    assert float(np.max(np.abs(after - before))) > 1e-5
