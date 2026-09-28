#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Iridescent draining film and an explicit double-bubble rupture frame.

The three panels show a spherical film before and after a small physical
lubrication-drainage run, then a Plateau-border double bubble at the source
epoch of an accepted rupture proposal. Every E sheet uses its own sheet-slot
liquid content, dual area, and oriented manifold normal. The removed sheet and
Plateau-border pixels are diagnostic masks: no border thickness is invented.
The encoded sRGB surface fields are evaluated first and then consumed by
``SurfaceImagePlan``. Rendering hashes the physics arrays before and after to
make its downstream, non-certifying role observable.
"""

from __future__ import annotations

import struct
import tempfile
import zlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.foams import (
    BurstProposal,
    FoamRupturePlan,
    PlateauBorderPlan,
    PlateauBorderState,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
    seed_double_bubble,
    seed_sphere,
)
from phydrax.interfacial_transport import (
    FilmStepStatus,
    prepare_film_sheet_slots,
    PreparedFilmSheetSlots,
    PreparedSurfaceLubrication,
    SurfaceFilmEvidence,
    SurfaceLubricationPlan,
    SurfaceLubricationState,
)
from phydrax.rendering import ThinFilmAppearancePlan


_SECOND_RADIATION_CONSTANT = 1.438776877e-2  # m K (CODATA 2018)
_IMAGE_SHAPE = (96, 96)
_BACKGROUND = np.asarray((0.012, 0.018, 0.03), dtype=np.float64)
_UNSUPPORTED_COLOR = np.asarray((0.95, 0.08, 0.55), dtype=np.float64)
_BORDER_COLOR = np.asarray((0.08, 0.9, 1.0), dtype=np.float64)
_CONTRACT = phx.SpatialCoordinateContract(
    phx.units.METER,
    coordinate_system="cartesian-world",
    reference_frame="world",
)


def _planckian(wavelengths: np.ndarray, temperature: float) -> np.ndarray:
    return wavelengths**-5 / np.expm1(
        _SECOND_RADIATION_CONSTANT / (wavelengths * temperature)
    )


def _appearance() -> ThinFilmAppearancePlan:
    wavelengths = np.arange(380.0, 781.0, 5.0, dtype=np.float64) * 1.0e-9
    illuminant = phx.rendering.SpectralIlluminant(
        wavelengths,
        _planckian(wavelengths, 6504.0),
        illuminant_id="planckian-6504-kelvin",
    )
    return ThinFilmAppearancePlan(
        phx.optics.wave.ThinFilmInterferencePlan(wavelengths, 1.0, 1.33, 1.0),
        phx.rendering.SpectralColorimetryPlan(
            wavelengths, illuminant, exposure=4.0
        ),
        two_sided=True,
    )


def _png(pixels: np.ndarray) -> bytes:
    height, width, _ = pixels.shape
    scanlines = np.concatenate(
        (np.zeros((height, 1), dtype=np.uint8), pixels.reshape(height, 3 * width)),
        axis=1,
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines.tobytes()))
        + chunk(b"IEND", b"")
    )


def _boundary_corners(faces: np.ndarray) -> np.ndarray:
    opposite_edges = np.stack(
        (faces[:, (1, 2)], faces[:, (2, 0)], faces[:, (0, 1)]), axis=1
    )
    keys = np.sort(opposite_edges.reshape((-1, 2)), axis=1)
    _, inverse, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return (count[inverse] == 1).reshape((-1, 3))


def _render_sheet(
    vertices: np.ndarray,
    faces: np.ndarray,
    encoded_srgb: np.ndarray,
    /,
    *,
    source_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = phx.geometry.surface.SurfaceMetadata(
        source_id=source_id,
        source_revision="1",
        coordinate_contract=_CONTRACT,
        provenance=("synthetic-foam-iridescence-example",),
    )
    realization = phx.geometry.surface.SurfaceModel.from_triangles(
        vertices,
        faces,
        metadata,
        repair_orientation=True,
        orient_closed_outward=True,
    ).prepare()
    support = phx.imaging.ImagePlaneSupport(_IMAGE_SHAPE, detector_frame_id="camera")
    center = (0.5 * (_IMAGE_SHAPE[0] - 1), 0.5 * (_IMAGE_SHAPE[1] - 1))
    camera = phx.imaging.camera.CameraModel(
        phx.imaging.camera.CameraIntrinsics(
            (70.0, 70.0), center, image_shape=_IMAGE_SHAPE
        )
    )
    quantity = phx.measurement.QuantitySpec(
        "rendering",
        "thin-film-color",
        "encoded-srgb",
        phx.units.ONE,
        "rendering.thin-film-color",
    )
    layout = phx.measurement.ValueLayout(
        phx.measurement.ValueKind.VECTOR,
        (3,),
        ("red", "green", "blue"),
        component_frame_id="srgb",
    )
    prepared = phx.rendering.SurfaceImagePlan(
        realization,
        support,
        camera,
        _CONTRACT,
        quantity,
        layout,
        phx.measurement.SamplingSemantics(
            phx.measurement.SpatialSamplingKind.POINT
        ),
    ).prepare()
    rendered = prepared.render(vertices, encoded_srgb, geometry_id=source_id)
    hit = np.asarray(rendered.hit)
    triangle = np.maximum(np.asarray(rendered.primitive_ids), 0)
    boundary = _boundary_corners(faces)[triangle]
    barycentric = np.asarray(rendered.local_coordinates)
    border = hit & np.any(boundary & (barycentric < 0.075), axis=-1)
    return (
        np.asarray(rendered.prediction.values),
        np.asarray(rendered.prediction.valid_mask),
        border,
        hit,
    )


def _surface_frame(
    vertices: np.ndarray,
    faces: np.ndarray,
    encoded_srgb: np.ndarray,
    /,
    *,
    source_id: str,
) -> tuple[np.ndarray, int]:
    color, valid, border, _ = _render_sheet(
        vertices, faces, encoded_srgb, source_id=source_id
    )
    frame = np.broadcast_to(_BACKGROUND, (*_IMAGE_SHAPE, 3)).copy()
    frame[valid] = color[valid]
    frame[border & valid] = 0.25 * frame[border & valid] + 0.75 * _BORDER_COLOR
    return frame, int(np.count_nonzero(border & valid))


def _sphere_physics() -> tuple[
    PreparedSurfaceLubrication,
    SurfaceLubricationState,
    SurfaceLubricationState,
    np.ndarray,
]:
    seed = seed_sphere(1.0e-2, subdivisions=1)
    mesh = phx.geometry.TriangleMesh(seed.positions, seed.faces)
    surface = phx.interfacial_transport.prepare_film_surface(mesh)
    prepared = SurfaceLubricationPlan(
        surface,
        mobility_law="immobile-free-film",
        surface_tension_n_m=0.03,
        viscosity_pa_s=1.0e-3,
        density_kg_m3=1000.0,
        gravity_m_s2=(0.0, -9.81, 0.0),
        tolerance=1.0e-9,
        maximum_iterations=40,
    ).prepare()
    initial = prepared.initial_state(650.0e-9)
    state = initial
    status = np.asarray(FilmStepStatus.ACCEPTED, dtype=np.int32)
    for _ in range(6):
        result = prepared.step(state, 50.0)
        status = np.asarray(result.status)
        if int(status) != int(FilmStepStatus.ACCEPTED):
            raise RuntimeError(f"Spherical drainage failed with status {int(status)}.")
        state = result.state
    return prepared, initial, state, status


def _sphere_frame(
    appearance: ThinFilmAppearancePlan,
    prepared: PreparedSurfaceLubrication,
    state: SurfaceLubricationState,
    /,
    *,
    source_id: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    surface = prepared.plan.surface
    thickness = prepared.thickness(state)
    vertices = np.asarray(surface.coordinates) + np.asarray((0.0, 0.0, 0.04))
    view = -vertices
    colors = phx.rendering.thin_film_surface_colors(
        appearance,
        thickness,
        surface.vertex_normal,
        view,
        jnp.ones(thickness.shape, dtype=jnp.bool_),
    )
    frame, border_pixels = _surface_frame(
        vertices,
        np.asarray(surface.topology.mesh.faces),
        np.asarray(colors.encoded_srgb),
        source_id=source_id,
    )
    return frame, np.asarray(thickness), border_pixels


def _foam_physics() -> tuple[
    MultiRegionSurfaceState,
    PreparedFilmSheetSlots,
    PlateauBorderState,
    BurstProposal,
    int,
]:
    seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-iridescence-example"))
    base = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, base)
    slot_area = np.asarray(surface.slot_areas(base.positions))
    slots = np.asarray(topology.vertex_pair_slots)
    active = np.asarray(topology.slot_active)
    height = np.asarray(base.positions[:, 1])[:, None]
    scale = max(float(np.ptp(height)), 1.0)
    thickness = 240.0e-9 + 170.0e-9 * (height - np.min(height)) / scale
    thickness = thickness + 55.0e-9 * np.maximum(slots, 0)
    finite = np.asarray(topology.finite_region_indices)
    pairs = np.asarray(topology.region_pairs[: topology.region_pair_count])
    separating_pair = int(
        np.flatnonzero(np.all(pairs == np.sort(finite)[None, :], axis=1))[0]
    )
    separating_slots = active & (slots == separating_pair)
    thickness = np.broadcast_to(thickness, active.shape).copy()
    thickness[separating_slots] = 45.0e-9
    liquid = np.where(active, thickness * slot_area, 0.0)
    state = MultiRegionSurfaceState(
        topology,
        base.positions,
        sheet_fields=liquid[..., None],
        sheet_field_names=("film_liquid_volume",),
    )
    surface = PreparedMultiRegionSurface(topology, state)
    film_slots = prepare_film_sheet_slots(surface, state)
    edge_valence = np.sum(
        np.asarray(topology.edge_faces[: topology.edge_count]) >= 0, axis=1
    )
    border_count = int(np.count_nonzero(edge_valence == 3))
    borders = PlateauBorderPlan(
        border_edge_capacity=border_count,
        quad_point_capacity=0,
        density_kg_m3=1000.0,
        viscosity_pa_s=1.0e-3,
        surface_tension_n_m=0.03,
        time_step_s=1.0e-4,
        resource_id="foam-iridescence-example",
    ).prepare(surface, film_slots, state)
    border_state = borders.initial_state(
        liquid,
        np.zeros_like(liquid),
        np.full((border_count,), 2.0e-4),
    )
    film_evidence = SurfaceFilmEvidence(
        liquid_volume_residual_m3=jnp.asarray(0.0),
        boundary_exchange_m3=jnp.asarray(0.0),
        minimum_thickness_m=jnp.asarray(np.min(thickness[active])),
        rupture_mask=jnp.asarray(separating_slots),
        energy_change_j=jnp.asarray(0.0),
        dissipation_guaranteed=jnp.asarray(True),
        positivity_guaranteed=jnp.asarray(True),
        conductance_admissible=jnp.asarray(True),
        nonlinear_status=jnp.asarray(0, dtype=jnp.int32),
        nonlinear_iterations=jnp.asarray(0, dtype=jnp.int32),
        nonlinear_residual_norm=jnp.asarray(0.0),
        converged=jnp.asarray(True),
        finite=jnp.asarray(True),
        geometry_revision=jnp.asarray(0, dtype=jnp.int32),
    )
    proposals = FoamRupturePlan(60.0e-9).propose(
        topology,
        state,
        thickness,
        FilmStepStatus.ACCEPTED,
        film_evidence,
        0,
    )
    if len(proposals) != 1:
        raise RuntimeError(
            "The double-bubble source epoch did not yield one rupture proposal."
        )
    ruptured_pair = next(
        index
        for index, view in enumerate(film_slots.views.views)
        if tuple(sorted(view.region_ids)) == proposals[0].region_ids
    )
    return state, film_slots, border_state, proposals[0], ruptured_pair


def _foam_frame(
    appearance: ThinFilmAppearancePlan,
    film_slots: PreparedFilmSheetSlots,
    border_state: PlateauBorderState,
    ruptured_pair: int,
    /,
) -> tuple[np.ndarray, int, int, list[float]]:
    frame = np.broadcast_to(_BACKGROUND, (*_IMAGE_SHAPE, 3)).copy()
    unsupported_mask = np.zeros(_IMAGE_SHAPE, dtype=np.bool_)
    border_mask = np.zeros(_IMAGE_SHAPE, dtype=np.bool_)
    thickness_ranges: list[float] = []
    angle = np.deg2rad(32.0)
    rotation = np.asarray(
        (
            (np.cos(angle), 0.0, np.sin(angle)),
            (0.0, 1.0, 0.0),
            (-np.sin(angle), 0.0, np.cos(angle)),
        )
    )
    for sheet_index, (view, surface) in enumerate(
        zip(film_slots.views.views, film_slots.surfaces, strict=True)
    ):
        thickness = film_slots.sheet_content(
            border_state.sheet_liquid_m3, sheet_index
        ) / surface.vertex_area
        vertices = np.asarray(surface.coordinates) @ rotation.T
        vertices = vertices + np.asarray((0.0, 0.0, 6.0))
        normals = np.asarray(surface.vertex_normal) @ rotation.T
        supported = sheet_index != ruptured_pair
        support = jnp.full(thickness.shape, supported, dtype=jnp.bool_)
        colors = phx.rendering.thin_film_surface_colors(
            appearance,
            thickness,
            normals,
            -vertices,
            support,
        )
        rendered, valid, border, hit = _render_sheet(
            vertices,
            np.asarray(view.mesh.faces),
            np.asarray(colors.encoded_srgb),
            source_id=f"foam-iridescence-sheet-{sheet_index}",
        )
        if not supported:
            # NaN optical values stay invalid; geometric hits explicitly supply
            # only the unsupported sheet's diagnostic pixel support.
            unsupported_mask |= hit
        else:
            alpha = 0.68
            frame[valid] = (1.0 - alpha) * frame[valid] + alpha * rendered[valid]
        border_mask |= border
        thickness_ranges.append(float(np.ptp(np.asarray(thickness))))
    frame[unsupported_mask] = _UNSUPPORTED_COLOR
    frame[border_mask] = _BORDER_COLOR
    return (
        frame,
        int(np.count_nonzero(unsupported_mask)),
        int(np.count_nonzero(border_mask)),
        thickness_ranges,
    )


def run() -> dict[str, object]:
    appearance = _appearance()
    sphere, sphere_initial, sphere_drained, film_status = _sphere_physics()
    foam_state, film_slots, border_state, proposal, ruptured_pair = _foam_physics()
    physics = {
        "sphere_initial_liquid": sphere_initial.liquid_volume_m3,
        "sphere_drained_liquid": sphere_drained.liquid_volume_m3,
        "foam_positions": foam_state.positions,
        "foam_sheet_fields": foam_state.sheet_fields,
        "plateau_sheet_liquid": border_state.sheet_liquid_m3,
        "plateau_border_liquid": border_state.border_liquid_m3,
    }
    before = phx.array_tree_fingerprint(physics)["sha256"]
    initial_frame, initial_thickness, initial_border = _sphere_frame(
        appearance, sphere, sphere_initial, source_id="sphere-film-initial"
    )
    drained_frame, drained_thickness, drained_border = _sphere_frame(
        appearance, sphere, sphere_drained, source_id="sphere-film-drained"
    )
    foam_frame, unsupported_pixels, border_pixels, sheet_thickness_ranges = _foam_frame(
        appearance, film_slots, border_state, ruptured_pair
    )
    after = phx.array_tree_fingerprint(physics)["sha256"]
    if before != after:
        raise RuntimeError(
            "Downstream rendering mutated the physical film or foam state."
        )
    image = np.concatenate((initial_frame, drained_frame, foam_frame), axis=1)
    pixels = np.rint(255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    path = Path(tempfile.gettempdir()) / "phydrax_advanced_foam_iridescence.png"
    path.write_bytes(_png(pixels))
    color_shift = float(np.mean(np.abs(drained_frame - initial_frame)))
    if color_shift <= 1.0e-4 or unsupported_pixels == 0 or border_pixels == 0:
        raise RuntimeError(
            "The rendered frames did not expose the required optical masks."
        )
    return {
        "png": str(path),
        "image_shape": list(pixels.shape),
        "film_status": int(film_status),
        "sphere_thickness_range_initial_m": float(np.ptp(initial_thickness)),
        "sphere_thickness_range_drained_m": float(np.ptp(drained_thickness)),
        "sphere_mean_color_shift": color_shift,
        "sphere_border_pixels": initial_border + drained_border,
        "rupture_proposal_id": proposal.proposal_id,
        "foam_sheet_thickness_ranges_m": sheet_thickness_ranges,
        "unsupported_pixels": unsupported_pixels,
        "plateau_border_pixels": border_pixels,
        "physics_hash_before": before,
        "physics_hash_after": after,
        "physics_unchanged": before == after,
    }


if __name__ == "__main__":
    print(run())
