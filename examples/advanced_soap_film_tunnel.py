#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cylinder wake in a gravity-driven soap-film tunnel, rendered in interference colors.

A 1.5 um soap film falls at its terminal speed (gravity balanced by linear air
drag) through a 12 d x 6 d window around a d = 1 mm cylinder that pierces it.
The declared film viscosity gives Re = U d / nu_2D = 150 and the Marangoni
elasticity a film Mach number of about 0.3. After the wake develops, the
Strouhal number of the lift is compared with blockage-aware
2D circular-cylinder references. The complete force-coefficient series is
written as CSV, and the final thickness field is rendered with air-soap-air
interference colors under a uniform 6504 K Planckian environment and written
as an 8-bit sRGB PNG in the system temporary directory. Requires the optional
phydrax-meshcore library (``phydrax[meshcore]`` or
``PHYDRAX_MESHCORE_LIBRARY``).
"""

import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.soap_film_tunnel import (
    PreparedSoapFilmTunnel,
    SoapFilmInflow,
    SoapFilmTunnelGeometry,
    SoapFilmTunnelPlan,
    SoapFilmTunnelState,
)


_SECOND_RADIATION_CONSTANT = 1.438776877e-2  # m K (CODATA 2018)
_DIAMETER = 1.0e-3
_LENGTH, _WIDTH = 12.0 * _DIAMETER, 6.0 * _DIAMETER
_CENTER = (3.0 * _DIAMETER, 3.0 * _DIAMETER)
_DENSITY, _THICKNESS, _CONCENTRATION = 1000.0, 1.5e-6, 1.6e-6
_SPEED, _GRAVITY, _BULK_VISCOSITY = 1.0, 9.81, 1.0e-3
_REYNOLDS = 150.0


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
        + chunk(b"IDAT", zlib.compress(scanlines.tobytes(), 9))
        + chunk(b"IEND", b"")
    )


def _tunnel() -> PreparedSoapFilmTunnel:
    # nu_2D = (mu h + 2 mu_s) / (rho h) fixes the surface shear viscosity.
    viscosity = _SPEED * _DIAMETER / _REYNOLDS
    surface_viscosity = 0.5 * (
        _DENSITY * _THICKNESS * viscosity - _BULK_VISCOSITY * _THICKNESS
    )
    plan = SoapFilmTunnelPlan(
        SoapFilmTunnelGeometry(
            _LENGTH,
            _WIDTH,
            mesh_size_m=0.4 * _DIAMETER,
            obstacle_diameter_m=_DIAMETER,
            obstacle_center_m=_CENTER,
            rim_segments=32,
        ),
        phx.interfacial_transport.LangmuirSurfactantLaw(0.072, 298.15, 4.0e-6),
        SoapFilmInflow(
            velocity_m_s=_SPEED,
            thickness_m=_THICKNESS,
            surface_concentration_mol_m2=_CONCENTRATION,
        ),
        density_kg_m3=_DENSITY,
        # Linear drag that makes the inflow speed the terminal velocity.
        air_drag_coefficient_kg_m2_s=_DENSITY * _THICKNESS * _GRAVITY / _SPEED,
        gravity_m_s2=_GRAVITY,
        viscosity_pa_s=_BULK_VISCOSITY,
        surface_shear_viscosity_n_s_m=surface_viscosity,
        surface_dilatational_viscosity_n_s_m=surface_viscosity,
        surface_diffusivity_m2_s=1.0e-9,
    )
    return plan.prepare()


def _seeded_state(prepared: PreparedSoapFilmTunnel) -> SoapFilmTunnelState:
    """Uniform inflow plus a small cross-stream kick behind the cylinder."""
    points = np.asarray(prepared.flow.plan.surface.coordinates)
    distance = ((points[:, 0] - _CENTER[0] - _DIAMETER) / _DIAMETER) ** 2 + (
        (points[:, 1] - _CENTER[1]) / _DIAMETER
    ) ** 2
    velocity = np.zeros_like(points)
    velocity[:, 0] = _SPEED
    velocity[:, 1] = 0.1 * _SPEED * np.exp(-distance)
    return prepared.initial_state(velocity)


def _thickness_image(
    prepared: PreparedSoapFilmTunnel, state: SoapFilmTunnelState, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the P1 thickness at pixel centers with downward rays."""
    mesh = prepared.channel.mesh
    query = phx.geometry.prepare_triangle_ray_query(
        phx.geometry.TriangleRayQueryPlan(
            mesh.vertices,
            mesh.faces,
            entity_ids=np.zeros((mesh.faces.shape[0],), dtype=np.int32),
        )
    )
    width = round(height * _LENGTH / _WIDTH)
    x = (np.arange(width) + 0.5) * _LENGTH / width
    y = (np.arange(height)[::-1] + 0.5) * _WIDTH / height
    grid_x, grid_y = np.meshgrid(x, y)
    origins = np.stack((grid_x, grid_y, np.full_like(grid_x, _DIAMETER)), axis=-1)
    directions = np.broadcast_to(np.asarray((0.0, 0.0, -1.0)), origins.shape)
    hits = eqx.filter_jit(phx.geometry.intersect_triangle_rays)(
        query, origins, directions
    )
    thickness = prepared.thickness(state)
    corners = jnp.asarray(mesh.faces)[hits.triangle_indices]
    sampled = jnp.sum(hits.barycentric_coordinates * thickness[corners], axis=-1)
    return np.asarray(sampled), np.asarray(hits.successful)


def _render(thickness: np.ndarray, film: np.ndarray) -> tuple[np.ndarray, Any]:
    wavelengths = np.arange(380.0, 781.0, 5.0) / 1.0e9
    illuminant = phx.rendering.SpectralIlluminant(
        wavelengths,
        wavelengths**-5 / np.expm1(_SECOND_RADIATION_CONSTANT / (wavelengths * 6504.0)),
        illuminant_id="planckian-6504-kelvin",
    )
    appearance = phx.rendering.ThinFilmAppearancePlan(
        phx.optics.wave.ThinFilmInterferencePlan(wavelengths, 1.0, 1.33, 1.0),
        phx.rendering.SpectralColorimetryPlan(wavelengths, illuminant, exposure=4.0),
        two_sided=True,
    )
    view = jnp.asarray((0.0, 0.0, 1.0))
    result = eqx.filter_jit(appearance.evaluate)(
        jnp.where(film, thickness, _THICKNESS), view, view
    )
    encoded = np.asarray(result.colors.encoded_srgb)
    pixels = np.where(film[..., None], np.round(255.0 * encoded), 24.0)
    return pixels.astype(np.uint8), result


def run() -> dict[str, Any]:
    prepared = _tunnel()
    scales = prepared.scales
    rim_spacing = np.pi * _DIAMETER / prepared.plan.geometry.rim_segments
    step_size = 0.1 * rim_spacing / _SPEED
    steps = 4000
    initial = _seeded_state(prepared)
    result = prepared.run(initial, jnp.asarray(step_size), steps)
    state, evidence = result.state, result.evidence
    estimate = prepared.strouhal(result, transient_s=0.5 * float(state.time_s))
    reference = prepared.cylinder_wake_reference(estimate.mean_drag_coefficient)
    thickness, film = _thickness_image(prepared, state, 240)
    pixels, rendered = _render(thickness, film)
    directory = Path(tempfile.gettempdir())
    path = directory / "phydrax_soap_film_tunnel.png"
    path.write_bytes(_png(pixels))
    dynamic_force = 0.5 * _DENSITY * _THICKNESS * _SPEED**2 * _DIAMETER
    force_coefficients = np.asarray(evidence.obstacle_force_n) / dynamic_force
    lift_path = directory / "phydrax_soap_film_tunnel_lift.csv"
    np.savetxt(
        lift_path,
        np.column_stack(
            (
                np.asarray(evidence.time_s),
                force_coefficients[:, 0],
                force_coefficients[:, 1],
                np.asarray(evidence.accepted, dtype=np.int8),
            )
        ),
        delimiter=",",
        header="time_s,drag_coefficient,lift_coefficient,accepted",
        comments="",
    )
    coordinates = np.asarray(prepared.flow.plan.surface.coordinates)
    velocity = np.asarray(prepared.velocity(state))
    wake = (coordinates[:, 0] > _CENTER[0] + 0.5 * _DIAMETER) & (
        coordinates[:, 0] < _CENTER[0] + 5.0 * _DIAMETER
    )
    wake_cross_stream_rms = float(np.sqrt(np.mean(velocity[wake, 1] ** 2)))
    initial_volume = float(jnp.sum(initial.film.liquid_volume_m3))
    volume_change = float(jnp.sum(state.film.liquid_volume_m3)) - initial_volume
    accumulated_volume_exchange = step_size * float(
        jnp.sum(evidence.inflow_volume_rate_m3_s - evidence.outflow_volume_rate_m3_s)
    )
    return {
        "vertices": prepared.flow.plan.surface.topology.num_vertices,
        "reynolds_number": round(float(scales.reynolds_number), 1),
        "cell_reynolds_number": round(float(scales.cell_reynolds_number), 1),
        "inflow_film_mach_number": round(float(scales.film_mach_number), 3),
        "blockage_ratio": float(scales.blockage_ratio),
        "terminal_velocity_m_s": float(scales.terminal_velocity_m_s),
        "steps": int(evidence.status.shape[0]),
        "accepted_fraction": float(np.mean(np.asarray(evidence.accepted))),
        "simulated_time_s": float(state.time_s),
        "maximum_film_mach_number": float(jnp.max(evidence.film_mach_number)),
        "minimum_thickness_m": float(jnp.min(evidence.minimum_thickness_m)),
        "mean_inflow_volume_rate_m3_s": float(jnp.mean(evidence.inflow_volume_rate_m3_s)),
        "mean_outflow_volume_rate_m3_s": float(
            jnp.mean(evidence.outflow_volume_rate_m3_s)
        ),
        "volume_change_m3": volume_change,
        "accumulated_boundary_volume_exchange_m3": accumulated_volume_exchange,
        "volume_flux_ledger_residual_m3": volume_change - accumulated_volume_exchange,
        "maximum_volume_residual_m3": float(
            jnp.max(jnp.abs(evidence.volume_residual_m3))
        ),
        "maximum_surfactant_residual_mol": float(
            jnp.max(jnp.abs(evidence.surfactant_residual_mol))
        ),
        "maximum_momentum_residual_n_s": float(
            jnp.max(jnp.abs(evidence.momentum_residual_n_s))
        ),
        "shedding_status": estimate.status.name,
        "strouhal_number": float(estimate.strouhal_number),
        "shedding_periods": int(estimate.periods),
        "strouhal_standard_error": float(estimate.strouhal_standard_error),
        "lift_coefficient_amplitude": float(estimate.lift_coefficient_amplitude),
        "mean_drag_coefficient": float(estimate.mean_drag_coefficient),
        "unconfined_reference_strouhal_number": float(
            reference.unconfined_strouhal_number
        ),
        "blockage_corrected_reference_strouhal_number": float(
            reference.blockage_corrected_strouhal_number
        ),
        "blockage_velocity_correction": float(reference.velocity_correction),
        "corrected_reynolds_number": float(reference.corrected_reynolds_number),
        "wake_cross_stream_rms_m_s": wake_cross_stream_rms,
        "png": str(path),
        "lift_series_csv": str(lift_path),
        "image_shape": list(pixels.shape),
        "rendered_fraction": float(np.mean(np.asarray(rendered.accepted) & film)),
    }


if __name__ == "__main__":
    print(run())
