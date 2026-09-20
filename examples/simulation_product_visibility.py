# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Turn a physical simulated Stokes image into visibilities and closures."""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    astro = phx.applications.astrophysics
    steradian = phx.units.derived_unit("sr", ((phx.units.RADIAN, 2),))
    jansky = phx.units.JANSKY
    intensity_unit = phx.units.derived_unit("Jy/sr", ((jansky, 1), (steradian, -1)))

    axis = jnp.linspace(-30.0e-6, 30.0e-6, 17)
    screen_x, screen_y = jnp.meshgrid(axis, axis, indexing="xy")
    coordinates = jnp.stack((screen_x, screen_y), axis=-1)
    spacing = axis[1] - axis[0]
    solid_angle = jnp.full(screen_x.shape, spacing**2)
    screen = astro.GRImageScreen(
        coordinates,
        solid_angle,
        jnp.ones(screen_x.shape, dtype="bool"),
        angular_unit=phx.units.RADIAN,
        solid_angle_unit=steradian,
    )

    shifted_radius = ((screen_x - 7.0e-6) / 12.0e-6) ** 2 + (
        (screen_y + 4.0e-6) / 8.0e-6
    ) ** 2
    intensity = 3.0e9 * jnp.exp(-0.5 * shifted_radius)
    polarization_angle = jnp.arctan2(screen_y, screen_x + 1.0e-30)
    stokes = jnp.stack(
        (
            intensity,
            0.2 * intensity * jnp.cos(2.0 * polarization_angle),
            0.2 * intensity * jnp.sin(2.0 * polarization_angle),
            0.03 * intensity,
        )
    )
    frequency = 230.0e9
    image = astro.StokesImage(
        stokes,
        screen,
        astro.ObservationDataProvenance.native("analytic-polarized-image"),
        frequency=frequency,
        redshift=jnp.ones(screen.pixel_shape),
        redshift_valid=jnp.ones(screen.pixel_shape, dtype="bool"),
        lensing_masks=jnp.ones((1, *screen.pixel_shape), dtype="bool"),
        lensing_labels=("direct",),
        intensity_unit=intensity_unit,
        flux_density_unit=jansky,
        frequency_unit=phx.units.HERTZ,
    )
    image_payload = astro.stokes_image_to_fits_payload(image)
    restored_image = astro.stokes_image_from_fits_payload(image_payload)

    station_uv = jnp.asarray(
        (
            (0.0, 0.0),
            (12_000.0, -5_000.0),
            (-8_000.0, 15_000.0),
            (18_000.0, 17_000.0),
        )
    )
    station_pairs = jnp.asarray(((0, 1), (1, 2), (0, 2), (2, 3), (1, 3)))
    uv_coordinates = jnp.stack(
        tuple(station_uv[second] - station_uv[first] for first, second in station_pairs)
    )
    sampling = astro.VisibilitySampling(
        uv_coordinates,
        station_pairs,
        frequency,
        ("A", "B", "C", "D"),
    )
    visibility = astro.direct_stokes_visibilities(restored_image, sampling)
    visibility_payload = astro.visibility_data_to_uvfits_payload(visibility)
    restored_visibility = astro.visibility_data_from_uvfits_payload(visibility_payload)
    polarization = astro.polarization_visibility_products(restored_visibility)
    topology = astro.ClosureTopology.from_station_cycles(
        sampling,
        phase_cycles=(("A", "B", "C"),),
        amplitude_cycles=(("A", "B", "C", "D"),),
    )
    closures = astro.closure_products(restored_visibility, topology)

    print("image_pixels", screen.pixel_shape, "valid", int(jnp.sum(image.valid_mask)))
    print("first_visibility_amplitude_Jy", float(jnp.abs(visibility.total_intensity[0])))
    print("visibility_finite", bool(visibility.finite))
    print("closure_phase_rad", float(closures.closure_phase[0]))
    print("closure_amplitude", float(closures.closure_amplitude[0]))
    print(
        "closures_physical",
        bool(closures.phase_physically_valid[0]),
        bool(closures.amplitude_physically_valid[0]),
    )
    print(
        "fractional_linear_polarization",
        complex(polarization.fractional_linear_polarization[0]),
    )
    print(
        "neutral_payload_roundtrip",
        restored_image.content_id == image.content_id,
        restored_visibility.content_id == visibility.content_id,
    )
    print(
        "product_lineage_bound",
        restored_visibility.parent_product_ids == (image.content_id,),
    )


if __name__ == "__main__":
    main()
