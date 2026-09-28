#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Soap-film interference colors over thickness and viewing angle.

A free-standing air-soap-air film (n = 1.33) is rendered as a color ramp:
thickness 0-1500 nm left to right, viewing angle 0-60 degrees top to bottom,
under a uniform 6504 K Planckian environment (CIE D65 itself is a CC BY-SA
host resource and is not bundled). The image is written as an 8-bit sRGB PNG in
the system temporary directory.
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


_SECOND_RADIATION_CONSTANT = 1.438776877e-2  # m K (CODATA 2018)


def _planckian(wavelengths: np.ndarray, temperature: float) -> np.ndarray:
    return wavelengths**-5 / np.expm1(
        _SECOND_RADIATION_CONSTANT / (wavelengths * temperature)
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
        + chunk(b"IDAT", zlib.compress(scanlines.tobytes(), 9))
        + chunk(b"IEND", b"")
    )


def run() -> dict[str, Any]:
    wavelengths = np.arange(380.0, 781.0, 5.0) / 1.0e9
    illuminant = phx.rendering.SpectralIlluminant(
        wavelengths,
        _planckian(wavelengths, 6504.0),
        illuminant_id="planckian-6504-kelvin",
    )
    appearance = phx.rendering.ThinFilmAppearancePlan(
        phx.optics.wave.ThinFilmInterferencePlan(wavelengths, 1.0, 1.33, 1.0),
        phx.rendering.SpectralColorimetryPlan(wavelengths, illuminant, exposure=4.0),
        two_sided=True,
    )

    thickness = jnp.linspace(0.0, 1.5e-6, 360)
    angles = jnp.deg2rad(jnp.linspace(0.0, 60.0, 72))
    views = jnp.stack(
        (jnp.zeros_like(angles), jnp.sin(angles), jnp.cos(angles)), axis=-1
    )
    result = eqx.filter_jit(appearance.evaluate)(
        thickness[None, :], jnp.asarray([0.0, 0.0, 1.0]), views[:, None, :]
    )
    encoded = np.asarray(result.colors.encoded_srgb)
    pixels = np.round(255.0 * encoded).astype(np.uint8)
    path = Path(tempfile.gettempdir()) / "phydrax_thin_film_iridescence.png"
    path.write_bytes(_png(pixels))

    normal_row = encoded[0]
    samples = {
        f"{float(thickness[column]) * 1.0e9:.0f}nm": [
            round(float(value), 3) for value in normal_row[column]
        ]
        for column in (0, 24, 72, 120, 240, 359)
    }
    return {
        "png": str(path),
        "image_shape": list(pixels.shape),
        "accepted_fraction": float(np.mean(np.asarray(result.accepted))),
        "minimum_samples_per_fringe": float(
            np.min(np.asarray(result.evidence.minimum_samples_per_fringe))
        ),
        "maximum_energy_residual": float(
            np.max(np.asarray(result.evidence.maximum_energy_residual))
        ),
        "out_of_gamut_fraction": float(np.mean(np.asarray(result.evidence.out_of_gamut))),
        "white_neutral_error": appearance.colorimetry.white_neutral_error,
        "normal_incidence_srgb": samples,
    }


if __name__ == "__main__":
    print(run())
