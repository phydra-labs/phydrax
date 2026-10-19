#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native reconstruction import-isolation consumer scenarios."""

import numpy as np

import phydrax as phx


def prepare_native_reconstruction() -> None:
    """Resolve the lazy reconstruction owner before measuring its calls."""
    phx.geometry.reconstruct_surface_region


def native_reconstruction() -> None:
    """Exercise surface, planar, and terrain reconstruction without providers."""
    rng = np.random.default_rng(0)
    sphere = rng.normal(size=(800, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    surface = phx.geometry.reconstruct_surface_region(sphere)
    phx.geometry.reconstruct_planar_region(rng.uniform(size=(200, 2)))
    grid = np.sin(np.linspace(0.0, 3.0, 12))[None, :] * np.ones((10, 1))
    phx.geometry.reconstruct_dem_region(grid)
    assert surface.report.euler_characteristic == 2
