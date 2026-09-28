"""Reflect a collimated beam from a semi-infinite scattering tissue."""

import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.optics.transport import (
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    OpticalVarianceReduction,
    prepare_optical_monte_carlo,
    simulate_optical_photons,
    TissueOpticalMedium,
)


# Tissue (medium 0, n = 1.5) fills z > 0 under an ambient half-space (medium 1).
# Lengths are centimeters: mu_a = 10 /cm, mu_s = 90 /cm, isotropic scattering.
extent = 1.0e3
surfaces = phx.optics.geometric.NonSequentialSurfaceTable(
    np.asarray(
        (
            (-extent, -extent, 0.0),
            (extent, -extent, 0.0),
            (extent, extent, 0.0),
            (-extent, extent, 0.0),
        )
    ),
    np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
    np.asarray((1, 1)),
    np.asarray((0, 0)),
    np.asarray((1.5, 1.0)),
    surface_ids=np.asarray((0, 0)),
)
medium = TissueOpticalMedium(
    np.asarray((10.0, 0.0)),
    np.asarray((90.0, 0.0)),
    np.asarray((0.0, 0.0)),
    np.asarray((1.5, 1.0)),
)
# Times are then seconds with c expressed exactly in cm/s.
centimeter_relativity = phx.RelativityScaleContract.from_si(
    phx.DimensionalScaleContract(
        phx.units.CENTIMETER,
        phx.units.KILOGRAM,
        phx.units.SECOND,
        length_coordinate_kind="physical",
    )
)
prepared = prepare_optical_monte_carlo(
    OpticalMonteCarloPlan(
        surfaces,
        medium,
        relativity=centimeter_relativity,
        maximum_interactions=400,
        photon_batch_size=2048,
        variance_reduction=OpticalVarianceReduction(
            roulette_threshold=1.0e-4, roulette_survival_probability=0.1
        ),
    )
)
count = 4096
photons = launch_optical_photons(
    np.tile((0.0, 0.0, -1.0e-3), (count, 1)),
    np.tile((0.0, 0.0, 1.0), (count, 1)),
    1,
    wavelengths=633.0e-9,
)
result = simulate_optical_photons(prepared, ExplicitPhotonSource(photons), jr.key(0))
if not bool(result.all_successful):
    raise RuntimeError("Optical transport evidence rejected the result.")
print(
    {
        "total_reflectance": float(result.tallies.escape),
        "standard_error": float(result.standard_errors.escape),
        "giovanelli_1955": 0.2600,
        "absorbed": np.asarray(result.tallies.absorption).tolist(),
        "ledger_residual": float(result.maximum_absolute_ledger_residual),
    }
)
