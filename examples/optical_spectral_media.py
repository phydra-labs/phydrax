"""Shift ultraviolet light in a scattering, dye-doped water slab."""

import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.optics.transport import (
    ExplicitPhotonSource,
    launch_optical_photons,
    MieParticles,
    OpticalMonteCarloPlan,
    prepare_optical_monte_carlo,
    RayleighScattering,
    simulate_optical_photons,
    SpectralOpticalMedium,
    WavelengthShifter,
)


# A 1 cm water slab (medium 0) between air half-spaces (medium 1). Lengths are
# centimeters and wavelengths vacuum metres, so one wavelength unit is 100 cm.
extent = 1.0e3
corners = np.asarray(
    ((-extent, -extent), (extent, -extent), (extent, extent), (-extent, extent))
)
vertices = np.concatenate(
    (np.column_stack((corners, np.zeros(4))), np.column_stack((corners, np.ones(4))))
)
surfaces = phx.optics.geometric.NonSequentialSurfaceTable(
    vertices,
    np.asarray(((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)), dtype=np.int32),
    np.asarray((1, 1, 0, 0)),  # medium on the negative-z side of each triangle
    np.asarray((0, 0, 1, 1)),  # medium on the positive-z side
    np.asarray((1.33, 1.0)),
    surface_ids=np.asarray((0, 0, 1, 1)),
)
wavelengths = np.asarray((300e-9, 350e-9, 400e-9, 450e-9, 500e-9, 550e-9, 600e-9))
nodes = wavelengths.shape[0]
absent = np.full(nodes, np.inf)
medium = SpectralOpticalMedium(
    wavelengths,
    np.stack(((1.349, 1.343, 1.339, 1.337, 1.335, 1.333, 1.332), np.ones(nodes))),
    np.stack((np.full(nodes, 50.0), absent)),
    # Molecular Rayleigh scattering, lambda^-4 from 20 cm at 400 nm.
    rayleigh=RayleighScattering(np.stack((20.0 * (wavelengths / 400e-9) ** 4, absent))),
    # 0.25 um polystyrene spheres at 1e8 per cubic centimeter.
    mie=MieParticles(
        np.asarray((0.25e-4, 0.25e-4)),
        np.full((2, nodes), 1.59 + 0.0j),
        np.asarray((1.0e8, 0.0)),
        length_per_wavelength_unit=100.0,
    ),
    # A dye absorbing below 400 nm and emitting a 450-550 nm band.
    wavelength_shifter=WavelengthShifter(
        np.stack((np.where(wavelengths <= 400e-9, 0.05, np.inf), absent)),
        np.asarray((450e-9, 500e-9, 550e-9)),
        np.asarray(((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))),
        np.asarray((0.9, 0.0)),
        np.asarray((2.0e-9, 0.0)),
    ),
)
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
        surfaces, medium, relativity=centimeter_relativity, maximum_interactions=200
    )
)
count = 4096
photons = launch_optical_photons(
    np.tile((0.0, 0.0, -1.0e-3), (count, 1)),
    np.tile((0.0, 0.0, 1.0), (count, 1)),
    1,
    wavelengths=350.0e-9,
)
result = simulate_optical_photons(prepared, ExplicitPhotonSource(photons), jr.key(0))
if not bool(result.all_successful):
    raise RuntimeError("Optical transport evidence rejected the result.")
evidence = medium.mie_evidence
if evidence is None:
    raise RuntimeError("The Mie process was not prepared.")
print(
    {
        "net_flux_into_slab": float(result.tallies.surface_flux[0]),
        "transmitted_through_back": float(result.tallies.surface_flux[1]),
        "escaped": float(result.tallies.escape),
        "absorbed": np.asarray(result.tallies.absorption).tolist(),
        "ledger_residual": float(result.maximum_absolute_ledger_residual),
        "mie_size_parameters": np.asarray(evidence.size_parameters[0]).round(3).tolist(),
        "mie_table_residual": float(np.max(evidence.normalization_residuals)),
    }
)
