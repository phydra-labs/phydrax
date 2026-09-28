#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned external FBPIC oracle for quasi-cylindrical laser-wakefield runs.

FBPIC (Lehe et al., Comput. Phys. Commun. 203, 66, 2016; BSD-3-Clause) is run
by a caller-pinned Python interpreter through :func:`phydrax.run_pinned_command`;
nothing is imported into this process. The adapter maps a laser-wakefield
configuration declared in plasma units (lengths in ``c/ω_p``, times in
``1/ω_p``, fields in ``m_e c ω_p/e``) onto FBPIC's SI inputs at a declared
electron density, runs infinite-order PSATD with the curl-free current
correction, linear shapes, no current filter, periodic ``z``, a reflective
radial wall, and an immobile neutralizing ion background, and returns FBPIC's
mode-0 ``E_z`` in plasma units. SI constants come from
`ElectromagneticScaleContract.si`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from ...._external_runtime import PinnedExecutable, run_pinned_command
from ...._physical import ElectromagneticScaleContract
from ....discretization.pic import QuasiCylindricalGrid


_FBPIC_SCRIPT = b"""
import json, sys
import numpy as np
from scipy.constants import e, m_e
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser

d = json.load(open(sys.argv[1]))
sim = Simulation(
    d["Nz"], d["zmax"], d["Nr"], d["rmax"], d["Nm"], d["dt"], zmin=d["zmin"],
    n_order=-1, use_cuda=False, filter_currents=False, use_galilean=False,
    current_correction="curl-free", boundaries={"z": "periodic", "r": "reflective"},
    particle_shape="linear", verbose_level=0,
)
sim.add_new_species(
    q=-e, m=m_e, n=d["n_e"], p_nz=d["p_nz"], p_nr=d["p_nr"], p_nt=d["p_nt"],
    p_zmin=d["p_zmin"], p_zmax=d["p_zmax"], p_rmax=d["p_rmax"],
    continuous_injection=False,
)
add_laser(sim, d["a0"], d["w0"], d["ctau"], d["z0"], lambda0=d["lambda0"])
sim.step(d["steps"], show_progress=False)
field = sim.fld.interp[0]
json.dump(
    {
        "z": field.z.tolist(),
        "r": field.r.tolist(),
        "ez": np.real(field.Ez).tolist(),
    },
    open("output.json", "w"),
)
"""


@dataclass(frozen=True, slots=True)
class FBPICProvider:
    """A pinned Python interpreter with FBPIC installed (external oracle only)."""

    executable: PinnedExecutable

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be a PinnedExecutable Python interpreter.")


@dataclass(frozen=True, slots=True)
class FBPICWakefieldResult:
    """FBPIC's mode-0 ``E_z[z, r]`` on its cell-centered nodes, in plasma units."""

    axial_coordinates: np.ndarray
    radial_coordinates: np.ndarray
    longitudinal_field: np.ndarray


def fbpic_laser_wakefield(
    provider: FBPICProvider,
    grid: QuasiCylindricalGrid,
    /,
    *,
    density: float,
    a0: float,
    wavelength: float,
    waist: float,
    length: float,
    center: float,
    plasma_lower: float,
    plasma_upper: float,
    plasma_radius: float,
    particles_per_cell: tuple[int, int, int],
    time_step: float,
    steps: int,
    timeout: float = 3600.0,
) -> FBPICWakefieldResult:
    """Run one FBPIC laser-wakefield configuration declared in plasma units.

    The grid's axial nodes ``lower + kΔz`` are FBPIC's cell centers, so FBPIC's
    box starts half a cell below ``grid.lower``. The laser is FBPIC's
    ``GaussianLaser`` at focus: ``E_x = a₀ k₀ exp(−r²/w₀² − (z − z₀)²/L²)
    cos(k₀(z − z₀))`` with ``k₀ = 2π/wavelength`` (``length`` is FBPIC's
    ``cτ``). Electrons of ``density`` (SI, m⁻³) fill
    ``[plasma_lower, plasma_upper] × [0, plasma_radius]`` with
    ``particles_per_cell = (p_nz, p_nr, p_nt)``.
    """
    if not isinstance(provider, FBPICProvider):
        raise TypeError("provider must be an FBPICProvider.")
    if not isinstance(grid, QuasiCylindricalGrid):
        raise TypeError("grid must be a QuasiCylindricalGrid.")
    electron_density = float(density)
    if not math.isfinite(electron_density) or electron_density <= 0.0:
        raise ValueError("density must be finite and positive.")
    counts = tuple(int(value) for value in particles_per_cell)
    if len(counts) != 3 or any(value < 1 for value in counts) or int(steps) < 1:
        raise ValueError("particles_per_cell needs three positive counts; steps >= 1.")
    # Plasma units: k_p = ω_p/c with ω_p² = n e²/(ε₀ m_e); FBPIC is SI.
    scale = ElectromagneticScaleContract.si()
    charge = float(scale.elementary_charge)
    mass = float(scale.electron_mass)
    permittivity = float(scale.vacuum_permittivity)
    speed = float(scale.speed_of_light)
    skin = speed / math.sqrt(electron_density * charge**2 / (permittivity * mass))
    shift = 0.5 * grid.axial_spacing
    document = {
        "Nz": grid.axial_count,
        "zmin": (grid.lower - shift) * skin,
        "zmax": (grid.upper - shift) * skin,
        "Nr": grid.radial_count,
        "rmax": grid.radius * skin,
        "Nm": grid.mode_count,
        "dt": float(time_step) * skin / speed,
        "n_e": electron_density,
        "p_nz": counts[0],
        "p_nr": counts[1],
        "p_nt": counts[2],
        "p_zmin": float(plasma_lower) * skin,
        "p_zmax": float(plasma_upper) * skin,
        "p_rmax": float(plasma_radius) * skin,
        "a0": float(a0),
        "w0": float(waist) * skin,
        "ctau": float(length) * skin,
        "z0": float(center) * skin,
        "lambda0": float(wavelength) * skin,
        "steps": int(steps),
    }
    run = run_pinned_command(
        provider.executable,
        ("fbpic_wakefield.py", "input.json"),
        inputs={
            "fbpic_wakefield.py": _FBPIC_SCRIPT,
            "input.json": json.dumps(document).encode(),
        },
        outputs=("output.json",),
        timeout=timeout,
    ).require_success()
    output = json.loads(run.output("output.json"))
    field_scale = mass * speed * speed / (charge * skin)
    z = np.asarray(output["z"], dtype=np.float64) / skin
    r = np.asarray(output["r"], dtype=np.float64) / skin
    ez = np.asarray(output["ez"], dtype=np.float64) / field_scale
    if ez.shape != (grid.axial_count, grid.radial_count) or not np.all(np.isfinite(ez)):
        raise ValueError("FBPIC returned a field of the wrong shape or nonfinite.")
    return FBPICWakefieldResult(z, r, ez)


__all__ = ["FBPICProvider", "FBPICWakefieldResult", "fbpic_laser_wakefield"]
