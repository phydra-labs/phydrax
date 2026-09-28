#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned external CSR oracles: Ocelot chicane tracking and PyCSR3D wakes.

Both providers are Python packages run by a caller-pinned interpreter through
:func:`phydrax.run_pinned_command`; nothing is imported into this process.
Ocelot (GPL-3.0) and PyCSR3D serve only as external reference executables.
Inputs and outputs are JSON documents with SI values.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from ..._external_runtime import PinnedExecutable, run_pinned_command
from ._beam import AcceleratorBunch
from ._csr import CSRLattice, CSRPlan


_OCELOT_SCRIPT = b"""
import json, sys
import numpy as np
from ocelot import Drift, SBend, MagneticLattice, ParticleArray, Navigator, track
from ocelot.cpbd.csr import CSR

document = json.load(open(sys.argv[1]))
cell = []
for index, (length, angle, e1, e2) in enumerate(document["elements"]):
    if angle == 0.0:
        cell.append(Drift(l=length, eid="d%d" % index))
    else:
        cell.append(SBend(l=length, angle=angle, e1=e1, e2=e2, eid="b%d" % index))
lattice = MagneticLattice(cell)
coordinates = np.asarray(document["coordinates"], dtype=float)
particles = ParticleArray(n=coordinates.shape[0])
particles.rparticles[:] = coordinates.T
particles.q_array[:] = np.asarray(document["charges"], dtype=float)
particles.E = document["energy_GeV"]
csr = CSR()
csr.traj_step = document["trajectory_step"]
csr.apply_step = document["apply_step"]
csr.n_bin = document["bins"]
csr.sigma_min = document["sigma_min"]
navigator = Navigator(lattice)
navigator.unit_step = document["apply_step"]
navigator.add_physics_proc(csr, lattice.sequence[0], lattice.sequence[-1])
_, particles = track(lattice, particles, navigator, calc_tws=False, print_progress=False)
json.dump({"coordinates": particles.rparticles.T.tolist()}, open("output.json", "w"))
"""

_PYCSR3D_SCRIPT = b"""
import json, sys
import numpy as np
from csr3d.wake import green_mesh
from csr3d.convolution import fftconvolve3

document = json.load(open(sys.argv[1]))
density = np.asarray(document["density"], dtype=float)
deltas = tuple(document["spacing"])
green = green_mesh(density.shape, deltas, rho=document["radius"], gamma=document["gamma"], component="s")
slope = np.gradient(density, deltas[2], axis=2)
(wake,) = fftconvolve3(slope, green)
json.dump({"wake": wake.tolist()}, open("output.json", "w"))
"""


def _require_python(executable: object, /) -> PinnedExecutable:
    if not isinstance(executable, PinnedExecutable):
        raise TypeError("executable must be a PinnedExecutable Python interpreter.")
    return executable


@dataclass(frozen=True, slots=True)
class OcelotCSRProvider:
    """A pinned Python interpreter with Ocelot installed (external oracle only)."""

    executable: PinnedExecutable

    def __post_init__(self) -> None:
        _require_python(self.executable)


@dataclass(frozen=True, slots=True)
class PyCSR3DProvider:
    """A pinned Python interpreter with PyCSR3D installed (external oracle only)."""

    executable: PinnedExecutable

    def __post_init__(self) -> None:
        _require_python(self.executable)


def ocelot_csr_tracking(
    provider: OcelotCSRProvider,
    lattice: CSRLattice,
    bunch: AcceleratorBunch,
    /,
    *,
    electron_charge: float,
    trajectory_step: float,
    apply_step: float,
    bins: int,
    sigma_min: float,
    timeout: float = 600.0,
) -> np.ndarray:
    """Track ``bunch`` through ``lattice`` with Ocelot's 1-D transient CSR.

    Returns final coordinates in the accelerator convention. Ocelot's ``tau``
    is taken as ``+zeta`` (``"positive-late"``) and its ``p`` as ``δ`` to first
    order; bunch energies are in joules (``ElectromagneticScaleContract.si``).
    Drift/bend elements map to ``Drift``/``SBend`` with the lattice pole faces.
    """
    if not isinstance(provider, OcelotCSRProvider):
        raise TypeError("provider must be an OcelotCSRProvider.")
    if not isinstance(lattice, CSRLattice) or not isinstance(bunch, AcceleratorBunch):
        raise TypeError("lattice and bunch must use accelerator types.")
    if bunch.convention.longitudinal_sign != "positive-late":
        raise ValueError("The Ocelot adapter maps the 'positive-late' convention.")
    lengths = np.asarray(lattice.lengths)
    angles = np.asarray(lattice.curvatures) * lengths
    entrance = np.asarray(lattice.entrance_edges)
    exit_ = np.asarray(lattice.exit_edges)
    active = np.asarray(bunch.active & bunch.valid)
    coordinates = np.asarray(bunch.coordinates, dtype=np.float64)[active]
    charges = (
        np.asarray(bunch.weights, dtype=np.float64)[active]
        * float(bunch.reference_charge)
        * electron_charge
    )
    energy = math.hypot(
        float(bunch.reference_momentum), float(bunch.reference_rest_energy)
    )
    document = {
        "elements": [
            [float(length), float(angle), float(e1), float(e2)]
            for length, angle, e1, e2 in zip(lengths, angles, entrance, exit_)
        ],
        "coordinates": coordinates.tolist(),
        "charges": np.abs(charges).tolist(),
        "energy_GeV": energy / (electron_charge * 1.0e9),
        "trajectory_step": float(trajectory_step),
        "apply_step": float(apply_step),
        "bins": int(bins),
        "sigma_min": float(sigma_min),
    }
    run = run_pinned_command(
        provider.executable,
        ("ocelot_csr.py", "input.json"),
        inputs={
            "ocelot_csr.py": _OCELOT_SCRIPT,
            "input.json": json.dumps(document).encode(),
        },
        outputs=("output.json",),
        timeout=timeout,
    ).require_success()
    result = np.asarray(
        json.loads(run.output("output.json"))["coordinates"], dtype=np.float64
    )
    if result.shape != coordinates.shape or not np.all(np.isfinite(result)):
        raise ValueError("Ocelot returned coordinates of the wrong shape or nonfinite.")
    return result


def pycsr3d_longitudinal_wake(
    provider: PyCSR3DProvider,
    plan: CSRPlan,
    density: np.ndarray,
    /,
    *,
    curvature: float,
    timeout: float = 600.0,
) -> np.ndarray:
    """PyCSR3D steady longitudinal wake of ``density`` on ``plan.grid``.

    The result has the units of :attr:`CSRWake.wake` for a unit reference
    charge prefactor ``q/(4πε₀) = 1``: PyCSR3D's mesh convolution of ``ψ_s``
    with ``∂ρ/∂z`` times ``β²/ρ`` and the cell volume.
    """
    if not isinstance(provider, PyCSR3DProvider):
        raise TypeError("provider must be a PyCSR3DProvider.")
    if not isinstance(plan, CSRPlan) or plan.model != "3d-steady-igf":
        raise ValueError("The PyCSR3D oracle compares a 3d-steady-igf plan.")
    values = np.asarray(density, dtype=np.float64)
    if values.shape != plan.shape:
        raise ValueError("density must have the plan grid shape.")
    radius = 1.0 / abs(float(curvature))
    document = {
        "density": values.tolist(),
        "spacing": list(plan.spacing),
        "radius": math.copysign(radius, float(curvature)),
        "gamma": plan.gamma,
    }
    run = run_pinned_command(
        provider.executable,
        ("pycsr3d_wake.py", "input.json"),
        inputs={
            "pycsr3d_wake.py": _PYCSR3D_SCRIPT,
            "input.json": json.dumps(document).encode(),
        },
        outputs=("output.json",),
        timeout=timeout,
    ).require_success()
    wake = np.asarray(json.loads(run.output("output.json"))["wake"], dtype=np.float64)
    if wake.shape != plan.shape or not np.all(np.isfinite(wake)):
        raise ValueError("PyCSR3D returned a wake of the wrong shape or nonfinite.")
    volume = float(np.prod(plan.spacing))
    return plan.beta**2 / radius * volume * wake


__all__ = [
    "OcelotCSRProvider",
    "PyCSR3DProvider",
    "ocelot_csr_tracking",
    "pycsr3d_longitudinal_wake",
]
