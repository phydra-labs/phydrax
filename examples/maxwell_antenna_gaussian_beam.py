#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Focused Gaussian beam launched by a one-way sampled plane antenna.

A y-uniform (cylindrical) paraxial Gaussian pulse is sampled half a Rayleigh range
upstream of its waist and launched along +z by a `SampledPlaneCurrentAntennaPlan`.
Transient spectra ``∫ E_x(t) e^{+iωt} dt`` on downstream planes give the beam width
``2σ`` and the on-axis phase, which are compared with the paraxial beam propagated with
the Yee lattice's axial wavenumber ``k̃`` (``sin(k̃/2) = sin(ωΔt/2)/Δt``) and
diffraction wavenumber ``sin k̃``: ``w(z) = w₀√(1 + (z/z_R)²)`` and the Gouy phase
``−½ arctan(z/z_R)``. The antenna work ledger is compared with the field energy.
"""

import jax.numpy as jnp
import numpy as np

import phydrax as phx


mx = phx.solver.maxwell
columns, cells, periodic_width = 140, 170, 4.0
grid = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(columns),
        phx.discretization.UniformCellAxisSpec(2, periodic=True),
        phx.discretization.UniformCellAxisSpec(cells),
    ),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0, 0.0, 0.0], [columns, 2 * periodic_width, cells]]))
bridge = phx.discretization.StructuredCochainBridge(grid)
omega = 2.0 * np.pi / 12.0  # 12 cells per wavelength
step = 0.95 * float(phx.solver.CompatibleMaxwellPlan(bridge).prepare().stable_dt)
axial = 2.0 * np.arcsin(np.sin(0.5 * omega * step) / step)
diffraction = np.sin(axial)
waist, antenna_plane = 18.0, 10.0
rayleigh = 0.5 * diffraction * waist**2
distance = 0.5 * rayleigh
focus = antenna_plane + distance

first = np.linspace(0.0, columns, 281)
offset = first - 0.5 * columns
width = waist * np.sqrt(1.0 + (distance / rayleigh) ** 2)
profile = np.sqrt(waist / width) * np.exp(
    -((offset / width) ** 2)
    - 0.5j * diffraction * distance * offset**2 / (distance**2 + rayleigh**2)
    + 0.5j * np.arctan(distance / rayleigh)
)
times = np.linspace(0.0, 80.0, 801)
electric = np.zeros((first.size, 2, times.size, 2), dtype=np.complex128)
electric[..., 0] = profile[:, None, None] * np.exp(-(((times - 36.0) / 12.0) ** 2))
antenna = mx.SampledPlaneCurrentAntennaPlan(
    bridge,
    2,
    antenna_plane,
    first,
    [-10.0, 20.0],
    times,
    electric,
    carrier_angular_frequency=omega,
)

planes = np.asarray([16, 34, 52, 70, 88, 106, 124])
shape = bridge.orientation_shapes[1][0]
cell_index, plane_index = np.meshgrid(np.arange(columns), planes, indexing="xy")
edges = bridge.orientation_offsets[1][0] + np.ravel_multi_index(
    np.stack(
        (
            cell_index.reshape(-1),
            np.zeros(cell_index.size, dtype=np.int64),
            plane_index.reshape(-1),
        )
    ).astype(np.int64),
    shape,
)
acquisition = mx.MaxwellSpectralAcquisition(
    np.asarray([omega]), sign="positive", measure="time-integral"
)
runtime = phx.solver.CompatibleMaxwellPlan(
    bridge,
    sources=(antenna,),
    observers=(
        mx.DFTObserverPlan(mx.FieldProbePlan("electric", edges), acquisition),
        mx.MaxwellAntennaWorkObserverPlan(antenna),
    ),
).prepare()
# Stop before the reflection from the z = 170 boundary reaches any plane.
result = mx.solve_compatible_maxwell(
    runtime, runtime.initialize(), 0.0, step, int(190.0 / step)
)

phasors = np.asarray(result.observations[0])[0].reshape(planes.size, columns)
centers = np.arange(columns) + 0.5 - 0.5 * columns
intensity = np.abs(phasors) ** 2
widths = 2.0 * np.sqrt(np.sum(centers**2 * intensity, 1) / np.sum(intensity, 1))
axis = 0.5 * (phasors[:, columns // 2 - 1] + phasors[:, columns // 2])
gouy = np.angle(axis * np.exp(-1j * axial * (planes - antenna_plane)))
source, work_observer = runtime.sources[0], runtime.observers[1]
if not isinstance(source, mx.PreparedSampledPlaneCurrentAntenna) or not isinstance(
    work_observer, mx.PreparedMaxwellAntennaWorkObserver
):
    raise TypeError("The prepared runtime must carry the antenna and its work ledger.")
evidence = source.evidence
print(
    f"focus z={focus:.2f}  z_R={rayleigh:.2f}  magnetic projection elided: "
    f"{runtime.magnetic_projection_elided}  closure defect="
    f"{float(evidence.magnetic_closure_defect):.1e}"
)
for plane, measured, phase in zip(planes, widths, gouy, strict=True):
    z = plane - focus
    print(
        f"z-z_f={z:7.2f}  w={measured:6.3f} (paraxial {waist * np.hypot(1.0, z / rayleigh):6.3f})"
        f"  Gouy={phase:+.4f} (paraxial {-0.5 * np.arctan(z / rayleigh):+.4f})"
    )
work = work_observer.evidence(result.final_state.observations[1])
print(
    f"antenna work={float(work.total_work):.6e}  "
    f"field energy={float(runtime.energy(result.final_state)):.6e}"
)
