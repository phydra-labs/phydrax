#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Laser-wakefield stage simulated in a Lorentz-boosted frame (γ_b = 2).

A one-dimensional stage in code units (c = 1, laser wavelength 1): a
cos²-envelope pulse with a₀ = 1 drives a wake in a 16-long plasma of density
n/n_c = 0.01; 21 test electrons at γ = 20 trail the pulse across one plasma
period. The lab plasma is loaded into Galilean coordinates comoving with the
boosted plasma, the laser is sampled on the initial boosted slice, and the run
reports the lab-frame energy gain of each witness at the plasma exit (from
lab-frame tracks), a back-transformed lab snapshot of the wake, and the NCI
guard evidence.
"""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization.pic import (
    ExternalFieldSample,
    PIC_CODE_RELATIVITY,
    PICTrackRecorder,
)
from phydrax.units import CHARGE, UnitDefinition


D = phx.discretization
spectral = phx.solver.maxwell.spectral

GAMMA_BOOST = 2.0
BETA_BOOST = float(np.sqrt(1.0 - 1.0 / GAMMA_BOOST**2))
WAVENUMBER = 2.0 * np.pi
PLASMA_FREQUENCY2 = 0.01 * WAVENUMBER**2
PLASMA = (26.0, 42.0)
WITNESSES = np.linspace(5.0, 14.5, 21)
WITNESS_GAMMA = 20.0
EXIT = 43.0
LAB_SNAPSHOT_TIME = 30.0


class PlaneLaser(phx.StrictModule):
    """Lab vacuum pulse E_x = c B_y = E₀ cos²(πξ/2L) cos(kξ), ξ = z − ct − z₀."""

    @property
    def source_id(self) -> str:
        return "boosted-lwfa-example-laser"

    def external_fields(
        self, positions: jax.Array, times: jax.Array, /
    ) -> ExternalFieldSample:
        xi = positions[:, 2] - times - 20.0
        envelope = jnp.where(jnp.abs(xi) < 5.0, jnp.cos(0.1 * jnp.pi * xi) ** 2, 0.0)
        value = WAVENUMBER * envelope * jnp.cos(WAVENUMBER * xi)
        zero = jnp.zeros_like(value)
        return ExternalFieldSample(
            jnp.stack((value, zero, zero), axis=-1),
            jnp.stack((zero, value, zero), axis=-1),
            jnp.ones(value.shape, dtype=jnp.bool_),
        )


def species(
    offset: int, count: int, specific: float, name: str
) -> tuple[D.pic.PICSpeciesPlan, D.PreparedChargedParticles]:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + count), jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(specific * jnp.ones((count,)), name).prepare(support)
    plan = D.pic.PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        D.pic.PICChargeModelPlan(
            specific,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )
    return plan, charged


frame = phx.solver.BoostedFramePlan(
    phx.LorentzFrame(phx.boost_matrix(jnp.asarray([0.0, 0.0, BETA_BOOST]))),
    phx.geometry.Box([1.5, 1.5, 24.0], [3.0, 3.0, 48.0]),
)
# The boosted slice through the lab plasma entrance at t = 0: plasma unperturbed,
# laser in vacuum.
start, _ = frame.from_lab(0.0, jnp.asarray([1.5, 1.5, PLASMA[0]]))
beta_w = float(np.sqrt(1.0 - 1.0 / WITNESS_GAMMA**2))
travel = (EXIT + 0.5 - WITNESSES[0]) / beta_w
stop, _ = frame.from_lab(travel, jnp.asarray([1.5, 1.5, EXIT + 0.5]))

# About 16 cells per Doppler-stretched wavelength; contracted plasma edges on nodes.
entrance, exit_ = (value / GAMMA_BOOST for value in PLASMA)
wavelength = GAMMA_BOOST * (1.0 + BETA_BOOST)
cells = round((exit_ - entrance) * 16 / wavelength)
spacing = (exit_ - entrance) / cells
lanes = WITNESSES.size
witness_lab = np.stack((np.full(lanes, 1.5), np.full(lanes, 1.5), WITNESSES), axis=-1)
witness_velocity = np.zeros((lanes, 3))
witness_velocity[:, 2] = beta_w
probe = frame.boost_particles(witness_lab, witness_velocity, boosted_time=start)
trailing = float(
    jnp.min(probe.positions[:, 2]) - float(start) * frame.galilean_velocity[2]
)
lower = entrance - np.ceil((entrance - trailing + 4.0) / spacing) * spacing
count = int(np.ceil((30.0 - lower) / spacing))
upper = lower + count * spacing

column = entrance + (np.arange(4 * cells) + 0.5) * spacing / 4
plasma = np.stack(
    (np.full(column.size, 1.5), np.full(column.size, 1.5), GAMMA_BOOST * column),
    axis=-1,
)
weight = PLASMA_FREQUENCY2 * 9.0 * GAMMA_BOOST * spacing / 4
electrons, electron_charge = species(0, lanes + plasma.shape[0], -1.0, "electrons")
ions, ion_charge = species(10**6, plasma.shape[0], 1.0 / 1836.0, "ions")
masses = (
    np.concatenate((np.full(lanes, 1.0e-9 * weight), np.full(plasma.shape[0], weight))),
    np.full(plasma.shape[0], 1836.0 * weight),
)
lab_velocity = np.zeros((lanes + plasma.shape[0], 3))
lab_velocity[:lanes] = witness_velocity
boosted_electrons = frame.boost_particles(
    np.concatenate((witness_lab, plasma)), lab_velocity, boosted_time=start
)
boosted_ions = frame.boost_particles(plasma, np.zeros_like(plasma), boosted_time=start)

grid = D.TensorGridPlan(
    (
        D.UniformCellAxisSpec(3, periodic=True),
        D.UniformCellAxisSpec(3, periodic=True),
        D.UniformCellAxisSpec(count, periodic=True),
    ),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0, 0.0, lower], [3.0, 3.0, upper]]))
bridge = D.StructuredCochainBridge(grid)
transfer = D.pic.PICParticleCochainTransferPlan(bridge, shape_order=1)
transfers = tuple(transfer.prepare(value) for value in (electron_charge, ion_charge))
currents = tuple(D.pic.ChargeConservingCurrentPlan(value) for value in transfers)
solver = spectral.SpectralMaxwellPlan(
    bridge,
    variant="galilean",
    galilean_velocity=frame.galilean_velocity,
    charge_conservation="update-with-rho",
).prepare(transfers, currents)

dt = 0.45 * spacing / 1.85
steps = int(np.ceil((float(stop) - float(start)) / dt))
recorder = PICTrackRecorder(
    (electrons, ions),
    np.zeros(lanes, dtype=np.int32),
    (np.zeros(lanes, dtype=np.uint32), np.arange(lanes, dtype=np.uint32)),
    relativity=PIC_CODE_RELATIVITY,
    sample_capacity=steps,
)
boosted = frame.prepare(
    phx.solver.ElectromagneticPICPlan(
        solver, species=(electrons, ions), recorders=(recorder,)
    ),
    snapshots=phx.solver.BoostedSnapshotPlan((LAB_SNAPSHOT_TIME,), species=(0,)),
)
state = boosted.initialize(
    (boosted_electrons.positions, boosted_ions.positions),
    (boosted_electrons.velocities, boosted_ions.velocities),
    dt,
    time=start,
    masses=masses,
    vacuum_fields=(PlaneLaser(),),
)


def body(
    value: phx.solver.BoostedFrameState, _: None
) -> tuple[phx.solver.BoostedFrameState, jax.Array]:
    result = boosted.step_detailed(value, dt)
    return result.accepted_state, result.successful


final, successful = jax.jit(lambda value: jax.lax.scan(body, value, None, length=steps))(
    state
)
if not bool(jnp.all(successful)):
    raise RuntimeError("A boosted PIC step was rejected.")

scale = phx.ElectromagneticScaleContract.code_units(
    PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=1,
    boltzmann_constant=1,
    elementary_charge=1,
    electron_mass=1,
    vacuum_permittivity=1,
    constant_set_id="boosted-lwfa-example",
)
tracks = boosted.lab_trajectory(final, 0, scale)
positions = np.asarray(tracks.positions)
proper = np.asarray(tracks.proper_velocities)
active = np.asarray(tracks.active)
gains = np.asarray(
    [
        np.interp(
            EXIT,
            positions[active[:, lane], lane, 2],
            np.sqrt(1.0 + np.sum(proper[active[:, lane], lane] ** 2, axis=-1)),
        )
        - WITNESS_GAMMA
        for lane in range(lanes)
    ]
)
snapshot = boosted.lab_field_snapshot(final, 0)
filled = np.asarray(snapshot.filled)
wake = np.asarray(snapshot.electric[1, 1, :, 2])[filled]
lab_cells, lab_steps = 48 * 16, int(np.ceil(travel / (0.45 / 16)))
print(
    {
        "boosted_cells": count,
        "boosted_steps": steps,
        "equivalent_lab_cells": lab_cells,
        "equivalent_lab_steps": lab_steps,
        "witness_gain_min": float(gains.min()),
        "witness_gain_max": float(gains.max()),
        "lab_snapshot_planes": int(filled.sum()),
        "lab_snapshot_peak_wake": float(np.max(np.abs(wake))),
        "nci_rejections": int(final.evidence.nci_rejections),
        "nci_high_k_fraction": float(final.evidence.nci_fraction),
    }
)
