#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Substituting the cochain PIC field solver with Cartesian PSATD in one declaration.

A cold electron–ion Langmuir oscillation runs through one explicit
`ElectromagneticPICPlan` declaration in which only the prepared field solver
differs: the species, prepared transfers, current plans, runtime options,
initial particles, and step size are the same objects and values. The
substitution constructs a new prepared solver; it does not convert a running
field state.

Electrons start on a one-per-cell quiet lattice at the ``E_x`` edge positions,
displaced by ``δ sin(k x̄)``. For this lattice the quadratic node deposit of a
displacement is its node difference and the linear edge gather samples its own
edge, so the cochain Gauss law inverts the deposit exactly and the continuous-
time Langmuir frequency is ``ω_p√(1 + m_e/m_i)``. PSATD inverts it with the
exact wavenumber, which scales ``ω²`` by ``sin(kΔ/2)/(kΔ/2)``. Both share the
explicit leapfrog of the longitudinal mode, so each measured frequency is
``(2/Δt) asin(ω Δt/2)`` of its own continuous frequency, and the two solvers
converge to each other at second order in ``kΔ``. The step is fixed across
resolutions, so only the spatial error varies.
"""

from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx


D = phx.discretization

LENGTH = 1.0
TRANSVERSE_CELLS = 4
RESOLUTIONS = (16, 32, 64)
ION_MASS_RATIO = 1836.0
WAVENUMBER = 2.0 * np.pi / LENGTH
# Small enough that the first-order knot-crossing nonlinearity of the quadratic
# shape (relative frequency shift ≲ k²Δδ) stays below 1e-6.
DISPLACEMENT = 1.0e-6 * LENGTH
# One fixed step, 0.4 of the 3-D Yee Courant limit of the finest grid.
STEP = 0.4 * (LENGTH / RESOLUTIONS[-1]) / np.sqrt(3.0)
PLASMA_FREQUENCY = 2.0 * np.pi / (64 * STEP)
STEPS = 160


class LangmuirCase(NamedTuple):
    spacing: float
    bridge: Any
    species: tuple[Any, ...]
    transfers: tuple[Any, ...]
    currents: tuple[Any, ...]
    lattice: np.ndarray
    electrons: np.ndarray


def langmuir_case(cells: int) -> LangmuirCase:
    """Periodic ``cells × 4 × 4`` box, one electron and one ion per cell."""
    h = LENGTH / cells
    counts = (cells, TRANSVERSE_CELLS, TRANSVERSE_CELLS)
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(n, periodic=True) for n in counts),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * h for n in counts]]))
    bridge = D.StructuredCochainBridge(grid)
    ix, iy, iz = np.meshgrid(*(np.arange(n) for n in counts), indexing="ij")
    lattice = np.stack(((ix + 0.5) * h, (iy + 0.5) * h, (iz + 0.5) * h), axis=-1)
    lattice = lattice.reshape(-1, 3)
    count = lattice.shape[0]
    weight = PLASMA_FREQUENCY**2 * h**3  # ω_p² = n q²/(ε m) with q = m = weight
    species, charged = [], []
    for offset, specific, name, mass in (
        (0, -1.0, "electrons", weight),
        (10**6, 1.0 / ION_MASS_RATIO, "ions", ION_MASS_RATIO * weight),
    ):
        support = D.ParticleSetPlan(
            jnp.arange(offset, offset + count),
            mass * jnp.ones((count,)),
            ambient_dimension=3,
        ).prepare()
        charged.append(
            D.ChargedParticlePlan(specific * mass * jnp.ones((count,)), name).prepare(
                support
            )
        )
        species.append(
            D.pic.PICSpeciesPlan(
                D.ParticlePopulationPlan(support),
                D.pic.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    transfer_plan = D.pic.PICParticleCochainTransferPlan(bridge, shape_order=2)
    transfers = tuple(transfer_plan.prepare(value) for value in charged)
    currents = tuple(D.pic.ChargeConservingCurrentPlan(value) for value in transfers)
    electrons = lattice.copy()
    electrons[:, 0] += DISPLACEMENT * np.sin(WAVENUMBER * lattice[:, 0])
    return LangmuirCase(
        h, bridge, tuple(species), transfers, currents, lattice, electrons
    )


def cochain_field_solver(case: LangmuirCase) -> Any:
    maxwell = phx.solver.CompatibleMaxwellPlan(
        case.bridge,
        sources=(phx.solver.PICMaxwellCurrentSourcePlan(),),
        plan_id="langmuir-cochain",
    ).prepare()
    electrostatic = phx.solver.CochainElectrostaticPlan(
        case.bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(case.bridge)
    )
    return phx.solver.CochainMaxwellPICFieldSolver(
        maxwell, electrostatic, case.transfers, case.currents
    )


def psatd_field_solver(case: LangmuirCase) -> Any:
    # Infinite-order staggered PSATD: E_x on the same Yee edges the transfers
    # gather from, and the default spectral correction keeps Gauss's law for the
    # solver's own divergence at roundoff with the unchanged Esirkepov deposit.
    return phx.solver.maxwell.spectral.SpectralMaxwellPlan(
        case.bridge, grid="staggered"
    ).prepare(case.transfers, case.currents)


def langmuir_pic(case: LangmuirCase, solver: Any) -> phx.solver.ElectromagneticPICPlan:
    """The one PIC declaration; the prepared field solver is its only argument."""
    return phx.solver.ElectromagneticPICPlan(solver, species=case.species)


def run(case: LangmuirCase, solver: Any) -> dict[str, np.ndarray]:
    pic = langmuir_pic(case, solver)
    x0 = jnp.asarray(case.lattice[:, 0])
    velocity = np.zeros(case.lattice.shape)
    state = eqx.filter_jit(
        lambda: pic.initialize((case.electrons, case.lattice), (velocity, velocity), STEP)
    )()
    # The net charge is a kδ-small difference of the species densities, so the
    # Gauss residual is reported relative to the unsigned electron density.
    electrons = state.species[0]
    density, _ = solver.deposit_charge(
        0,
        electrons.particles.position,
        pic.species[0].macrocharge(electrons),
        electrons.population.active,
    )
    scale = jnp.max(jnp.abs(density))

    def amplitude(position: Array) -> Array:
        shift = jnp.mod(position[:, 0] - x0 + 0.5 * LENGTH, LENGTH) - 0.5 * LENGTH
        return 2.0 * jnp.mean(shift * jnp.sin(WAVENUMBER * x0))

    def body(value: Any, _: None) -> tuple[Any, dict[str, Array]]:
        result = pic.step_detailed(value, STEP)
        accepted = result.accepted_state
        diagnostics = result.diagnostics
        synchronized = pic.synchronized_energy(accepted, STEP)
        return accepted, {
            "amplitude": amplitude(accepted.species[0].particles.position),
            "successful": result.successful,
            "gauss": diagnostics.electric_constraint / scale,
            "continuity": diagnostics.continuity_defect / diagnostics.continuity_scale,
            "magnetic": diagnostics.magnetic_constraint,
            "ledger_defect": diagnostics.energy.defect,
            "ledger_total": diagnostics.energy.total,
            "energy": synchronized.total,
            "field_energy": synchronized.electric_field + synchronized.magnetic_field,
        }

    _, series = eqx.filter_jit(
        lambda value: jax.lax.scan(body, value, None, length=STEPS)
    )(state)
    result = {key: np.asarray(value) for key, value in series.items()}
    first = amplitude(state.species[0].particles.position)
    result["amplitude"] = np.concatenate((np.asarray(first)[None], result["amplitude"]))
    if not result["successful"].all():
        raise RuntimeError("A Langmuir PIC step was rejected.")
    return result


def prony_frequency(series: np.ndarray) -> float:
    """``ω`` from ``D(n+1) + D(n−1) = 2cos(ωΔt) D(n)`` on first differences."""
    d = np.diff(series)
    ratio = d[1:-1] @ (d[2:] + d[:-2]) / (d[1:-1] @ d[1:-1])
    return float(np.arccos(0.5 * ratio) / STEP)


def leapfrog_frequency(shape_factor: float) -> float:
    """Discrete leapfrog frequency of ``ω² = ω_p²(1 + m_e/m_i)·shape_factor``."""
    continuous = PLASMA_FREQUENCY * np.sqrt((1.0 + 1.0 / ION_MASS_RATIO) * shape_factor)
    return float(2.0 / STEP * np.arcsin(0.5 * continuous * STEP))


reference = leapfrog_frequency(1.0)
report: dict[str, Any] = {"reference_frequency": reference, "successful": True}
for cells in RESOLUTIONS:
    case = langmuir_case(cells)
    half = 0.5 * WAVENUMBER * case.spacing
    predicted = {"cochain": reference, "psatd": leapfrog_frequency(np.sin(half) / half)}
    solvers = {"cochain": cochain_field_solver(case), "psatd": psatd_field_solver(case)}
    for name, solver in solvers.items():
        result = run(case, solver)
        omega = prony_frequency(result["amplitude"])
        energy = result["energy"]
        report[f"{name}/{cells}"] = {
            "frequency": omega,
            "relative_error": (omega - reference) / reference,
            "predicted_relative_error": (predicted[name] - reference) / reference,
            "field_energy_frequency_ratio": prony_frequency(result["field_energy"])
            / omega,
            "max_gauss_relative": float(result["gauss"].max()),
            "max_continuity_relative": float(result["continuity"].max()),
            "max_magnetic_constraint": float(result["magnetic"].max()),
            "synchronized_energy_variation": float(np.ptp(energy) / energy.min()),
            "max_ledger_drift": float(
                np.abs(np.cumsum(result["ledger_defect"])).max()
                / result["ledger_total"].max()
            ),
        }
    report[f"difference/{cells}"] = (
        abs(
            report[f"cochain/{cells}"]["frequency"]
            - report[f"psatd/{cells}"]["frequency"]
        )
        / reference
    )
    if cells == RESOLUTIONS[0]:
        report["capabilities"] = {
            name: solver.pic_capabilities.admitted for name, solver in solvers.items()
        }
print(report)
