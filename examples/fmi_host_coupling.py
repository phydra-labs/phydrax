#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A real FMI co-simulation zone coupled to a native Phydrax node on the host route.

The FMU is Phydrax's original thermal-zone specimen
(`tests/interchange/data/thermal_zone.{c,xml}`), compiled here with the local C
compiler and loaded through FMPy: one lumped capacity `C1 dT1/dt = G (T_b - T1)`
with its boundary temperature held over each communication step and integrated
exactly. The native participant is a node of capacity `C2` advanced by SSPRK(3,3)
substeps. The node temperature is held as the FMU boundary temperature; the FMU's
cumulative conducted heat (J) is the node's whole-window heat loss, exchanged as
an extensive, ledger-certified amount.

Both routes run through `prepare_host_coupling`: an explicit zone-first sweep and
an implicit damped fixed-point window whose every iterate restores the FMU's
native state. The reference is the analytic two-capacity solution; the total heat
`C1 T1 + C2 T2` is conserved. Without FMPy or a C compiler the example reports that
the FMI evidence is inconclusive and exits without running.
"""

import hashlib
import importlib
import importlib.util
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.interchange.fmi import (
    FMICoSimulationSession,
    FMICouplingBinding,
    FMICouplingParticipant,
    FMIVariableBinding,
)


cpl = phx.solver.coupling
SPECIMEN = Path(__file__).resolve().parents[1] / "tests" / "interchange" / "data"
ZONE_CAPACITY, CONDUCTANCE, ZONE_START = 2.0, 0.5, 350.0
NODE_CAPACITY, NODE_START = 1.0, 300.0
FINAL_TIME = 1.0
WINDOW_SIZES = (0.1, 0.05, 0.025)

SCALAR = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="lumped-scalar")
HEAT_FIELD = phx.discretization.DiscreteFieldSpace(
    "zone-node-heat",
    "zone-node-interface",
    phx.discretization.EntityDofLayout("zone-node-interface/entities", 1, 1),
    SCALAR,
    representation="cell_integral",
)
HEAT_MEASUREMENT = cpl.CouplingMeasurement.extensive(
    SCALAR, HEAT_FIELD.support_id, provenance_id="zone-node-conduction"
)
HEAT = cpl.CouplingQuantity("heat", phx.units.JOULE, sign_convention="node-to-zone")
TEMPERATURE = cpl.CouplingQuantity("temperature", phx.units.KELVIN)


def toolchain_absence() -> str | None:
    if importlib.util.find_spec("fmpy") is None:
        return "FMPy is not installed (install the 'fmi' extra)"
    if sys.platform not in ("darwin", "linux"):
        return "the specimen build supports POSIX macOS/Linux"
    if shutil.which("cc") is None:
        return "no C compiler 'cc' is available to build the FMU specimen"
    return None


def build_fmu(root: Path, /) -> tuple[Path, str]:
    """Compile the specimen and package it with its model description."""
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    library = root / ("thermal_zone" + extension)
    subprocess.run(
        [
            "cc",
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-fPIC",
            "-O2",
            str(SPECIMEN / "thermal_zone.c"),
            "-o",
            str(library),
            "-lm",
        ],
        check=True,
        timeout=60,
        capture_output=True,
    )
    platform = importlib.import_module("fmpy").platform
    archive = root / "thermal_zone.fmu"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.write(SPECIMEN / "thermal_zone.xml", "modelDescription.xml")
        output.write(library, f"binaries/{platform}/{library.name}")
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def heat_port(port_id: str, direction: Any, /) -> Any:
    return cpl.CouplingPort(
        port_id,
        direction,
        SCALAR,
        field_space=HEAT_FIELD,
        quantity=HEAT,
        measurement=HEAT_MEASUREMENT,
        temporal_kind="interval_integral",
        reference_scale=1.0,
    )


def temperature_port(port_id: str, direction: Any, /) -> Any:
    return cpl.CouplingPort(
        port_id, direction, SCALAR, quantity=TEMPERATURE, reference_scale=1.0
    )


def zone_participant(session: FMICoSimulationSession, /) -> FMICouplingParticipant:
    binding = FMICouplingBinding(
        session.model,
        (
            FMIVariableBinding(
                "boundary_temperature",
                temperature_port("zone/boundary-temperature", "input"),
                "hold",
            ),
            FMIVariableBinding(
                "conducted_heat", heat_port("zone/conducted-heat", "output"), "increment"
            ),
        ),
        time_unit=phx.units.SECOND,
    )
    return FMICouplingParticipant(session, binding, subsystem_id="zone")


def node_participant() -> Any:
    def rate(time: Any, state: Any, args: Any) -> Any:
        del time
        return jnp.full_like(state, args)

    def bind(window: Any, views: Any, model_state: Any, key: Any, args: Any) -> Any:
        del key, args
        # views[0] is this substep's uniform share of the window's heat loss.
        return cpl.MethodWindowBinding(
            -views[0][0] / (window.size * NODE_CAPACITY), model_state
        )

    return cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(rate),
        bind,
        lambda native, args: (native,),
        subsystem_id="node",
        substeps=2,
        input_ports=(heat_port("node/heat", "input"),),
        output_ports=(temperature_port("node/temperature", "output"),),
    )


EXCHANGES = (
    cpl.CouplingExchange(
        "boundary-temperature", "node/temperature", "zone/boundary-temperature"
    ),
    cpl.CouplingExchange(
        "conducted-heat",
        "zone/conducted-heat",
        "node/heat",
        temporal=cpl.CouplingTemporalConversion("window-integral"),
    ),
)
ZONE_FIRST = cpl.CouplingSweep("gauss-seidel", subsystem_order=("zone", "node"))
POLICIES = {
    "explicit zone-first sweep": cpl.ExplicitCouplingPolicy(ZONE_FIRST),
    "implicit fixed point, FMU restored per iterate": cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-11,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=60,
        ),
        (
            cpl.CouplingTolerance("zone/boundary-temperature", absolute=1e-9),
            cpl.CouplingTolerance("node/heat", absolute=1e-9),
        ),
        fixed_point_sweep=ZONE_FIRST,
    ),
}


def analytic(time: float, /) -> tuple[float, float]:
    total = ZONE_CAPACITY + NODE_CAPACITY
    mean = (ZONE_CAPACITY * ZONE_START + NODE_CAPACITY * NODE_START) / total
    rate = CONDUCTANCE * (1.0 / ZONE_CAPACITY + 1.0 / NODE_CAPACITY)
    difference = (ZONE_START - NODE_START) * math.exp(-rate * time)
    return (
        mean + NODE_CAPACITY / total * difference,
        mean - ZONE_CAPACITY / total * difference,
    )


def run(archive: tuple[Path, str], policy: Any, window_size: float, /) -> dict[str, Any]:
    path, digest = archive
    with FMICoSimulationSession(
        path.name,
        sha256=digest,
        trusted_root=path.parent,
        license_id="LicenseRef-PHYDRA-Proprietary",
        start_values={
            "capacity": ZONE_CAPACITY,
            "conductance": CONDUCTANCE,
            "initial_temperature": ZONE_START,
        },
    ) as session:
        node = node_participant()
        prepared = cpl.prepare_host_coupling(
            cpl.PartitionedCouplingDeclaration(
                (zone_participant(session), node),
                EXCHANGES,
                policy,
                time_unit=phx.units.SECOND,
            ),
            {"node": node.initial_state(jnp.asarray([NODE_START]))},
            t0=0.0,
            window_size=window_size,
        )
        solution = cpl.solve_host_coupling(prepared, t1=FINAL_TIME)
        zone = session.get_values(("temperature",))["temperature"]
    if not solution.successful:
        raise RuntimeError(
            f"A host window was not accepted at window size {window_size}."
        )
    state = solution.final_state
    node_value = float(
        np.asarray(state.participant_states[state.subsystem_ids.index("node")].native)[0]
    )
    ledger = np.asarray(state.cumulative_exchange_budget)
    return {
        "zone": zone,
        "node": node_value,
        "evaluations": sum(result.host_evaluations for result in solution.windows),
        "restores": sum(result.restores for result in solution.windows),
        "ledger": float(np.max(np.abs(ledger.sum(axis=1)))),
    }


def main() -> None:
    absent = toolchain_absence()
    if absent is not None:
        print(f"FMI host coupling skipped: {absent}. FMI evidence is inconclusive.")
        return
    reference = analytic(FINAL_TIME)
    initial = ZONE_CAPACITY * ZONE_START + NODE_CAPACITY * NODE_START
    with tempfile.TemporaryDirectory(prefix="phydrax-fmi-") as directory:
        archive = build_fmu(Path(directory).resolve())
        print(
            f"analytic reference at t = {FINAL_TIME}: T1 = {reference[0]:.9f} K, T2 = {reference[1]:.9f} K"
        )
        for label, policy in POLICIES.items():
            print(label)
            print(
                f"  {'window':>7s} {'max error':>11s} {'rate':>6s} {'heat drift':>11s} {'ledger':>9s} {'FMU steps':>9s} {'restores':>8s}"
            )
            errors: list[float] = []
            for window_size in WINDOW_SIZES:
                result = run(archive, policy, window_size)
                errors.append(
                    max(
                        abs(result["zone"] - reference[0]),
                        abs(result["node"] - reference[1]),
                    )
                )
                drift = (
                    abs(
                        ZONE_CAPACITY * result["zone"]
                        + NODE_CAPACITY * result["node"]
                        - initial
                    )
                    / initial
                )
                rate = (
                    ""
                    if len(errors) == 1
                    else f"{math.log2(errors[-2] / errors[-1]):.2f}"
                )
                print(
                    f"  {window_size:7.4f} {errors[-1]:11.3e} {rate:>6s} {drift:11.1e} "
                    f"{result['ledger']:9.1e} {result['evaluations']:9d} {result['restores']:8d}"
                )
            if not 0.9 < math.log2(errors[-2] / errors[-1]) < 1.1:
                raise RuntimeError(f"{label} is not first order in the window size.")


if __name__ == "__main__":
    main()
