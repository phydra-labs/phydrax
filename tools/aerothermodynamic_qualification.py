#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


def qualify() -> dict[str, object]:
    support = phx.applications.aerothermodynamics.AerothermodynamicSupportTuple(
        gas_system_id="ionized-multitemperature",
        transport_id="ambipolar",
        thermochemistry_id="fixed-work-source",
        radiation_id="multigroup-nonlte",
        surface_id="finite-rate",
        material_id="porous-response",
        kinetic_id="dsmc",
        hybrid_id="continuum-dsmc",
        turbulence_id="sst-ddes",
        discretization_id="conservative-fv-dg",
        topology_id="ale-amr",
        backend="jax",
        precision="float64",
        species_count=11,
        mode_count=2,
        radiation_group_count=8,
    )
    ledger = phx.applications.aerothermodynamics.AerothermodynamicConservationLedger.from_exchanges(
        mass=0.0,
        elements=jnp.zeros((5,)),
        charge=0.0,
        momentum=jnp.zeros((3,)),
        energy=0.0,
        surface_sites=0.0,
    )
    interface = phx.solver.FixedContinuumDSMCInterfacePlan(5).exchange(
        jnp.asarray((1.0, 0.1, 0.2, 0.3, 2.0)),
        jnp.asarray((1.01, 0.09, 0.21, 0.29, 2.02)),
        jnp.full((5,), 0.01),
        0.001,
        1.0,
    )
    sst = phx.equations.SSTTurbulencePlan("sst-2003-m").evaluate(
        1.0,
        1.8e-5,
        0.1,
        10.0,
        jnp.asarray(((0.0, 20.0), (0.0, 0.0))),
        jnp.asarray((0.01, 0.0)),
        jnp.asarray((0.1, 0.0)),
        0.01,
    )
    checks = {
        "support_identity": bool(support.support_id),
        "global_ledger": bool(ledger.successful),
        "hybrid_action_reaction": bool(interface.successful),
        "sst_variant": bool(sst.successful),
    }
    return {
        "qualification": "aerothermodynamic-production-closure",
        "support_id": support.support_id,
        "checks": checks,
        "successful": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not report["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
