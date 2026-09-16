#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


def _mesh():
    return phx.geometry.TriangleMesh(
        jnp.asarray(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        ),
        jnp.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))),
        source_id="synthetic-superconducting-square",
    )


def _manifest():
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-superconducting-material",
        checksum_algorithm="sha256",
        checksum="8" * 64,
        size_bytes=1,
        license_id="synthetic-qualification",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"identity": 1.0},
        uncertainty={"synthetic": 0.0},
        lineage_ids=("synthetic:superconductivity",),
    )


def build_quasiclassical_case():
    velocities = 1.0e6 * jnp.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))
    fermi = phx.applications.superconductivity.FermiSurfacePlan(
        velocities,
        jnp.full((4,), 0.25),
        jnp.ones((1, 4), dtype=jnp.complex128),
        ("s-wave",),
    )
    trajectory = phx.applications.superconductivity.RiccatiTrajectoryPlan(
        fermi, jnp.full((4, 3), 1.0e-9)
    )
    matsubara = phx.applications.superconductivity.MatsubaraQuadraturePlan(1.0e-23, 8)
    gap = 2.0e-22
    pair = (
        2.0
        * jnp.pi
        * matsubara.temperature_energy
        * jnp.sum(gap / jnp.sqrt(matsubara.frequencies**2 + gap**2))
    )
    equilibrium = phx.applications.superconductivity.QuasiclassicalSuperconductivityPlan(
        trajectory,
        matsubara,
        jnp.asarray(((gap / pair,),)),
        damping=1.0,
        iterations=2,
    )
    return equilibrium, gap


def build_cable_case():
    material = phx.equations.SuperconductingMaterialLawPlan(
        jnp.asarray((2.0, 10.0, 19.0)),
        jnp.asarray((0.0, 5.0)),
        jnp.asarray((0.0, 0.5 * jnp.pi)),
        1.0e8 * jnp.ones((3, 2, 2)),
        _manifest(),
        critical_temperature=20.0,
        stabilizer_resistivity=1.0e-4,
    )
    return phx.applications.superconductivity.SuperconductingCablePlan(
        material,
        jnp.ones((3,)),
        jnp.ones((3,)),
        jnp.zeros((3,)),
        jnp.full((3,), 1000.0),
        jnp.full((3,), 1000.0),
        superconductor_area=1.0e-6,
        stabilizer_area=1.0e-6,
        inductance=1.0,
        axial_thermal_conductance=1.0,
        coolant_heat_transfer_per_length=0.0,
        coolant_mass_flow_heat_capacity=0.0,
        coolant_inlet_temperature=4.0,
        dump_resistance=1.0,
        protection_trigger_temperature=4.0001,
        tolerance=1.0e-7,
    )


def qualify() -> dict[str, object]:
    mesh = _mesh()
    london_plan = phx.applications.superconductivity.ThinFilmLondonPlan(
        mesh, pearl_length=0.2
    )
    london = london_plan.solve(0.0)
    gl_plan = phx.applications.superconductivity.GaugeCovariantGLPlan(
        mesh,
        alpha=-1.0,
        beta=1.0,
        kinetic_coefficient=0.5,
        magnetic_coefficient=1.0,
        gauge_coupling=1.0,
    )
    gl_state = gl_plan.initialize(jnp.ones((5,), dtype=jnp.complex128))
    gl = gl_plan.solve(gl_state, iterations=8)
    tdgl = gl_plan.tdgl_step(
        gl_plan.initialize(0.5 * jnp.ones((5,), dtype=jnp.complex128)), 0.01
    )
    equilibrium_plan, gap = build_quasiclassical_case()
    quasiclassical = equilibrium_plan.solve(jnp.asarray((gap + 0.0j,)))
    spectroscopy = phx.applications.superconductivity.RetardedSpectroscopyPlan(
        equilibrium_plan,
        jnp.asarray((-3.0 * gap, 0.0, 3.0 * gap)),
        broadening=0.01 * gap,
    ).evaluate(quasiclassical)
    cable_plan = build_cable_case()
    cable_state = cable_plan.initialize(150.0, jnp.full((3,), 4.0), jnp.full((3,), 4.0))
    cable = cable_plan.advance(cable_state, 1.0e-6, 0.0)
    bridge = phx.applications.superconductivity.GLLondonBridgePlan(
        jnp.zeros((mesh.topology.edges.shape[0], london.face_sheet_current.size)),
        relative_tolerance=1.0e-12,
        minimum_scale_separation=5.0,
    ).evaluate(gl_plan, gl_state, london, scale_separation=10.0)
    checks = {
        "london": bool(london.evidence.successful),
        "gl": bool(gl.evidence.successful),
        "tdgl": bool(tdgl.successful),
        "quasiclassical": bool(quasiclassical.evidence.successful),
        "spectroscopy": bool(spectroscopy.successful),
        "cable": bool(cable.successful),
        "bridge": bool(bridge.successful),
    }
    return {
        "qualification": "superconductivity-native-closure",
        "evidence_scope": "synthetic-numerical-validity-only",
        "checks": checks,
        "metrics": {
            "london_linear_residual": float(london.evidence.linear_residual),
            "gl_gradient_norm": float(gl.evidence.gradient_norm),
            "quasiclassical_gap_residual": float(quasiclassical.evidence.gap_residual),
            "cable_energy_residual": float(
                jnp.abs(cable.evidence.electrical_energy_residual)
                + jnp.abs(cable.evidence.thermal_energy_residual)
            ),
            "bridge_relative_residual": float(bridge.relative_residual),
        },
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
