#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


def _slip(direction, normal):
    return phx.applications.crystal_plasticity.CrystalSlipSystem(
        jnp.asarray(direction), jnp.asarray(normal)
    )


def _model(*systems):
    cp = phx.applications.crystal_plasticity
    return cp.CrystalPlasticityModel(
        systems,
        cp.CrystalPlasticityParameters(8.0, 20.0, 0.1, 1.0, 1.5, 1.0),
    )


def _point_law():
    model = _model(_slip((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    state = model.initial_state()
    deformation = jnp.eye(3).at[0, 1].set(0.35)
    orientation = jnp.eye(3)
    direction = jnp.asarray(((0.01, -0.02, 0.0), (0.0, 0.01, -0.01), (0.0, 0.0, 0.01)))
    update = model.update(deformation, state, orientation, 0.1)
    _, tangent_action = jax.jvp(
        lambda value: model.update(value, state, orientation, 0.1).first_piola,
        (deformation,),
        (direction,),
    )
    energy_stress = jax.grad(lambda value: model.free_energy(value, state))(deformation)
    stress_defect = jnp.max(
        jnp.abs(energy_stress - model.first_piola(deformation, state))
    )
    passed = (
        update.accepted
        & update.thermodynamic_admissible
        & (update.slip_increment[0] > 0.0)
        & (jnp.abs(update.plastic_determinant - 1.0) < 2.0e-5)
        & (update.elastic_determinant > 0.0)
        & (update.incremental_dissipation >= -1.0e-6)
        & jnp.all(jnp.isfinite(tangent_action))
        & (stress_defect < 2.0e-4)
    )
    return {
        "passed": bool(passed),
        "slip_increment": float(update.slip_increment[0]),
        "plastic_determinant": float(update.plastic_determinant),
        "elastic_determinant": float(update.elastic_determinant),
        "incremental_dissipation": float(update.incremental_dissipation),
        "energy_stress_defect": float(stress_defect),
        "jvp_norm": float(jnp.sqrt(jnp.sum(tangent_action**2))),
    }


def _frame_covariance():
    model = _model(_slip((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    state = model.initial_state()
    deformation = jnp.eye(3).at[0, 1].set(0.35)
    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    base = model.update(deformation, state, jnp.eye(3), 0.1)
    transformed = model.update(rotation @ deformation @ rotation.T, state, rotation, 0.1)
    stress_defect = jnp.max(
        jnp.abs(transformed.first_piola - rotation @ base.first_piola @ rotation.T)
    )
    plastic_defect = jnp.max(
        jnp.abs(
            transformed.state.plastic_deformation
            - rotation @ base.state.plastic_deformation @ rotation.T
        )
    )
    passed = transformed.accepted & (stress_defect < 3.0e-4) & (plastic_defect < 3.0e-4)
    return {
        "passed": bool(passed),
        "stress_covariance_defect": float(stress_defect),
        "plastic_covariance_defect": float(plastic_defect),
    }


def _discretization():
    coordinates = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 0.0, 1.0),
        )
    )
    blocks = (
        phx.discretization.CellBlock(
            "phase-a",
            "tetrahedron",
            jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
            global_ids=jnp.asarray((10,)),
        ),
        phx.discretization.CellBlock(
            "phase-b",
            "tetrahedron",
            jnp.asarray(((4, 5, 6, 7),), dtype=jnp.int32),
            global_ids=jnp.asarray((20,)),
        ),
    )
    return phx.discretization.FiniteElementPlan(
        phx.discretization.CellMesh(coordinates, blocks),
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("tetrahedron", 1),
            component_shape=(3,),
        ),
    ).prepare()


def _routed_state():
    cp = phx.applications.crystal_plasticity
    discretization = _discretization()
    first = _model(_slip((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    second = _model(
        _slip((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        _slip((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    route = cp.CrystalPlasticityRoute(
        discretization,
        "u",
        (
            ("phase-a", first, jnp.eye(3)),
            ("phase-b", second, rotation),
        ),
    )
    transaction = route.initialize()
    checkpoint = route.checkpoint(transaction)
    restored = route.restore(checkpoint)
    form = cp.cpfem_equilibrium_form(discretization, "u", route, transaction, 0.1)
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    residual, auxiliary = compiled.residual_with_auxiliary(jnp.zeros((8, 3)))
    residual_norm = jnp.sqrt(jnp.sum(residual**2))
    widths = tuple(shape[-1] for shape in route.state_shapes)
    passed = (
        auxiliary.valid
        & (residual_norm < 2.0e-6)
        & (widths == (11, 12))
        & (restored.transaction_id == transaction.transaction_id)
    )
    return {
        "passed": bool(passed),
        "state_widths": list(widths),
        "residual_norm": float(residual_norm),
        "transaction_layout": transaction.layout_id,
        "checkpoint_identity": checkpoint.payload_id,
        "route_identity": route.route_id,
    }


def qualify():
    sections = {
        "point_law": _point_law(),
        "frame_covariance": _frame_covariance(),
        "routed_state": _routed_state(),
    }
    return {
        "maturity": "experimental",
        "passed": all(section["passed"] for section in sections.values()),
        "scope": (
            "finite-strain point law and static phase-homogeneous cell-block routing"
        ),
        "sections": sections,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/crystal_plasticity_qualification.json"),
    )
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(arguments.output)
    print(payload, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
