#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network


def axial_chain(node_count: int):
    nodes = int(node_count)
    edges = jnp.stack((jnp.arange(nodes - 1), jnp.arange(1, nodes)), axis=-1).astype(
        jnp.int32
    )
    constraints = jnp.zeros((nodes, 2), dtype=bool).at[:, 1].set(True).at[0, 0].set(True)
    structure = sm.ForceDensityStructure.from_edges(
        edges, nodes, 2, constrained_dofs=constraints
    )
    positions = jnp.stack((jnp.linspace(0.0, 1.0, nodes), jnp.zeros((nodes,))), axis=-1)
    material = mn.LinearElasticMaterial(1000.0, 400.0, 1.0)
    properties = mn.MemberPropertyMap(
        (material,),
        (mn.AxialSection(1.0),),
        (0,) * (nodes - 1),
        (0,) * (nodes - 1),
    )
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure, rotation_constrained=jnp.ones((nodes, 1), dtype=bool)
    )
    definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
    assembly = mn.MemberNetworkAssembly((mn.AxialMemberBlock(jnp.arange(nodes - 1)),))
    problem = mn.MemberNetworkProblem(definition, assembly)
    initial = mn.MemberKinematics(positions, jnp.zeros((nodes, 1)))
    inputs = mn.MemberNetworkInputs(
        structure.prescribed_values(positions),
        dofs.prescribed_rotations(initial.rotation_vectors),
        jnp.zeros((nodes, 2)).at[-1, 0].set(1.0),
        jnp.zeros((nodes, 1)),
        reference.rest_lengths,
    )
    return problem, inputs, initial


def run_axial(node_count: int, repeats: int):
    problem, inputs, initial = axial_chain(node_count)
    started = time.perf_counter()
    plan = mn.plan_member_network(problem, inputs, initial)
    plan_seconds = time.perf_counter() - started
    prepared = mn.prepare_member_network(plan, inputs, initial)
    started = time.perf_counter()
    result = mn.solve_member_network(prepared)
    jax.block_until_ready(result.state.kinematics.positions)
    solve_seconds = time.perf_counter() - started
    current = prepared
    started = time.perf_counter()
    for index in range(repeats):
        changed = eqx.tree_at(
            lambda value: value.nodal_forces,
            inputs,
            inputs.nodal_forces.at[-1, 0].set(1.0 + 0.01 * index),
        )
        current = mn.refresh_member_network(current, changed, result.state.kinematics)
        refreshed = mn.solve_member_network(current)
        jax.block_until_ready(refreshed.state.kinematics.positions)
    refresh_seconds = (time.perf_counter() - started) / repeats
    return {
        "node_count": node_count,
        "member_count": node_count - 1,
        "plan_seconds": plan_seconds,
        "solve_seconds": solve_seconds,
        "refresh_seconds": refresh_seconds,
        "residual_norm": float(result.diagnostics.residual_norm),
        "status": int(result.status),
        "tip_displacement": float(result.state.kinematics.positions[-1, 0] - 1.0),
    }


def run_beam():
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        constrained_dofs=jnp.asarray(((True, True), (False, False))),
    )
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    material = mn.LinearElasticMaterial(1000.0, 400.0, 1.0)
    properties = mn.MemberPropertyMap(
        (material,), (mn.BeamSection(1.0, 1.0, 1.0, 0.5, 100.0, 100.0),), (0,), (0,)
    )
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure, rotation_constrained=jnp.asarray(((True,), (False,)))
    )
    definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
    assembly = mn.MemberNetworkAssembly((mn.CorotationalFrameBlock((0,)),))
    problem = mn.MemberNetworkProblem(definition, assembly)
    initial = mn.MemberKinematics(positions, jnp.zeros((2, 1)))
    inputs = mn.MemberNetworkInputs(
        structure.prescribed_values(positions),
        dofs.prescribed_rotations(initial.rotation_vectors),
        jnp.asarray(((0.0, 0.0), (0.0, -1.0))),
        jnp.zeros((2, 1)),
        reference.rest_lengths,
    )
    started = time.perf_counter()
    result = mn.member_network_equilibrium(problem, inputs, initial)
    stability = mn.tangent_stability(problem, inputs, result.state.kinematics)
    jax.block_until_ready(stability.eigenvalues)
    return {
        "status": int(result.status),
        "tip_y": float(result.state.kinematics.positions[1, 1]),
        "minimum_tangent_eigenvalue": float(stability.minimum_eigenvalue),
        "wall_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/member_network.json"),
    )
    args = parser.parse_args()
    sizes = (8,) if args.smoke else (25, 100, 500)
    repeats = 1 if args.smoke else 3
    payload = {
        "axial": [run_axial(size, repeats) for size in sizes],
        "beam": run_beam(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
