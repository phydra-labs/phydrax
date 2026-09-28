"""Biomembrane remesh compile, warm-runtime, and memory scaling by vertex capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)

from phydrax.applications.cellular_mechanics import BiomembranePlan
from phydrax.geometry.multiregion_surface import (
    EdgeSplitProposal,
    MultiRegionRemeshPlan,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceState,
    remesh_edge_flags,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0)),
        dtype=np.float64,
    ) / np.sqrt(3.0)
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    return vertices, faces


def _prepared(capacity: int, /) -> tuple[Any, MultiRegionSurfaceState]:
    vertices, faces = _tetrahedron()
    remesh_capacity = MultiRegionSurfaceCapacityPlan(
        vertex_capacity=capacity,
        edge_capacity=3 * capacity - 6,
        face_capacity=2 * capacity - 4,
        region_capacity=2,
        region_pair_capacity=1,
        maximum_edge_valence=2,
        maximum_vertex_region_pairs=1,
        event_capacity=1,
        resource_id=f"biomembrane-benchmark-{capacity}",
    )
    membrane = BiomembranePlan(
        faces,
        bending_rigidity=0.0,
        species_diffusivity=(0.1,),
        species_ids=("lipid",),
        remesh_capacity=remesh_capacity,
    ).prepare(vertices)
    padded = np.zeros((capacity, 3), dtype=np.float64)
    padded[: vertices.shape[0]] = vertices
    state = MultiRegionSurfaceState(membrane.remesh_topology, padded)
    return membrane, state


def _measure(capacity: int, warmup: int, repeats: int, /) -> dict[str, Any]:
    membrane, surface_state = _prepared(capacity)
    plan = MultiRegionRemeshPlan(
        minimum_edge_length=0.1,
        maximum_edge_length=1.0,
        operations=("split",),
    )

    def evaluate(positions: jax.Array) -> tuple[jax.Array, jax.Array]:
        flags = remesh_edge_flags(membrane.prepared_remesh, positions, plan)
        return flags.edge_lengths, flags.split

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(surface_state.positions),
        lambda lowered: lowered.compile(),
    )
    _, execution = measure_repeated(
        lambda: compiled(surface_state.positions), warmup=warmup, repeats=repeats
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )
    proposal, host_seconds = measure_host(
        lambda: membrane.propose_remesh(
            membrane.state(species_mass=np.ones((4, 1), dtype=np.float64)),
            EdgeSplitProposal((0, 1)),
        )
    )
    sheet = proposal.surface_result.sheet_transfer
    face = proposal.surface_result.face_transfer
    return {
        "vertex_capacity": capacity,
        "edge_capacity": membrane.remesh_topology.edge_capacity,
        "face_capacity": membrane.remesh_topology.face_capacity,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "warm_execution": execution.to_seconds_dict(),
        "host_transaction_seconds": host_seconds,
        "compiler": {
            "flops": compiler.flops,
            "argument_bytes": compiler.argument_bytes,
            "temporary_bytes": compiler.temporary_bytes,
            "output_bytes": compiler.output_bytes,
            "generated_code_bytes": compiler.generated_code_bytes,
        },
        "logical_prepared_bytes": logical_array_bytes(membrane.prepared_remesh),
        "logical_sheet_transfer_bytes": 0
        if sheet is None
        else logical_array_bytes(sheet),
        "logical_face_transfer_bytes": 0 if face is None else logical_array_bytes(face),
        "surface_status": proposal.surface_result.evidence.status.name,
        "committed_candidate": proposal.surface_result.committed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", nargs="+", type=int, default=(8, 32, 128, 512))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        any(capacity < 5 for capacity in arguments.capacities)
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError("capacities >= 5, warmup >= 0, and repeats >= 1 are required.")
    rows = [
        _measure(capacity, arguments.warmup, arguments.repeats)
        for capacity in arguments.capacities
    ]
    payload = {
        "benchmark": "biomembrane-remesh-scaling",
        "environment": capture_environment().to_dict(),
        "rows": rows,
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
