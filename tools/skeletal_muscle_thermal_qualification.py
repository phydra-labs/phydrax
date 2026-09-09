#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Manufactured scalar Pennes qualification, never human physiological fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.energetics import RetainedHeatLedger
from phydrax.applications.skeletal_muscle.thermal import (
    Pennes1948Boundary,
    Pennes1948EvidenceBundle,
    Pennes1948Parameters,
    Pennes1948Plan,
    RetainedHeatProjection,
)
from phydrax.discretization import (
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)


_ASSET = (
    Path(__file__).resolve().parents[1]
    / "tests/data/skeletal_muscle/thermal/manufactured_scalar_box.json"
)


def manufactured_case(
    resolution=1,
    *,
    perfusion=0.0,
    heterogeneous=False,
    boundary="insulated",
    initial=None,
    projection=None,
):
    """Explicit numerical fixture constants from the repository-owned raw asset."""
    raw = _ASSET.read_bytes()
    asset = json.loads(raw)
    asset_hash = hashlib.sha256(raw).hexdigest()
    if resolution < 1:
        raise ValueError("Resolution must be positive.")
    axis = np.linspace(0, 1, resolution + 1)
    coordinates = np.asarray([(x, y, z) for x in axis for y in axis for z in axis])

    def vertex(i, j, k):
        return (i * (resolution + 1) + j) * (resolution + 1) + k

    cells = []
    split = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                corners = [
                    vertex(i + x, j + y, k + z)
                    for x, y, z in (
                        (0, 0, 0),
                        (1, 0, 0),
                        (0, 1, 0),
                        (1, 1, 0),
                        (0, 0, 1),
                        (1, 0, 1),
                        (0, 1, 1),
                        (1, 1, 1),
                    )
                ]
                cells.extend([[corners[x] for x in tetra] for tetra in split])
    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(coordinates), jnp.asarray(cells, dtype=jnp.int32)
    )
    disc = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "temperature",
            lagrange_element("tetrahedron", 1),
        ),
    ).prepare()
    exterior = np.asarray(disc.exterior_facet_domain.entity_indices)
    faces = np.asarray(mesh.connectivity.faces)[exterior]
    face_x = coordinates[faces, 0]
    patches = []
    if boundary in ("linear-flux", "linear-convection", "sine-dirichlet"):
        left = np.all(face_x == 0, axis=1)
        right = np.all(face_x == 1, axis=1)
        if boundary == "sine-dirichlet":
            patches = [
                Pennes1948Boundary(
                    exterior.tolist(),
                    "dirichlet",
                    300.0,
                    heat_transfer_W_per_m2_K=0.0,
                    asset_id=asset_hash,
                )
            ]
        else:
            patches.append(
                Pennes1948Boundary(
                    exterior[left].tolist(),
                    "dirichlet",
                    300.0,
                    heat_transfer_W_per_m2_K=0.0,
                    asset_id=asset_hash,
                )
            )
            patches.append(
                Pennes1948Boundary(
                    exterior[right].tolist(),
                    "flux" if boundary == "linear-flux" else "convection",
                    -2.0 if boundary == "linear-flux" else 303.0,
                    heat_transfer_W_per_m2_K=0.0 if boundary == "linear-flux" else 1.0,
                    asset_id=asset_hash,
                )
            )
            patches.append(
                Pennes1948Boundary(
                    exterior[~(left | right)].tolist(),
                    "insulated",
                    0.0,
                    heat_transfer_W_per_m2_K=0.0,
                    asset_id=asset_hash,
                )
            )
    else:
        patches = [
            Pennes1948Boundary(
                exterior.tolist(),
                "insulated",
                0.0,
                heat_transfer_W_per_m2_K=0.0,
                asset_id=asset_hash,
            )
        ]
    roles = (
        "equations",
        "geometry",
        "regions",
        "properties",
        "perfusion",
        "initial",
        "boundary",
        "source-projection",
        "retention",
        "validation",
    )
    evidence = Pennes1948EvidenceBundle(
        tuple(
            (
                role,
                _ASSET.as_uri(),
                asset_hash,
                asset["reuse_rights"],
                "SI: m,K,s,W,J; volumetric perfusion 1/s",
                "manufactured-numerical-only",
            )
            for role in roles
        ),
        coordinate_frame=asset["coordinate_frame"],
        length_unit="m",
        perfusion_unit="1/s",
    )
    count = len(cells)
    region_indices = [i % 2 for i in range(count)] if heterogeneous else [0] * count
    regions = (
        ("manufactured-a", "manufactured-b") if heterogeneous else ("manufactured-a",)
    )
    nregions = len(regions)
    parameters = Pennes1948Parameters(
        [2.0, 4.0] if heterogeneous else [2.0],
        [8.0] * nregions,
        [perfusion] * nregions,
        [4.0] * nregions,
        [300.0] * nregions,
    )
    if projection is None:
        projection = RetainedHeatProjection(
            ("manufactured-source",),
            np.zeros(count, dtype=np.int32),
            np.arange(count),
            np.full(count, 1 / count),
            cell_count=count,
            source_model_id="manufactured-retained-source",
            retention_evidence_id=asset_hash,
            asset_id=asset_hash,
        )
    temperature = np.full(len(coordinates), 300.0)
    if boundary.startswith("linear-"):
        temperature += coordinates[:, 0]
    elif boundary == "sine-dirichlet":
        temperature += np.prod(np.sin(np.pi * coordinates), axis=1)
        temperature[np.any((coordinates == 0) | (coordinates == 1), axis=1)] = 300.0
    if initial is not None:
        temperature = np.asarray(initial)
    return Pennes1948Plan(
        mesh,
        parameters,
        regions,
        region_indices,
        projection,
        patches,
        temperature,
        evidence,
        linear_relative_tolerance=1e-11,
        linear_absolute_tolerance=1e-11,
        balance_relative_tolerance=1e-8,
        balance_absolute_tolerance_J=1e-8,
        maximum_iterations=512,
    ).prepare()


def manufactured_source(prepared, state, dt=0.1, power=2.0):
    return RetainedHeatLedger(
        jnp.asarray([[power], [0.0], [0.0], [0.0], [0.0]]),
        ("manufactured-source",),
        source_model_id="manufactured-retained-source",
        source_state_id="manufactured-accepted-source",
        evidence_id=prepared.plan.projection.retention_evidence_id,
        time_start_s=state.time_s,
        time_end_s=state.time_s + dt,
        chemical_storage_power_W=jnp.zeros((1,)),
        chemical_export_power_W=jnp.zeros((1,)),
    )


def run_qualification(smoke=False):
    reports = []
    for label, kwargs in (
        ("uniform-source", {}),
        ("heterogeneous-source", {"heterogeneous": True}),
        ("linear-flux", {"boundary": "linear-flux"}),
        ("linear-convection", {"boundary": "linear-convection"}),
        ("perfusion", {"perfusion": 0.5}),
    ):
        prepared = manufactured_case(**kwargs)
        state = prepared.initial_state()
        power = 0.0 if label.startswith("linear-") else 2.0
        source = manufactured_source(prepared, state, power=power)
        candidate = eqx.filter_jit(prepared.propose)(state, source)
        if label.startswith("linear-"):
            expected = state.temperature_K
        elif label == "perfusion":
            expected = 300.0 + 0.2 / 8.2
        else:
            expected = 300.025
        error = float(jnp.max(jnp.abs(candidate.proposed_state.temperature_K - expected)))
        reports.append(
            {
                "case": label,
                "maximum_temperature_error_K": error,
                "balance_residual_J": float(candidate.ledger.balance_residual_J),
                "successful": bool(candidate.evidence.successful) and error < 1e-8,
            }
        )
    if not smoke:
        errors = []
        for resolution in (2, 4, 8):
            prepared = manufactured_case(resolution, boundary="sine-dirichlet")
            state = prepared.initial_state()
            initial = state.temperature_K
            dt = 0.01 / resolution**2
            steps = resolution**2
            action = eqx.filter_jit(prepared.propose)
            accepted = True
            for _ in range(steps):
                source = manufactured_source(prepared, state, dt=dt, power=0.0)
                candidate = action(state, source)
                accepted &= bool(candidate.evidence.successful)
                state = candidate.commit(
                    state, prepared=prepared, source_state_id=source.source_state_id
                )
            exact = 300 + (initial - 300) * np.exp(-3 * np.pi**2 * 2 / 8 * 0.01)
            error = float(jnp.max(jnp.abs(state.temperature_K - exact)))
            errors.append(error)
            reports.append(
                {
                    "case": "mesh-time-refinement",
                    "resolution": resolution,
                    "maximum_temperature_error_K": error,
                    "successful": accepted,
                }
            )
        reports.append(
            {
                "case": "mesh-time-order",
                "errors_K": errors,
                "successful": errors[-1] < errors[0],
            }
        )
    return {
        "claim_scope": "manufactured scalar numerical field only; no human validation",
        "cases": reports,
        "all_successful": all(r["successful"] for r in reports),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    result = run_qualification(args.smoke)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
