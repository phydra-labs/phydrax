#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Manufactured geometry: source-pure Uchida heat → conservative scalar Pennes."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.energetics import (
    uchida_umberger_retained_heat,
    UchidaUmberger2010Parameters,
    UchidaUmberger2010Plan,
)
from phydrax.applications.skeletal_muscle.thermal import RetainedHeatProjection
from tools.skeletal_muscle_thermal_qualification import manufactured_case


def main():
    jax.config.update("jax_enable_x64", True)
    fixture = manufactured_case()
    model = UchidaUmberger2010Plan(
        UchidaUmberger2010Parameters(
            [0.5],
            [0.5],
            [0.1],
            [10.0],
        ),
        ("manufactured-muscle",),
    )
    evidence_id = fixture.plan.projection.retention_evidence_id
    result = model.evaluate(
        jnp.asarray([0.8]),
        jnp.asarray([0.7]),
        jnp.asarray([100.0]),
        jnp.ones(1),
        jnp.asarray([0.1]),
        jnp.asarray([-0.01]),
    )
    heat = uchida_umberger_retained_heat(
        model,
        result,
        source_state_id="accepted-example-trajectory",
        evidence_id=evidence_id,
        time_start_s=0.0,
        time_end_s=0.1,
    )
    count = fixture.geometry.cell_volume_m3.size
    projection = RetainedHeatProjection(
        model.muscle_ids,
        np.zeros(count, dtype=np.int32),
        np.arange(count),
        np.full(count, 1 / count),
        cell_count=count,
        source_model_id=model.model_id,
        retention_evidence_id=evidence_id,
        asset_id=evidence_id,
    )
    prepared = manufactured_case(projection=projection)
    initial = prepared.initial_state()
    candidate = prepared.propose(initial, heat)
    committed = candidate.commit(
        initial, prepared=prepared, source_state_id=heat.source_state_id
    )
    payload = {
        "claim_scope": "Numerical example only: arbitrary manufactured material/geometry; no human thermal prediction",
        "prepared_id": prepared.prepared_id,
        "metabolic_power_W_not_used_as_heat": float(
            result.total_muscle_metabolic_power_W
        ),
        "retained_power_W": float(jnp.sum(heat.retained_power_W)),
        "integrated_field_source_W": float(
            jnp.sum(
                candidate.volumetric_retained_heat_W_per_m3
                * prepared.geometry.cell_volume_m3
            )
        ),
        "mean_temperature_K": float(jnp.mean(committed.temperature_K)),
        "balance_residual_J": float(candidate.ledger.balance_residual_J),
        "successful": bool(candidate.evidence.successful),
        "accepted_steps": int(committed.accepted_steps),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
