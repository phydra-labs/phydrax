#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


def qualification_record() -> dict[str, object]:
    weights = phx.particle_physics.EventWeightSet(
        jnp.asarray([[1.0, 1.1], [-0.25, -0.3], [0.5, 0.45]]),
        names=("nominal", "scale-up"),
        variation_kinds=(
            phx.particle_physics.WeightVariationKind.NOMINAL,
            phx.particle_physics.WeightVariationKind.SHAPE,
        ),
        correlation_groups=("nominal", "scale"),
    )
    ledger = phx.particle_physics.summarize_event_weights(
        weights,
        attempted_count=3,
        cross_section=1.25,
        cross_section_uncertainty=0.1,
    )
    histogram = phx.applications.collider_analysis.fill_weighted_histogram(
        phx.applications.collider_analysis.HistogramPlan(
            jnp.asarray([0.0, 1.0, 2.0]),
            observable_id="qualification-observable",
            unit_id="1",
        ),
        jnp.asarray([0.25, 0.75, 1.25]),
        weights.nominal,
    )
    convention = phx.applications.accelerator.AcceleratorConvention()
    bunch = phx.applications.accelerator.AcceleratorBunch(
        jnp.asarray(
            [[1.0e-3, 0.0, 1.0e-3, 0.0, 0.0, 0.0], [-1.0e-3, 0.0, -1.0e-3, 0.0, 0.0, 0.0]]
        ),
        jnp.ones(2),
        jnp.asarray([0, 1]),
        reference_rest_energy=1.0,
        reference_momentum=2.0,
        reference_charge=1.0,
        convention=convention,
        bunch_id="qualification",
    )
    tracked = phx.applications.accelerator.track_beamline(
        phx.applications.accelerator.BeamlinePlan(
            jnp.asarray([0]),
            jnp.asarray([1.0]),
            jnp.asarray([0.0]),
            jnp.asarray([0.0]),
            element_ids=("drift",),
            convention=convention,
        ),
        bunch,
    )
    passed = bool(ledger.successful) and bool(histogram.finite) and bool(tracked.accepted)
    return {
        "passed": passed,
        "event_accounting": {
            "generated": int(ledger.generated_count),
            "positive": int(ledger.positive_count),
            "negative": int(ledger.negative_count),
            "sum_weights": float(ledger.sum_weights),
            "sum_squared_weights": float(ledger.sum_squared_weights),
        },
        "histogram": histogram.sum_weights.tolist(),
        "accelerator_transmission": float(
            phx.applications.accelerator.beam_diagnostics(tracked.bunch).transmission
        ),
        "nonclaims": [
            "no general native NLO or shower claim",
            "no microscopic detector-transport claim",
            "no generic finite-density sign-problem solution",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = qualification_record()
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
