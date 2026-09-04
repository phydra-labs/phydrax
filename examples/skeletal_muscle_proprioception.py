#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax

from phydrax.applications.skeletal_muscle.proprioception import (
    MILEUSNIC_SPINDLE_2006_DOI,
    MileusnicSpindle2006Plan,
    MileusnicSpindleInput,
)


def main() -> None:
    runtime = MileusnicSpindle2006Plan().prepare()
    rest = MileusnicSpindleInput(1.0, 0.0, 0.0, 0.0, 0.0)
    stretch = MileusnicSpindleInput(1.02, 0.1, 0.0, 70.0, 70.0)
    initial = runtime.initialize(rest)

    def step(state, _):
        candidate = runtime.candidate(state, stretch, 1.0e-4)
        return candidate.commit(), candidate.evidence.successful

    final, successful = jax.lax.scan(step, initial, xs=None, length=1_000)
    output = runtime.output(final, stretch)
    payload = {
        "source_doi": MILEUSNIC_SPINDLE_2006_DOI,
        "prepared_id": runtime.prepared_id,
        "duration_s": 0.1,
        "all_steps_successful": bool(successful.all()),
        "primary_afferent_pps": float(output.primary_afferent_pps),
        "secondary_afferent_pps": float(output.secondary_afferent_pps),
        "bag1_dynamic_activation": float(final.bag1_dynamic_activation),
        "bag2_static_activation": float(final.bag2_static_activation),
        "species_scope": "feline only",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
