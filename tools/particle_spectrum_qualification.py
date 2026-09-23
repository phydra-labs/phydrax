#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict SLHA and analytic native scale-BVP candidate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax.interchange.hep import (
    parse_slha,
    serialize_slha,
    spectrum_observables_from_slha,
)
from phydrax.particle_physics import (
    particle_spectrum_candidate_profiles,
    ScaleBVPPlan,
    solve_scale_bvp,
)


def run_qualification() -> dict[str, object]:
    source = (
        b"# spectrum qualification\n"
        b"BLOCK MASS # pole masses\n"
        b"  25 1.251000000D+02 # h0\n"
        b"  1000022 2.000000000E+02 # neutralino\n"
        b"BLOCK XQUAL Q= 1.000000000E+03 # retained unknown\n"
        b"  1 2 3.5\n"
        b"DECAY 25 4.000000000E-03\n"
        b"  1.0 2 5 -5\n"
    )
    document = parse_slha(source)
    serialized = serialize_slha(document)
    restored = parse_slha(serialized)
    observables = spectrum_observables_from_slha(restored)

    coefficient = 0.2
    lower = 10.0
    upper = 1000.0
    target = 3.0
    bvp_plan = ScaleBVPPlan(
        ("g",),
        lambda log_scale, parameters: coefficient * parameters,
        lambda parameters: jnp.zeros((0,), dtype=parameters.dtype),
        lambda parameters: jnp.asarray((parameters[0] - target,)),
        low_residual_count=0,
        lower_scale=lower,
        upper_scale=upper,
        integration_steps=128,
        maximum_newton_steps=8,
        residual_tolerance=1e-11,
        source_ids=("analytic-beta-g-equals-0.2g",),
    )
    bvp = solve_scale_bvp(bvp_plan, jnp.asarray((1.0,)))
    expected_initial = target / np.exp(coefficient * np.log(upper / lower))
    roundtrip = (
        restored.diagnostics.unknown_block_names == ("XQUAL",)
        and b"retained unknown" in serialized
        and float(observables.value("pdg:25", kind="pole-mass")) == 125.1
    )
    successful = bool(roundtrip and bvp.converged and bvp.finite)
    return {
        "kind": "particle-spectrum-candidate-qualification",
        "profiles": [
            profile.to_record() for profile in particle_spectrum_candidate_profiles()
        ],
        "case": {
            "slha_source_id": document.source_id,
            "slha_profile_id": document.profile_id,
            "scale_bvp_plan_id": bvp_plan.plan_id,
        },
        "raw": {
            "serialized_slha": serialized.decode("utf-8"),
            "observable_labels": list(observables.labels),
            "observable_kinds": list(observables.kinds),
            "observable_values": np.asarray(observables.values).tolist(),
            "running_log_scales": np.asarray(bvp.log_scales).tolist(),
            "running_parameters": np.asarray(bvp.trajectory).tolist(),
            "residual_history": np.asarray(bvp.residual_history).tolist(),
            "accepted_steps": np.asarray(bvp.accepted_steps).tolist(),
        },
        "criteria": {
            "unknown_blocks_preserved": restored.diagnostics.unknown_block_names
            == ("XQUAL",),
            "semantic_roundtrip": roundtrip,
            "expected_initial_parameter": expected_initial,
            "observed_initial_parameter": float(bvp.final_parameters[0]),
            "final_parameter": float(bvp.trajectory[-1, 0]),
            "residual_norm": float(jnp.linalg.norm(bvp.residual)),
            "converged": bool(bvp.converged),
        },
        "successful": successful,
        "claim": "strict-interchange-and-analytic-scale-bvp-control-not-a-precision-bsm-spectrum",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_qualification()
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
