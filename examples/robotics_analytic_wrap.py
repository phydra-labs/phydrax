#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications.robotics import SphereRouteWrapPlan


def main() -> None:
    route = SphereRouteWrapPlan(48).prepare(jnp.zeros(3), 1.0)
    result = route.evaluate(
        jnp.asarray((-2.0, 0.35, 0.0)),
        jnp.asarray((2.1, 0.65, 0.0)),
    )
    payload = {
        "source_revision": result.evidence.source_revision,
        "prepared_id": result.prepared_id,
        "successful": bool(result.evidence.successful),
        "applied": bool(result.evidence.applied),
        "surface_length_m": float(result.surface_length_m),
        "total_length_m": float(result.total_length_m),
        "event_margin": float(result.evidence.event_margin),
        "tangency_residual": float(result.evidence.endpoint_tangency_residual),
        "surface_residual": float(result.evidence.surface_residual),
        "fixed_branch_gradient_supported": bool(
            result.evidence.fixed_branch_gradient_supported
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
