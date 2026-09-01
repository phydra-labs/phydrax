"""Qualification evidence for maximal native cosmology profiles."""

from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    cosmo = phx.applications.cosmology
    positions = jnp.asarray(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.8, 0.8, 0.8], [0.9, 0.8, 0.8]]
    )
    masses = jnp.ones((4,))
    displacement = positions[None, :, :] - positions[:, None, :]
    squared = jnp.sum(displacement**2, axis=-1) + 0.01**2
    direct = jnp.sum(
        jnp.where(
            (~jnp.eye(4, dtype=bool))[..., None],
            masses[None, :, None] * displacement / squared[..., None] ** 1.5,
            0.0,
        ),
        axis=1,
    )
    tree = cosmo.ParticleOctreePlan3D((1.0, 1.0, 1.0), 2).prepare(positions, masses)
    bh = cosmo.BarnesHutGravityPlan(1.0, softening=0.01, opening_angle=0.5).evaluate(tree)
    fmm = cosmo.UniformFMMPlan(
        1.0, cosmo.CartesianExpansionSpace(1), softening=0.01
    ).evaluate(tree)
    bh_error = jnp.max(jnp.sqrt(jnp.sum((bh.acceleration - direct) ** 2, axis=-1)))
    fmm_error = jnp.max(jnp.sqrt(jnp.sum((fmm.acceleration - direct) ** 2, axis=-1)))

    fof = cosmo.PeriodicFoFFinderPlan((1.0, 1.0, 1.0), 0.15, 4).find(
        jnp.arange(4),
        positions,
        jnp.zeros_like(positions),
        masses,
        jnp.ones((4,), dtype=bool),
    )
    manifold = cosmo.S3ManifoldPlan(2.0)
    point = jnp.asarray([[2.0, 0.0, 0.0, 0.0]])
    tangent = jnp.asarray([[0.0, 0.1, 0.0, 0.0]])
    target = manifold.exponential(point, tangent)
    recovered = manifold.logarithm(point, target)

    amr = cosmo.TwoLevelAMRPlan((2,), 1)
    coarse = jnp.asarray([[1.0], [2.0]])
    restricted = amr.restrict(amr.prolong(coarse))

    report = {
        "barnes_hut_max_absolute_error": float(bh_error),
        "barnes_hut_finite": bool(bh.successful),
        "fmm_max_absolute_error": float(fmm_error),
        "fmm_finite": bool(fmm.successful),
        "fof_groups": int(jnp.sum(fof.group_active)),
        "fof_successful": bool(fof.successful),
        "s3_exp_log_error": float(jnp.max(jnp.abs(recovered - tangent))),
        "amr_prolong_restrict_error": float(jnp.max(jnp.abs(restricted - coarse))),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
