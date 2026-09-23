#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Distributed ownership, AMR transfer, and evidence-tagged derivatives."""

import jax.numpy as jnp

import phydrax as phx


def run():
    distributed = phx.discretization.MPMDistributedPlan(
        (8, 8),
        (4, 4),
        jnp.asarray([[0, 0], [1, 1]]),
        periodic_axes=(False, False),
        device_count=2,
        particle_capacity_per_device=4,
    )
    migration = phx.discretization.migrate_particles(
        distributed,
        jnp.asarray([[0.1, 0.1], [0.8, 0.8]]),
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        jnp.asarray((0, 0)),
        jnp.asarray((True, True)),
    )
    transaction = phx.discretization.distributed_global_transaction(
        jnp.asarray((True, True)), 3
    )
    amr = phx.discretization.MPMAMRPlan(((4, 4), (8, 8)), (4, 16))
    fine = jnp.arange(64.0).reshape((8, 8))
    coarse = amr.restrict(fine)
    amr_parity = jnp.max(jnp.abs(amr.restrict(amr.prolong(coarse)) - coarse))
    derivative = phx.discretization.branchwise_gradient(
        lambda value: jnp.sum(value**2),
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((0.2, 0.3)),
        branch_margin=0.5,
        journal_digest=1,
        evidence_id="example-branch",
    )
    nondifferentiable = phx.discretization.nondifferentiable_result(
        jnp.asarray((1.0, 2.0)),
        reason_code=99,
        journal_digest=2,
        evidence_id="example-topology-change",
    )
    if not bool(
        migration.successful & transaction.global_success & derivative.evidence.valid
    ):
        raise RuntimeError("Distributed MPM migration, commit, or derivative failed")
    return {
        "distributed": {
            "migration_successful": True,
            "global_commit": True,
            "generation": int(transaction.commit_generation),
        },
        "amr": {"dense_transfer_parity": float(amr_parity)},
        "derivative": {
            "branchwise_valid": True,
            "directional": float(derivative.derivative),
            "topology_kind": int(nondifferentiable.evidence.kind),
            "topology_derivative_available": bool(nondifferentiable.evidence.valid),
        },
    }


if __name__ == "__main__":
    print(run())
