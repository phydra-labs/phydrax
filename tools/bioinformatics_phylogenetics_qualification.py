#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact fixed-tree phylogenetic likelihood qualification."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics import phylogenetics
from tools.bioinformatics_common_qualification import (
    emit_report,
    fingerprint,
    method_contract_evidence,
    qualification_report,
)


def _two_tip_pruning_case() -> dict[str, object]:
    parent = jnp.asarray((2, 2, -1), dtype=jnp.int32)
    topology = phylogenetics.tree_topology(parent)
    model = phylogenetics.jc69()
    tip_partials = jnp.asarray((((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)),))
    branch_lengths = jnp.asarray((0.17, 0.23, 0.0))
    partition = phylogenetics.LikelihoodPartition(
        jnp.asarray((True,)), model, partition_name="all-sites"
    )

    result = phylogenetics.felsenstein_pruning(
        topology,
        tip_partials,
        branch_lengths,
        (partition,),
    )
    transition_left = model.transition_matrix(branch_lengths[0])
    transition_right = model.transition_matrix(branch_lengths[1])
    likelihood = jnp.sum(
        model.root_distribution * transition_left[:, 0] * transition_right[:, 2]
    )
    oracle_log_likelihood = float(np.log(np.asarray(likelihood)))
    observed_log_likelihood = float(np.asarray(result.log_likelihood))
    likelihood_error = abs(observed_log_likelihood - oracle_log_likelihood)

    derivative_transition_left = model.rate_matrix @ transition_left
    derivative_likelihood = jnp.sum(
        model.root_distribution
        * derivative_transition_left[:, 0]
        * transition_right[:, 2]
    )
    analytic_gradient = float(np.asarray(derivative_likelihood / likelihood))

    def branch_objective(left_length):
        lengths = branch_lengths.at[0].set(left_length)
        return phylogenetics.felsenstein_pruning(
            topology,
            tip_partials,
            lengths,
            (partition,),
        ).log_likelihood

    automatic_gradient = float(np.asarray(jax.grad(branch_objective)(branch_lengths[0])))
    gradient_error = abs(automatic_gradient - analytic_gradient)

    insufficient = phylogenetics.tree_topology(parent, child_capacity=1)
    capacity_rejected = (
        not bool(np.asarray(insufficient.valid))
        and not bool(np.asarray(insufficient.evidence.child_capacity_sufficient))
        and int(np.asarray(insufficient.status)) != 0
    )
    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "parent_indices": parent,
        "tip_partials": tip_partials,
        "branch_lengths": branch_lengths,
        "rate_matrix": model.rate_matrix,
        "root_distribution": model.root_distribution,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "explicit root-state summation for a two-tip JC69 tree",
        "gradient_oracle": "d exp(Qt)/dt = Q exp(Qt)",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "observed_log_likelihood": observed_log_likelihood,
        "oracle_log_likelihood": oracle_log_likelihood,
        "absolute_log_likelihood_error": likelihood_error,
        "automatic_branch_gradient": automatic_gradient,
        "analytic_branch_gradient": analytic_gradient,
        "absolute_gradient_error": gradient_error,
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "transition_matrices_stochastic": bool(
            np.asarray(result.evidence.transition_matrices_stochastic)
        ),
        "topology_capacity_check": {
            "configured_child_capacity": 1,
            "required_child_capacity": 2,
            "status": int(np.asarray(insufficient.status)),
            "rejected": capacity_rejected,
        },
        "passed": bool(
            np.asarray(result.valid)
            and likelihood_error <= 2.0e-5
            and gradient_error <= 3.0e-5
            and capacity_rejected
        ),
    }


def qualification() -> dict[str, object]:
    return qualification_report(
        "phylogenetics",
        {"two_tip_felsenstein_pruning": _two_tip_pruning_case()},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify public fixed-tree phylogenetic APIs."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    return emit_report(qualification(), arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
