#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite SU(2), spin-network, EPRL-semantic, and zero-spin booster evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phydrax.applications import spin_foam, spin_network
from phydrax.operators.quantum.lattice import SU2SectorResourcePolicy


def _resources():
    return SU2SectorResourcePolicy(
        maximum_product_dimension=10_000,
        maximum_sector_dimension=1_000,
        maximum_matrix_elements=10_000_000,
    )


def run_qualification() -> dict[str, object]:
    bf = spin_foam.assess_su2_bf_identities(3)
    edges = tuple(
        spin_network.SpinNetworkEdge(f"e{index}", "left", "right", 2)
        for index in range(3)
    )
    network = spin_network.prepare_spin_network(
        spin_network.SpinNetworkGraphPlan(
            ("left", "right"),
            edges,
            {
                "left": ("e0", "e1", "e2"),
                "right": ("e0", "e1", "e2"),
            },
            _resources(),
        ),
        immirzi_parameter=0.2,
        planck_length_squared=1.0,
    )
    eprl_plan = spin_foam.EPRLVertexPlan(
        (0,) * 10,
        (0,) * 5,
        immirzi_parameter=0.2,
        delta_l=1,
        face_amplitude="dimension-2j-plus-1",
        edge_amplitude="unit",
        coherent_phase_convention="condon-shortley-outward-normals",
        normal_frame_id="five-outward-time-gauge-normals",
        quadrature_id="external-b4-declared",
        precision_bits=256,
        maximum_support_tuples=2048,
    )
    eprl = spin_foam.assess_eprl_semantics(eprl_plan)
    booster = spin_foam.evaluate_zero_spin_b4_booster(
        spin_foam.SL2CBoosterReferencePlan(
            radial_cutoff=20.0,
            quadrature_order=256,
            tolerance=1e-11,
        )
    )
    profiles = (
        *spin_network.spin_network_candidate_profiles(),
        *spin_foam.spin_foam_candidate_profiles(),
    )
    successful = bool(
        bf.accepted and network.evidence.accepted and eprl.accepted and booster.accepted
    )
    return {
        "kind": "spin-quantum-geometry-research-qualification",
        "profiles": [profile.to_record() for profile in profiles],
        "case": {
            "spin_network_id": network.prepared_id,
            "eprl_plan_id": eprl_plan.plan_id,
            "booster_plan_id": booster.plan_id,
        },
        "raw": {
            "edge_area_eigenvalues": np.asarray(network.edge_area_eigenvalues).tolist(),
            "vertex_multiplicities": np.asarray(network.vertex_multiplicities).tolist(),
            "eprl_internal_support": [
                list(value) for value in eprl_plan.internal_twice_spin_support
            ],
        },
        "criteria": {
            "cg_orthogonality": float(bf.maximum_clebsch_orthogonality_residual),
            "recoupling_unitarity": float(bf.maximum_recoupling_unitarity_residual),
            "tetrahedral_symmetry": float(bf.maximum_tetrahedral_symmetry_residual),
            "pentagon": float(bf.maximum_pentagon_residual),
            "spin_network_gauge_invariant": bool(network.evidence.gauge_invariant),
            "eprl_support_tuple_count": int(eprl.support_tuple_count),
            "zero_spin_b4": float(booster.value),
            "zero_spin_b4_exact": float(booster.exact_value),
            "zero_spin_b4_relative_error": float(booster.relative_error),
            "zero_spin_b4_tail_bound": float(booster.cutoff_tail_upper_bound),
        },
        "successful": successful,
        "claim": "finite-research-only-su2-and-zero-spin-controls-no-nonzero-eprl-or-quantum-gravity-validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(run_qualification(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
