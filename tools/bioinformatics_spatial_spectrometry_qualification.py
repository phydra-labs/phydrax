#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Spatial graph/statistic and mass-spectrometry binning qualification."""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.bioinformatics import spatial, spectrometry
from tools.bioinformatics_common_qualification import (
    emit_report,
    fingerprint,
    method_contract_evidence,
    qualification_report,
)


def _spatial_moran_case() -> dict[str, object]:
    coordinates = jnp.asarray(((0.0,), (1.0,), (2.0,), (3.0,)))
    values = jnp.asarray((1.0, 2.0, 4.0, 8.0))
    donors = jnp.asarray((0, 0, 1, 1), dtype=jnp.int32)
    sections = jnp.zeros((4,), dtype=jnp.int32)
    plan = spatial.SpatialNeighborPlan("radius", capacity=2, radius=1.1, weight="binary")
    graph = spatial.build_spatial_neighbor_graph(
        coordinates, plan, section_index=sections
    )
    result = spatial.spatial_autocorrelation_test(
        values,
        graph,
        jr.key(0),
        statistic="moran",
        permutations=16,
        donor_index=donors,
        section_index=sections,
    )

    host_values = np.asarray(values)
    centered = host_values - np.mean(host_values)
    indices = np.asarray(graph.relation.source_indices)
    routes = np.asarray(graph.relation.valid)
    weights = np.asarray(graph.weight) * routes
    graph_mass = float(np.sum(weights))
    numerator = float(np.sum(weights * centered[:, None] * centered[indices]))
    denominator = float(np.sum(centered * centered))
    oracle = len(host_values) * numerator / (graph_mass * denominator)
    observed = float(np.asarray(result.statistic))
    statistic_error = abs(observed - oracle)

    insufficient_plan = spatial.SpatialNeighborPlan(
        "radius", capacity=1, radius=1.1, weight="binary"
    )
    insufficient = spatial.build_spatial_neighbor_graph(
        coordinates, insufficient_plan, section_index=sections
    )
    capacity_rejected = (
        not bool(np.asarray(insufficient.valid))
        and int(np.asarray(insufficient.status)) != 0
        and int(np.asarray(insufficient.evidence.required_capacity)) == 2
    )
    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "coordinates": coordinates,
        "values": values,
        "donor_index": donors,
        "section_index": sections,
        "neighbor_mode": "radius",
        "radius": 1.1,
        "capacity": 2,
        "permutations": 16,
        "random_key": [0, 0],
    }
    return {
        "scope": "unit_qualification",
        "oracle": "direct weighted Moran ratio on the emitted sparse graph",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "experimental_units": {
            "independent_unit": "donor",
            "donor_count": int(np.asarray(result.evidence.donor_count)),
            "section_count": int(np.asarray(result.evidence.section_count)),
            "exchangeability_group_count": int(
                np.asarray(result.evidence.exchangeability_group_count)
            ),
        },
        "observed_moran": observed,
        "oracle_moran": oracle,
        "absolute_statistic_error": statistic_error,
        "permutation_p_value": float(np.asarray(result.p_value)),
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "capacity_check": {
            "configured_route_capacity": 1,
            "required_route_capacity": int(
                np.asarray(insufficient.evidence.required_capacity)
            ),
            "status": int(np.asarray(insufficient.status)),
            "rejected": capacity_rejected,
        },
        "passed": bool(
            np.asarray(result.valid) and statistic_error <= 2.0e-6 and capacity_rejected
        ),
    }


def _spectrometry_binning_case() -> dict[str, object]:
    spectrum = spectrometry.MassSpectrum(
        jnp.asarray((100.0, 101.0, 102.0, 103.0, 103.5)),
        jnp.asarray((1.0, 2.0, 3.0, 4.0, 5.0)),
        scan_id=7,
        retention_time=12.5,
    )
    plan = spectrometry.MassBinningPlan(jnp.asarray((99.5, 101.5, 103.5)))
    result = spectrometry.bin_mass_spectrum(spectrum, plan)
    expected_intensity = np.asarray((3.0, 12.0))
    intensity_error = float(
        np.max(np.abs(np.asarray(result.intensity) - expected_intensity))
    )

    def objective(intensity):
        varied = eqx.tree_at(
            lambda candidate: candidate.intensity,
            spectrum,
            intensity,
        )
        binned = spectrometry.bin_mass_spectrum(varied, plan)
        return jnp.sum(binned.intensity * binned.intensity)

    automatic_gradient = jax.grad(objective)(spectrum.intensity)
    expected_gradient = jnp.asarray((6.0, 6.0, 24.0, 24.0, 24.0))
    gradient_error = float(
        np.max(np.abs(np.asarray(automatic_gradient - expected_gradient)))
    )

    outside_spectrum = spectrometry.MassSpectrum(
        jnp.asarray((100.0, 104.0)),
        jnp.asarray((1.0, 1.0)),
    )
    outside = spectrometry.bin_mass_spectrum(outside_spectrum, plan)
    outside_rejected = (
        not bool(np.asarray(outside.valid)) and int(np.asarray(outside.status)) != 0
    )
    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "mass_to_charge": spectrum.mass_to_charge,
        "intensity": spectrum.intensity,
        "active_mask": spectrum.active_mask,
        "bin_edges": plan.edges,
        "mass_to_charge_unit": int(spectrum.units.mass_to_charge),
        "intensity_unit": int(spectrum.units.intensity),
    }
    return {
        "scope": "unit_qualification",
        "oracle": "left-closed/right-open mass bins with final-edge inclusion",
        "gradient_oracle": (
            "gradient of squared bin totals assigns twice each bin total to "
            "every contributing spectral point"
        ),
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "units": {
            "mass_to_charge": spectrum.units.mass_to_charge.name,
            "intensity": spectrum.units.intensity.name,
            "retention_time": spectrum.units.time.name,
        },
        "observed_binned_intensity": np.asarray(result.intensity).tolist(),
        "expected_binned_intensity": expected_intensity.tolist(),
        "maximum_intensity_error": intensity_error,
        "maximum_gradient_identity_error": gradient_error,
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "out_of_range_status_check": {
            "status": int(np.asarray(outside.status)),
            "rejected": outside_rejected,
        },
        "passed": bool(
            np.asarray(result.valid)
            and intensity_error == 0.0
            and gradient_error <= 2.0e-6
            and outside_rejected
        ),
    }


def qualification() -> dict[str, object]:
    return qualification_report(
        "spatial_spectrometry",
        {
            "design_aware_moran": _spatial_moran_case(),
            "mass_spectrum_binning": _spectrometry_binning_case(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify public spatial-statistics and spectrometry APIs."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    return emit_report(qualification(), arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
