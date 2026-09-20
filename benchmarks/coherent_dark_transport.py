#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, measure_repeated

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._quantum_dark_kinetics import QuantumKineticState
from phydrax.applications.curved_spacetime_qft._coherent_transport import (
    advance_coherent_transport,
    CoherentTransportPlan,
    initialize_coherent_state,
    LocalKrausCollisionMap,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)


def _units_and_frame():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1)
    units = RelativisticUnitContract(
        scale,
        RelativityConvention(metric_signature="mostly_minus"),
        spin_normalization="density-matrix-explicit",
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(11),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="coherent-benchmark-grid",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="coherent-benchmark-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="coherent-benchmark-observer",
        orientation_id="right-handed-future",
    )
    return units, frame


def benchmark(cells: int, momenta: int, repeats: int) -> dict:
    units, frame = _units_and_frame()
    species = (
        DarkSectorSpeciesPlan(
            "benchmark-dark-fermion",
            1.0,
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ),
    )
    support = QuantumKineticState(
        jnp.full((1, cells, momenta), 0.5),
        None,
        species=species,
        statistics=jnp.asarray((-1,), dtype=jnp.int8),
        spatial_active=jnp.ones((cells,), dtype="bool"),
        momentum_active=jnp.ones((momenta,), dtype="bool"),
        units=units,
        frame=frame,
    )
    plan = CoherentTransportPlan(
        support,
        jnp.full((momenta,), 1.0 / momenta),
        jnp.ones((cells,)),
        momentum_quadrature_id=f"uniform-{momenta}",
        internal_dimension=2,
    )
    state = initialize_coherent_state(plan)
    hamiltonian = jnp.asarray([[0.2, 0.7], [0.7, -0.2]], dtype=jnp.complex128)
    zero = jnp.zeros_like(hamiltonian)
    probability = 0.02
    kraus = jnp.stack(
        (
            jnp.sqrt(1.0 - probability) * jnp.eye(2),
            jnp.sqrt(probability) * jnp.diag(jnp.asarray((1.0, -1.0))),
        )
    ).astype(jnp.complex128)
    collision = LocalKrausCollisionMap(
        kraus,
        jnp.asarray((True, True)),
        internal_dimension=2,
        map_id="benchmark-dephasing",
    )
    result, timing = measure_repeated(
        lambda: advance_coherent_transport(
            plan,
            state,
            time_step=0.01,
            vacuum_hamiltonian=hamiltonian,
            mean_field_hamiltonian=zero,
            gravity_hamiltonian=zero,
            gauge_hamiltonian=zero,
            collision=collision,
        ),
        warmup=1,
        repeats=repeats,
    )
    return {
        "accepted": bool(result.accepted),
        "unitary_residual": float(result.unitary_residual),
        "trace_residual": float(result.trace_residual),
        "execution": timing.to_milliseconds_dict(),
        "resources": {
            "cells": cells,
            "momenta": momenta,
            "species": 1,
            "internal_dimension": 2,
            "density_matrix_elements": state.density_matrix.size,
            "density_matrix_bytes": state.density_matrix.nbytes,
            "hamiltonian_broadcast_elements": int(np.prod(plan.density_shape)),
            "hamiltonian_broadcast_bytes": int(
                np.prod(plan.density_shape) * hamiltonian.dtype.itemsize
            ),
            "kraus_elements": kraus.size,
            "kraus_bytes": kraus.nbytes,
            "maximum_matrix_bytes": plan.maximum_matrix_bytes,
        },
        "identities": {
            "plan_id": plan.plan_id,
            "support_id": plan.support_id,
            "frame_id": plan.frame_id,
            "frame_realization_id": plan.frame_realization_id,
            "unit_contract_id": plan.unit_contract_id,
        },
    }


def _json_default(value):
    if isinstance(value, jax.Array):
        host = np.asarray(jax.device_get(value))
        return host.item() if host.shape == () else host.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--momenta", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.cells < 1 or arguments.momenta < 1 or arguments.repeats < 1:
        raise ValueError("cells, momenta, and repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "coherent_dark_transport": benchmark(
            arguments.cells, arguments.momenta, arguments.repeats
        ),
    }
    encoded = json.dumps(payload, indent=2, default=_json_default)
    print(encoded)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
