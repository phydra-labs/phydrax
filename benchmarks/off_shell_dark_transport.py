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
from phydrax.applications.curved_spacetime_qft._kadanoff_baym import (
    advance_kadanoff_baym,
    initialize_kadanoff_baym_memory,
    KadanoffBaymTransportPlan,
    WignerGradientPlan,
)
from phydrax.applications.curved_spacetime_qft._off_shell_transport import (
    breit_wigner_off_shell_state,
    OffShellTransportPlan,
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
from phydrax.solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    empty_dark_sector_epoch_state,
)


def _units_and_frame():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1)
    units = RelativisticUnitContract(
        scale, RelativityConvention(metric_signature="mostly_minus")
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
        snapshot_token=jnp.asarray(12),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="off-shell-benchmark-grid",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="off-shell-benchmark-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="off-shell-benchmark-observer",
        orientation_id="right-handed-future",
    )
    return units, frame


def benchmark(
    cells: int, momenta: int, energy_nodes: int, memory_depth: int, repeats: int
) -> dict:
    units, frame = _units_and_frame()
    species = (
        DarkSectorSpeciesPlan(
            "benchmark-off-shell-fermion",
            1.0,
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ),
    )
    support = QuantumKineticState(
        jnp.full((1, cells, momenta), 0.2),
        None,
        species=species,
        statistics=jnp.asarray((-1,), dtype=jnp.int8),
        spatial_active=jnp.ones((cells,), dtype=bool),
        momentum_active=jnp.ones((momenta,), dtype=bool),
        units=units,
        frame=frame,
    )
    omega = np.linspace(-8.0, 12.0, energy_nodes)
    weights = np.full(omega.shape, omega[1] - omega[0])
    weights[[0, -1]] *= 0.5
    off_shell = OffShellTransportPlan(
        support,
        omega,
        weights,
        jnp.full((momenta,), 1.0 / momenta),
        jnp.ones((cells,)),
        energy_quadrature_id=f"uniform-energy-{energy_nodes}",
        momentum_quadrature_id=f"uniform-momentum-{momenta}",
        spectral_tolerance=0.05,
    )
    pole = jnp.linspace(0.8, 2.0, momenta)[:, None]
    state = breit_wigner_off_shell_state(
        off_shell,
        pole,
        0.2,
        inverse_temperature=0.25,
        chemical_potentials=jnp.asarray((0.0,)),
    )
    epoch = DarkSectorEpochPlan(
        packet_capacity=4,
        event_capacity=4,
        product_capacity=4,
        radiation_capacity=4,
        work_capacity=4,
        frontier_capacity=4,
        packet_width=4,
        event_width=4,
        product_width=4,
        radiation_width=4,
        work_width=4,
        frontier_width=4,
        species_revision_id="3" * 64,
        topology_revision_id="4" * 64,
    )
    gradient = WignerGradientPlan((cells, momenta), (1.0,), (1.0,))
    kb = KadanoffBaymTransportPlan(
        off_shell,
        gradient,
        epoch,
        time_step=0.01,
        memory_depth=memory_depth,
    )
    runtime = empty_dark_sector_epoch_state(epoch, epoch_sequence=0)
    memory = initialize_kadanoff_baym_memory(kb, runtime)
    zeros = jnp.zeros(off_shell.spectral_shape)
    kernel = jnp.full(off_shell.spectral_shape, 1.0e-3)
    result, timing = measure_repeated(
        lambda: advance_kadanoff_baym(
            kb,
            state,
            memory,
            runtime,
            collision_source=zeros,
            statistical_self_energy=zeros,
            real_retarded_propagator=zeros,
            memory_kernel=kernel,
            initial_correlation=zeros,
        ),
        warmup=1,
        repeats=repeats,
    )
    spectral_arrays = (
        state.spectral_function,
        state.occupation,
        state.real_retarded_self_energy,
        state.width,
        state.lesser_self_energy,
        state.greater_self_energy,
    )
    memory_arrays = (
        memory.source_history,
        memory.kernel_history,
        memory.initial_correlation_history,
        memory.sample_times,
        memory.valid,
    )
    return {
        "accepted": bool(result.evidence.accepted),
        "dyson_residual": float(result.accepted_state.evidence.dyson_residual),
        "kms_residual": float(result.accepted_state.evidence.kms_residual),
        "spectral_sum_rule_residual": float(
            result.accepted_state.evidence.spectral_sum_rule_residual
        ),
        "execution": timing.to_milliseconds_dict(),
        "resources": {
            "cells": cells,
            "momenta": momenta,
            "species": 1,
            "energy_nodes": energy_nodes,
            "memory_depth": memory_depth,
            "spectral_elements_per_field": int(state.spectral_function.size),
            "spectral_bytes_per_field": int(state.spectral_function.nbytes),
            "spectral_state_bytes": int(sum(value.nbytes for value in spectral_arrays)),
            "memory_history_elements": int(
                memory.source_history.size
                + memory.kernel_history.size
                + memory.initial_correlation_history.size
            ),
            "memory_resident_bytes": int(sum(value.nbytes for value in memory_arrays)),
            "maximum_spectral_bytes": off_shell.maximum_spectral_bytes,
            "maximum_memory_bytes": kb.maximum_memory_bytes,
            "capacity_revision_id": kb.capacity_revision_id,
            "compile_signature_id": kb.compile_signature_id,
        },
        "identities": {
            "off_shell_plan_id": off_shell.plan_id,
            "kadanoff_baym_plan_id": kb.plan_id,
            "support_id": off_shell.support_id,
            "frame_id": off_shell.frame_id,
            "frame_realization_id": off_shell.frame_realization_id,
            "unit_contract_id": off_shell.unit_contract_id,
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
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--momenta", type=int, default=16)
    parser.add_argument("--energy-nodes", type=int, default=128)
    parser.add_argument("--memory-depth", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.cells < 3
        or arguments.momenta < 3
        or arguments.energy_nodes < 3
        or arguments.memory_depth < 1
        or arguments.repeats < 1
    ):
        raise ValueError(
            "cells/momenta/energy-nodes require at least three; depth/repeats must be positive."
        )
    payload = {
        "environment": capture_environment().to_dict(),
        "off_shell_dark_transport": benchmark(
            arguments.cells,
            arguments.momenta,
            arguments.energy_nodes,
            arguments.memory_depth,
            arguments.repeats,
        ),
    }
    encoded = json.dumps(payload, indent=2, default=_json_default)
    print(encoded)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
