#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical-B PIC smoke and controlling particle-capacity benchmark campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._runtime import (
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications.numerical_relativity import NumericalRelativityDistributedPlan
from phydrax.discretization import (
    ChargedParticlePlan,
    ParticleSetPlan,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization.pic import (
    ChargeConservingCurrentPlan,
    PICParticleCochainTransferPlan,
)
from phydrax.exterior import FormType
from phydrax.solver import (
    CochainElectrostaticBoundaryPlan,
    CochainElectrostaticPlan,
    CochainMaxwellPICFieldSolver,
    CompatibleMaxwellPlan,
    PICMaxwellCurrentSourcePlan,
)
from phydrax.solver.maxwell import (
    CompatibleMaxwellState,
    LorentzDrudeMaxwellConstitutivePlan,
    MaxwellLorentzPoles,
)


def _prepare(
    capacity: int,
) -> tuple[CochainMaxwellPICFieldSolver, CompatibleMaxwellState, Array, Array]:
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = StructuredCochainBridge(grid)
    if bridge.dimension != 3:
        raise ValueError("Physical magnetic flux smoke requires a 3D bridge.")
    support = ParticleSetPlan(
        jnp.arange(capacity, dtype=jnp.int32),
        jnp.ones((capacity,), dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    charged = ChargedParticlePlan(
        jnp.ones((capacity,), dtype=jnp.float64), "probe"
    ).prepare(support)
    transfer = PICParticleCochainTransferPlan(bridge).prepare(charged)
    material = LorentzDrudeMaxwellConstitutivePlan(
        magnetic_poles=MaxwellLorentzPoles([1.0], [0.2], [0.7]),
        permeability_infinity=2.0,
    )
    maxwell = CompatibleMaxwellPlan(
        bridge,
        constitutive=material,
        sources=(PICMaxwellCurrentSourcePlan(),),
    ).prepare()
    solver = CochainMaxwellPICFieldSolver(
        maxwell,
        CochainElectrostaticPlan(
            bridge, CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        (transfer,),
        (ChargeConservingCurrentPlan(transfer),),
    )
    flux = bridge.pack_face_flux(
        (
            jnp.full((4, 4, 4), 0.4, dtype=jnp.float64),
            jnp.full((4, 4, 4), -0.3, dtype=jnp.float64),
            jnp.full((4, 4, 4), 1.1, dtype=jnp.float64),
        )
    )
    field = maxwell.initialize(magnetic_flux=flux)
    field = eqx.tree_at(
        lambda value: value.auxiliary.material.magnetization, field, flux[None, :]
    )
    positions = jnp.asarray(
        np.random.default_rng(147).uniform(0.05, 0.95, size=(capacity, 3)),
        dtype=jnp.float64,
    )
    return solver, field, positions, jnp.ones((capacity,), dtype=jnp.bool_)


def run_case(capacity: int, repeats: int, /) -> dict[str, Any]:
    (solver, field, positions, active), preparation = measure_synchronized(
        lambda: _prepare(capacity)
    )

    def gather(
        state: CompatibleMaxwellState, points: Array, mask: Array
    ) -> tuple[Array, Array, Array]:
        return solver.gather_fields(0, points, mask, state)

    prepared = jax.jit(gather)
    compiled, compilation = measure_lower_and_compile(
        lambda: prepared.lower(field, positions, active),
        lambda lowered: lowered.compile(),
    )
    result, first = measure_synchronized(lambda: compiled(field, positions, active))
    result, warm = measure_repeated(
        lambda: compiled(field, positions, active), warmup=1, repeats=repeats
    )
    electric, magnetic, support = result
    np.testing.assert_array_equal(support, True)
    np.testing.assert_array_equal(electric, 0.0)
    expected = np.broadcast_to(np.asarray([0.4, -0.3, 1.1]), (capacity, 3))
    np.testing.assert_allclose(magnetic, expected, atol=1e-13)
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax.jit.coChainPICPhysicalGather",
    )
    return {
        "particle_capacity": capacity,
        "preparation_seconds": preparation,
        "compilation": asdict(compilation),
        "first_execution_seconds": first,
        "warm_execution": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "retained_array_bytes": logical_array_bytes((solver, field, positions, active)),
        "maximum_magnetic_error": float(np.max(np.abs(np.asarray(magnetic) - expected))),
        "supported": bool(np.all(np.asarray(support))),
    }


def nr_sharding_smoke() -> dict[str, Any]:
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = StructuredCochainBridge(grid)
    distribution = NumericalRelativityDistributedPlan(
        "grmhd",
        (4, 4, 4),
        (1, 1, 1),
        halo_width=1,
        periodic=(True, True, True),
        grid_id=grid.topology.topology_id,
    ).prepare(jax.devices()[:1])
    form_type = FormType(3, 2)
    values = jnp.arange(bridge.cochain.cell_counts[2], dtype=jnp.float64)
    state = distribution.shard_cochain(bridge, form_type, values)
    np.testing.assert_array_equal(bridge.pack(2, state.components), values)
    shardings = distribution.cochain_component_shardings(bridge, 2)
    if any(
        component.sharding != sharding
        for component, sharding in zip(state.components, shardings, strict=True)
    ):
        raise RuntimeError("NR cochain component sharding changed.")
    if state.realization_id != bridge.cochain.realization_id:
        raise RuntimeError("NR cochain realization identity changed.")
    return {
        "form_type_id": state.form_type.form_type_id,
        "realization_id": state.realization_id,
        "layout_id": state.layout_id,
        "distribution_id": state.distribution_id,
        "component_shardings": [str(value.spec) for value in shardings],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacities", nargs="+", type=int, default=[2, 16, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.capacities) or arguments.repeats < 1:
        parser.error("capacities and repeats must be positive")
    record = {
        "cases": [run_case(value, arguments.repeats) for value in arguments.capacities],
        "nr_sharding": nr_sharding_smoke(),
    }
    encoded = json.dumps(record, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
