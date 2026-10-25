"""Shared builders and fresh-process continuation worker for enhanced sampling.

The integration scenarios and the fresh interpreter rebuild their runtimes
from these deterministic declarations, so a checkpoint written in one process
authenticates against the runtime rebuilt in the other.

Run as ``python _enhanced_sampling_restart_worker.py BIASED MULTISTATE OUTPUT``:
it reads the biased-dynamics and multistate checkpoints, continues both with
exactly the continuation the in-process test performs and writes every array
leaf of the continued states to ``OUTPUT`` (``.npz``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax import config as pxconfig
from phydrax.atomistic._classical import HarmonicBondPotential
from phydrax.atomistic._dynamics import AtomisticDynamicsPlan
from phydrax.atomistic._thermal import BAOABLangevinPlan
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.atomistic.sampling import (
    AtomisticBiasHistoryPlan,
    AtomisticBiasRowBinding,
    AtomisticBiasTerm,
    AtomisticCanonicalSamplingQualification,
    AtomisticMultistateCheckpointPlan,
    AtomisticMultistatePlan,
    AtomisticMultistateSegmentPlan,
    AtomisticReplicaExchangePlan,
    AtomisticThermodynamicStatePlan,
    CollectiveVariableKind,
    HarmonicBiasForm,
    HillHistoryBiasForm,
    PreparedBiasedDynamics,
    PreparedThermodynamicStateTable,
    read_atomistic_multistate_checkpoint,
    read_biased_dynamics_checkpoint,
)
from tests._support.atomistic_dynamics_fixtures import (
    atomistic_system,
    measure,
    prepare_potential,
    row,
    table,
)
from tests._support.atomistic_sampling_fixtures import (
    bias_term,
    biased_dynamics,
    cv_program,
)


# Reduced units (k_B = 1).  Physical pair Hamiltonian U(r) = K/2 (r - R0)^2.
BOND_STIFFNESS = 2.0
BOND_LENGTH = 1.0
WINDOW_STIFFNESS = 20.0
WINDOW_TEMPERATURE = 1.0
HILL_HEIGHT = 0.2
HILL_WIDTH = 0.1
HILL_CAPACITY = 16
FREEZE_CENTERS = (0.95, 1.05)

METADYNAMICS_POSITIONS = ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (0.0, 2.0, 0.0))
METADYNAMICS_VELOCITIES = ((-0.3, 0.1, 0.0), (0.3, -0.1, 0.0), (0.0, 0.0, 0.2))

SAVED_BIASED_STEPS = 3
CONTINUED_BIASED_STEPS = 4
SEGMENT_CAPACITY = 3


# ----------------------------------------------------------------- biased walker


@eqx.filter_jit
def biased_step(runtime: PreparedBiasedDynamics, state: Any) -> Any:
    return runtime.step(state)


def metadynamics_runtime() -> tuple[PreparedBiasedDynamics, Any]:
    """Stochastic (BAOAB) well-tempered distance metadynamics walker."""
    system = atomistic_system([0, 1, 2], [1.0, 1.0, 1.0])
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    history = AtomisticBiasHistoryPlan(
        1,
        height=0.3,
        widths=0.1,
        capacity=16,
        bias_factor=5.0,
        temperature=1.0,
    )
    term = bias_term(system, variables, HillHistoryBiasForm(history))
    dynamics = biased_dynamics(system, term, BAOABLangevinPlan(0.01, 1.0))
    thermodynamic = table(dynamics, row(measure(system), "nvt", temperature=1.0))
    runtime = PreparedBiasedDynamics(dynamics, thermodynamic, physical_force_groups=(0,))
    positions = jnp.asarray(METADYNAMICS_POSITIONS, dtype=jnp.float64)
    initial = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.asarray(METADYNAMICS_VELOCITIES, dtype=jnp.float64),
        key=jax.random.key(7),
    )
    return runtime, runtime.initialize(initial)


def advance_biased(
    runtime: PreparedBiasedDynamics, state: Any, steps: int, /
) -> tuple[Any, list[bool]]:
    successes = []
    for _ in range(steps):
        evaluation = biased_step(runtime, state)
        successes.append(bool(evaluation.successful))
        state = evaluation.accepted_state
    return state, successes


# ------------------------------------------------------------- umbrella windows


def window_hamiltonian(*, history: bool) -> tuple[Any, Any, AtomisticBiasTerm, Any]:
    """Bonded pair, a distance umbrella and (optionally) a shared hill history."""
    system = atomistic_system(
        [0, 1], [1.0, 1.0], topology=MolecularTopologyPlan(bonds=[[0, 1]])
    )
    variables = cv_program(system, CollectiveVariableKind.DISTANCE, [0, 1])
    umbrella = AtomisticBiasTerm(
        variables,
        HarmonicBiasForm(np.asarray([1.0]), WINDOW_STIFFNESS),
        name="window-umbrella",
        force_group=1,
    )
    terms: tuple[Any, ...] = (
        HarmonicBondPotential(np.asarray([BOND_STIFFNESS]), np.asarray([BOND_LENGTH])),
        umbrella,
    )
    hill = None
    if history:
        plan = AtomisticBiasHistoryPlan(
            1, height=HILL_HEIGHT, widths=HILL_WIDTH, capacity=HILL_CAPACITY
        )
        hill = AtomisticBiasTerm(
            variables, HillHistoryBiasForm(plan), name="window-history", force_group=1
        )
        terms = (*terms, hill)
    potential = prepare_potential(system, terms)
    dynamics = AtomisticDynamicsPlan(
        system, potential, BAOABLangevinPlan(0.01, 1.0)
    ).prepare()
    return system, dynamics, umbrella, hill


def window_table(
    dynamics: Any,
    umbrella: AtomisticBiasTerm,
    hill: AtomisticBiasTerm | None,
    centers: tuple[float, ...],
    declaration: str | None,
    /,
) -> PreparedThermodynamicStateTable:
    histories = None if hill is None or declaration is None else {hill.name: declaration}
    rows = tuple(
        AtomisticThermodynamicStatePlan(
            measure(dynamics.system),
            ensemble="nvt",
            temperature=WINDOW_TEMPERATURE,
            bias=AtomisticBiasRowBinding(
                {umbrella.name: np.asarray([center])}, histories=histories
            ),
            state_id=f"window-{index}",
        )
        for index, center in enumerate(centers)
    )
    return PreparedThermodynamicStateTable(dynamics, rows)


def window_runtime(dynamics: Any, thermodynamic: Any, run_id: str, /) -> Any:
    qualification = AtomisticCanonicalSamplingQualification(
        dynamics,
        thermodynamic,
        "enhanced-sampling-window-qualification",
        sampling_exact=False,
        sampling_bias_bound=1.0,
    )
    return AtomisticMultistatePlan(
        thermodynamic,
        np.arange(thermodynamic.state_count),
        qualification=qualification,
        exchange=AtomisticReplicaExchangePlan(1),
        run_id=run_id,
    ).prepare(dynamics)


def window_positions(distance: float, /) -> jax.Array:
    return jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=jnp.float64)


def window_lanes(
    dynamics: Any, thermodynamic: Any, centers: tuple[float, ...], /
) -> tuple:
    """Replica ``i`` starts at rest at window ``i``'s center."""
    return tuple(
        dynamics.initialize_state(
            window_positions(center),
            thermodynamic,
            state_index=index,
            velocity=jnp.zeros((2, 3), dtype=jnp.float64),
            key=jax.random.key(50 + index),
        )
        for index, center in enumerate(centers)
    )


def restart_multistate_runtime() -> tuple[Any, Any]:
    """Adaptive shared-history window exchange used by the restart scenario."""
    _, dynamics, umbrella, hill = window_hamiltonian(history=True)
    thermodynamic = window_table(dynamics, umbrella, hill, FREEZE_CENTERS, "adaptive")
    runtime = window_runtime(dynamics, thermodynamic, "enhanced-sampling-restart")
    initial = runtime.initialize(
        window_lanes(dynamics, thermodynamic, FREEZE_CENTERS),
        [0, 1],
        jax.random.key(61),
    )
    return runtime, initial


def continue_multistate(runtime: Any, segment: Any, /) -> Any:
    """Next committed segment after ``segment`` (watermark and predecessor chained)."""
    return AtomisticMultistateSegmentPlan(
        runtime,
        SEGMENT_CAPACITY,
        int(np.asarray(segment.stop_watermark)),
        int(np.asarray(segment.successor_state.segment_index)),
        segment.segment_id,
    ).run(segment.successor_state)


# ------------------------------------------------------------------- transport


def host_leaves(tree: Any, /) -> list[np.ndarray]:
    """Every array leaf in flattening order, typed PRNG keys as raw key data."""

    def host(leaf: Any) -> np.ndarray:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            leaf = jax.random.key_data(leaf)
        return np.asarray(leaf)

    return [host(leaf) for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))]


def leaf_archive(tree: Any, /, **identities: str) -> dict[str, np.ndarray]:
    arrays = {f"leaf_{index:05d}": leaf for index, leaf in enumerate(host_leaves(tree))}
    arrays.update({name: np.asarray(value) for name, value in identities.items()})
    return arrays


def main(arguments: list[str], /) -> int:
    biased_path, multistate_path, output = (Path(value) for value in arguments)
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        runtime, template = metadynamics_runtime()
        restored = read_biased_dynamics_checkpoint(biased_path, runtime, template)
        continued, successes = advance_biased(
            runtime, restored.state, CONTINUED_BIASED_STEPS
        )
        if not all(successes):
            raise RuntimeError(f"Fresh-process biased continuation failed: {successes}.")
        ms_runtime, ms_template = restart_multistate_runtime()
        checkpoint = read_atomistic_multistate_checkpoint(
            multistate_path,
            AtomisticMultistateCheckpointPlan(ms_runtime, SEGMENT_CAPACITY),
            ms_template,
        )
        segment = continue_multistate(ms_runtime, checkpoint.segment)
        if not bool(np.asarray(segment.successful)):
            raise RuntimeError("Fresh-process multistate continuation failed.")
        np.savez(
            output,
            allow_pickle=True,
            **leaf_archive(
                (continued, segment),
                biased_payload=restored.payload_id,
                multistate_payload=checkpoint.payload_id,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
