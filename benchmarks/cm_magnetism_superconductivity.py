"""Decision benchmark for bounded magnetism, LSWT, and fermionic BdG kernels."""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    realize_atomistic_graph,
)
from phydrax.atomistic._spin import (
    ClassicalSpinHamiltonianPlan,
    evaluate_classical_spin_hamiltonian,
    prepare_classical_spin_hamiltonian,
)
from phydrax.discretization import PeriodicCell
from phydrax.discretization._reciprocal import ReciprocalMeshPlan
from phydrax.operators.periodic._family import (
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
)
from phydrax.operators.quantum._fermionic_fock import FermionModeOrder
from phydrax.operators.quantum._spin_wave import (
    evaluate_linear_spin_wave,
    LinearSpinWavePlan,
    prepare_linear_spin_wave,
    SpinWaveReferenceState,
)
from phydrax.operators.quantum._superconductivity import (
    evaluate_bdg_spectrum,
    FermionicBdGPlan,
    FermionicPairingPlan,
    NambuConvention,
    prepare_fermionic_bdg,
)
from phydrax.sparse import EdgeRelation


def _constant_family(matrix, *, hermitian):
    value = np.asarray(matrix, dtype=complex)
    count = value.shape[0]
    target = np.repeat(np.arange(count), count)
    source = np.tile(np.arange(count), count)
    reverse = np.arange(count * count).reshape((count, count)).T.reshape((-1,))
    relation = EdgeRelation(source, target, source_size=count, target_size=count)
    plan = PeriodicTranslationFamilyPlan(
        relation,
        np.zeros((count * count, 1), dtype=int),
        reverse,
        hermitian=hermitian,
    )
    state = PeriodicTranslationFamilyState(
        plan, value[target, source].reshape((-1, 1, 1))
    )
    return prepare_periodic_translation_family(plan, state)


def _measure(function):
    start = time.perf_counter()
    value = function()
    jax.block_until_ready(value)
    return value, time.perf_counter() - start


def _classical_case(site_count: int):
    units = AtomisticUnitSystem.reduced()
    positions = np.zeros((site_count, 3))
    positions[:, 0] = np.arange(site_count)
    system = AtomisticSystemPlan(
        np.arange(site_count),
        np.ones(site_count, dtype=int),
        np.ones(site_count),
        units,
    ).prepare()
    structure = AtomicStructure(
        np.ones(site_count, dtype=int),
        positions,
        np.ones(site_count),
        units.scale,
        particle_ids=np.arange(site_count),
    )
    graph = realize_atomistic_graph(
        AtomisticBatch.from_structure(structure),
        AtomisticGraphExecutionPlan(2, maximum_dense_atoms=site_count),
        cutoff=1.01,
    )
    plan = ClassicalSpinHamiltonianPlan(
        system,
        graph,
        energy_unit="reduced-energy",
        field_unit="reduced-field",
        moment_unit="reduced-energy/reduced-field",
    )
    prepared = prepare_classical_spin_hamiltonian(
        plan,
        exchange=np.ones(plan.bond_count),
        moments=np.ones(site_count),
        magnetic_field=np.broadcast_to([0.0, 0.0, 0.1], (site_count, 3)),
    )
    directions = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (site_count, 3))
    result, elapsed = _measure(
        lambda: evaluate_classical_spin_hamiltonian(prepared, directions).effective_field
    )
    return {
        "sites": site_count,
        "once_only_bonds": plan.bond_count,
        "seconds": elapsed,
        "logical_input_bytes": int(
            prepared.exchange.nbytes
            + prepared.symmetric_exchange.nbytes
            + prepared.dmi.nbytes
            + directions.nbytes
        ),
        "field_norm": float(jnp.linalg.norm(result)),
    }


def _lswt_case(site_count: int, q_count: int):
    reference = SpinWaveReferenceState(
        np.broadcast_to([0.0, 0.0, 1.0], (site_count, 3)), np.ones(site_count)
    )
    mesh = ReciprocalMeshPlan.monkhorst_pack(PeriodicCell([[1.0]]), (q_count,))
    plan = LinearSpinWavePlan(
        reference,
        mesh,
        np.arange(q_count)[::-1],
        np.zeros(q_count, dtype=int),
        energy_unit="reduced-energy",
        maximum_site_count=site_count,
    )
    normal_family = _constant_family(np.eye(site_count), hermitian=True)
    pairing_family = _constant_family(np.zeros((site_count, site_count)), hermitian=False)
    prepared = prepare_linear_spin_wave(
        plan,
        normal_family,
        pairing_family,
        np.zeros((site_count, 3)),
    )
    frequencies, elapsed = _measure(
        lambda: evaluate_linear_spin_wave(prepared).frequencies
    )
    return {
        "sites_per_cell": site_count,
        "q_points": q_count,
        "seconds": elapsed,
        "logical_input_bytes": int(
            normal_family.state.values.nbytes + pairing_family.state.values.nbytes
        ),
        "minimum_frequency": float(jnp.min(frequencies)),
    }


def _bdg_case(mode_count: int, k_count: int):
    if mode_count % 2:
        raise ValueError("BdG benchmark mode_count must be even.")
    order = FermionModeOrder(tuple(f"mode-{index}" for index in range(mode_count)))
    pairing_plan = FermionicPairingPlan(
        order,
        np.arange(k_count),
        np.full(k_count, 1.0 / k_count),
        mesh_id=f"benchmark-k-{k_count}",
        energy_unit="reduced-energy",
    )
    plan = FermionicBdGPlan(
        NambuConvention(order), pairing_plan, maximum_mode_count=mode_count
    )
    diagonal = np.linspace(-1.0, 1.0, mode_count)
    normal = np.broadcast_to(np.diag(diagonal), (k_count, mode_count, mode_count)).copy()
    pair = np.zeros((mode_count, mode_count), dtype=complex)
    for index in range(0, mode_count, 2):
        pair[index, index + 1] = 0.2
        pair[index + 1, index] = -0.2
    pairing = np.broadcast_to(pair, normal.shape).copy()
    prepared = prepare_fermionic_bdg(
        plan,
        normal,
        pairing,
        chemical_potential=0.0,
        normal_family_id="benchmark-normal",
        pairing_family_id="benchmark-pairing",
    )
    eigenvalues, elapsed = _measure(lambda: evaluate_bdg_spectrum(prepared).eigenvalues)
    return {
        "particle_modes": mode_count,
        "nambu_dimension": 2 * mode_count,
        "k_points": k_count,
        "seconds": elapsed,
        "logical_input_bytes": int(normal.nbytes + pairing.nbytes),
        "minimum_gap": float(jnp.min(jnp.abs(eigenvalues))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--q-points", type=int, default=8)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--k-points", type=int, default=8)
    args = parser.parse_args()
    if args.sites < 2 or args.q_points < 1 or args.modes < 2 or args.k_points < 1:
        raise ValueError(
            "Benchmark sizes must be positive; sites and modes must be at least two."
        )
    report = {
        "classical_spin": _classical_case(args.sites),
        "linear_spin_wave": _lswt_case(args.sites, args.q_points),
        "fermionic_bdg": _bdg_case(args.modes, args.k_points),
        "backend": jax.default_backend(),
        "dtype": "float64/complex128",
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
