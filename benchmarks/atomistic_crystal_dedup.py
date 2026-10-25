# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native radial screening and certified crystal dedup cost versus pool size.

Each pool mixes distinct four-atom P1 cells (one single-atom O, Si, Fe and Cu
molecule per asymmetric unit; random fractional sites in one fixed lattice,
drawn until every pair keeps a declared minimum-image separation and every
atom has a partner inside the interaction range) with certified-equivalent
copies of them (integer basis change, origin shift and a jitter below the
equivalence tolerance; the atom order is the asymmetric unit's fixed site
order).  Every input is relaxed in one batched native fixed-cell relaxation
by ``relax_crystal_candidates`` (P1 atomic symmetry chart, so every atomic
coordinate moves and the lattice is held) under a declared toy switched
Lennard-Jones Hamiltonian over the classical minimum-image pairs, whose
cutoff stays below the lattice's unique-image radius.  That owner computes
the energies, the relaxation evidence and the relaxed structure identities.
Unconverged or failed relaxations are retained and excluded by the ranking
admission (``rank_crystal_candidates`` under the declared policy), never
qualified by hand.  Per candidate the record keeps the initial
minimum-image interacting pairs and separation, the evaluations and the
largest relaxed displacement, so a no-op relaxation is visible.  The
Hamiltonian exercises native mechanics only: it carries no chemical-accuracy
or polymorph-stability claim.

Relaxation preparation (inputs, program preparation, the batched relaxation
including its compilation) and admission are timed separately from the
algorithm phases.  Every algorithm phase consumes the identical admitted
structures in the ranking priority order, under the same radial policy,
equivalence tolerances, bank limits and chunk sizes: fingerprint
construction (first call including compilation, warm repeats), the greedy
radial screen (pairs scored), the persistent certified bank submission with
the ranking values as priorities (certified comparisons, published bytes),
and the composed screen/dedup/rank (screen misses, unconfirmed proposals and
exclusions), with compiler memory of the fixed-shape kernels, logical
fingerprint bytes and host allocation peaks.  Bank commits live in temporary
directories under the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import tempfile
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from functools import partial
from operator import methodcaller
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._memory import host_allocation_peak, process_peak_rss_bytes
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.atomistic import (
    AtomisticDenseImageSearch,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodSource,
    AtomisticPotentialProgram,
    AtomisticRelaxationConvergence,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
    AtomisticSiteDomain,
    AtomisticSymmetryChart,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    LennardJonesPotential,
    PreparedAtomisticPotentialProgram,
)
from phydrax.atomistic.crystal import (
    AsymmetricUnitPlan,
    AsymmetricUnitSite,
    AsymmetricUnitState,
    AtomicSymmetryPlan,
    ConformerPool,
    create_dedup_bank,
    CrystalStructure,
    DedupBankLimits,
    DedupCandidate,
    DedupPolicy,
    EquivalencePolicy,
    expand_asymmetric_unit,
    MolecularComponent,
    MolecularConformer,
    site_of_point,
    standard_space_group_setting,
    submit_to_dedup_bank,
)
from phydrax.atomistic.crystal._ranking import (
    _fingerprint_chunk,
    _score_pairs,
    CrystalRanking,
    CrystalRankingCandidate,
    CrystalRankingPolicy,
    CrystalRelaxationInput,
    radial_fingerprints,
    RadialScreenPolicy,
    rank_crystal_candidates,
    RANKING_NONCLAIM,
    relax_crystal_candidates,
    screen_dedup_rank,
    screen_radial,
)
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from phydrax.optim import TrustScaledLBFGS


_CELL = np.asarray([[5.3, 0.0, 0.0], [-1.4, 5.9, 0.0], [-0.9, 0.7, 7.3]])
_SPECIES = (8, 14, 26, 29)
# CIAAW 2021 conventional standard atomic weights (dalton).
_MASSES = (15.999, 28.085, 55.845, 63.546)
# Ground-term spin multiplicities of the isolated neutral atoms (O 3P, Si 3P,
# Fe 5D, Cu 2S): component identity only; the toy Hamiltonian ignores them.
_MULTIPLICITIES = (3, 3, 5, 2)
_SITE_POINT = (Fraction(1, 5), Fraction(1, 3), Fraction(1, 7))
_TRANSFORM = np.asarray([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
# +-2 translations hold every minimum image of these (also non-reduced) bases.
_IMAGES = np.asarray(list(itertools.product(range(-2, 3), repeat=3)), dtype=np.float64)
_ZERO_IMAGE = 62  # index of (0, 0, 0) in _IMAGES
_MINIMUM_SEPARATION = 1.5  # angstrom, every initial minimum-image pair
_PARTNER_DISTANCE = 2.4  # angstrom, every initial atom has one partner within
_MAXIMUM_DRAWS = 100_000
_LJ_EPSILON = 0.05  # eV, one shared atom type for every element
_LJ_SIGMA = 1.8  # angstrom; pair minimum at 2**(1/6) sigma = 2.02 angstrom
_LJ_SWITCH = 2.3  # angstrom, smooth switching onset
_LJ_CUTOFF = 2.6  # angstrom, below the unique-image radius 2.65 of the lattice
# Covers the lattice condition of both the reference and the copy basis.
_MAXIMUM_CONDITION = 3.0
_LBFGS_MAXIMUM_STEP = 0.02  # angstrom
_ROUTE_WIDTH = 256
_HAMILTONIAN_NONCLAIM = (
    "Declared toy Lennard-Jones Hamiltonian with one atom type shared by O, Si, "
    "Fe and Cu: its energies exercise native relaxation, admission, dedup and "
    "ranking mechanics only and make no chemical-accuracy, force-field or "
    "polymorph-stability claim."
)
_LIMITS = DedupBankLimits(
    max_commit_bytes=64_000_000,
    max_commits=16,
    max_entries=100_000,
    max_stored_atoms=1_000_000,
)


class _Seed(NamedTuple):
    """One relaxation input before expansion (fractional sites, row cell vectors)."""

    structure_id: str
    fractional: np.ndarray
    vectors: np.ndarray


class _Relaxation(NamedTuple):
    """Declared bounded relaxation definition shared by every case."""

    projected_tolerance: float
    maximum_evaluations: int
    maximum_rounds: int


def _equivalence() -> EquivalencePolicy:
    return EquivalencePolicy(
        position_tolerance=0.05,
        lattice_tolerance=1.0e-3,
        supercell="primitive-reduction",
        enantiomorph="proper-only",
        maximum_atoms=64,
        maximum_translation_order=4,
        maximum_lattice_vectors=100_000,
        maximum_basis_checks=1_000_000,
        maximum_bases=256,
        maximum_assignments=10_000,
        maximum_mappings=16,
        maximum_nodes=10_000,
    )


def _asymmetric_unit() -> AsymmetricUnitPlan:
    """P1 asymmetric unit of four single-atom components in ``_SPECIES`` order."""
    setting = standard_space_group_setting(1)
    site = site_of_point(setting, _SITE_POINT)
    sites = []
    for number, mass, multiplicity in zip(
        _SPECIES, _MASSES, _MULTIPLICITIES, strict=True
    ):
        component = MolecularComponent(
            (number,),
            (),
            bond_orders=(),
            masses=(mass,),
            spin_multiplicity=multiplicity,
            valence="unchecked",
            name=f"atom-{number}",
        )
        conformer = MolecularConformer(
            component, np.zeros((1, 3)), provenance="single atom"
        )
        sites.append(AsymmetricUnitSite(ConformerPool(component, (conformer,)), site))
    return AsymmetricUnitPlan(setting, sites, chirality="enantiopure")


def _minimum_image_distances(
    fractional: np.ndarray, vectors: np.ndarray, /
) -> np.ndarray:
    """Pair minimum-image distances (angstrom); the diagonal is infinite."""
    cartesian = fractional @ vectors
    distances = np.linalg.norm(
        cartesian[None, :, None, :] - cartesian[:, None, None, :] + _IMAGES @ vectors,
        axis=-1,
    ).min(axis=-1)
    np.fill_diagonal(distances, np.inf)
    return distances


def _interacting_sites(rng: np.random.Generator, /) -> np.ndarray:
    """Uniform sites kept when separated yet every atom has a partner in range."""
    count = len(_SPECIES)
    for _ in range(_MAXIMUM_DRAWS):
        sites = rng.uniform(0.0, 1.0, size=(count, 3))
        distances = _minimum_image_distances(sites, _CELL)
        if float(distances.min()) >= _MINIMUM_SEPARATION and bool(
            np.all(distances.min(axis=1) <= _PARTNER_DISTANCE)
        ):
            return sites
    raise RuntimeError(
        f"No site draw met the {_MINIMUM_SEPARATION} angstrom separation and "
        f"{_PARTNER_DISTANCE} angstrom partner bounds in {_MAXIMUM_DRAWS} draws."
    )


def _pool(size: int, duplicate_fraction: float, seed: int, /) -> list[_Seed]:
    """Return ``size`` inputs; ``copy-*`` inputs are certified-equivalent copies."""
    rng = np.random.default_rng(seed)
    duplicates = int(round(size * duplicate_fraction))
    families = max(size - duplicates, 1)
    bases = [_interacting_sites(rng) for _ in range(families)]
    pool = [_Seed(f"family-{index}", base, _CELL) for index, base in enumerate(bases)]
    for index in range(size - families):
        base = bases[index % families]
        jitter = rng.normal(scale=0.004, size=base.shape) @ np.linalg.inv(_CELL)
        fractional = (base + jitter) @ np.linalg.inv(_TRANSFORM).T + rng.uniform(
            -0.5, 0.5, size=3
        )
        pool.append(_Seed(f"copy-{index}", fractional, _TRANSFORM.T @ _CELL))
    return pool


def _relaxation_input(
    plan: AsymmetricUnitPlan, item: _Seed, ordinal: int, /
) -> CrystalRelaxationInput:
    sites = len(_SPECIES)
    state = AsymmetricUnitState(
        jnp.asarray(item.fractional - np.floor(item.fractional), dtype=jnp.float64),
        jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64), (sites, 1)),
        plan.chart.coefficients(
            jnp.asarray(item.vectors @ item.vectors.T, dtype=jnp.float64)
        ),
        jnp.zeros((sites,), dtype=jnp.int32),
    )
    return CrystalRelaxationInput(
        state,
        expand_asymmetric_unit(plan, state),
        structure_id=item.structure_id,
        lineage_id=f"atomistic-crystal-dedup-benchmark/{item.structure_id}",
        ordinal=ordinal,
    )


def _program(vectors: np.ndarray, /) -> PreparedAtomisticPotentialProgram:
    count = len(_SPECIES)
    system = AtomisticSystemPlan(
        np.arange(1, count + 1),
        np.asarray(_SPECIES, dtype=np.int32),
        np.asarray(_MASSES, dtype=np.float64),
        AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
        atom_type_ids=np.zeros((count,), dtype=np.int32),
        cell=PeriodicCell(vectors, maximum_condition_number=_MAXIMUM_CONDITION),
    ).prepare()
    program = AtomisticPotentialProgram(
        [
            LennardJonesPotential(
                np.asarray([_LJ_EPSILON]),
                np.asarray([_LJ_SIGMA]),
                _LJ_CUTOFF,
                switch_distance=_LJ_SWITCH,
            )
        ]
    )
    (requirement,) = program.requirements.neighborhoods
    source = AtomisticNeighborhoodSource(
        "pairs",
        AtomisticSiteDomain.PHYSICAL_ATOMS,
        (requirement,),
        AtomisticDenseImageSearch(maximum_dense_routes=1_000_000),
        ParticleImageCapacityLadder(
            (
                ParticleImageCapacity(
                    maximum_particles_per_cell=64,
                    maximum_edges=_ROUTE_WIDTH,
                    maximum_degree=_ROUTE_WIDTH,
                    maximum_images=4096,
                    maximum_case_edges=_ROUTE_WIDTH,
                ),
            )
        ),
        skin=0.3,
    )
    bundle = AtomisticNeighborhoodBundlePlan(
        (requirement,),
        (source,),
        (
            AtomisticRoleCapacityLadder(
                requirement,
                (AtomisticRoleCapacity(maximum_routes=_ROUTE_WIDTH, maximum_degree=64),),
            ),
        ),
    )
    return program.prepare(system, neighborhood_bundle=bundle)


def _convergence(settings: _Relaxation, /) -> AtomisticRelaxationConvergence:
    return AtomisticRelaxationConvergence(
        projected_tolerance=settings.projected_tolerance,
        maximum_evaluations=settings.maximum_evaluations,
    )


def _relaxation_definition(settings: _Relaxation, /) -> dict[str, Any]:
    return {
        "hamiltonian": {
            "term": "LennardJonesPotential",
            "pairs": "classical minimum-image pairs",
            "epsilon_ev": _LJ_EPSILON,
            "sigma_angstrom": _LJ_SIGMA,
            "switch_distance_angstrom": _LJ_SWITCH,
            "cutoff_angstrom": _LJ_CUTOFF,
            "atom_types": "one shared type for every element",
            "units": "electronvolt-angstrom-dalton-femtosecond",
            "nonclaim": _HAMILTONIAN_NONCLAIM,
        },
        "asymmetric_unit": {
            "setting": 1,
            "species": list(_SPECIES),
            "masses_dalton": list(_MASSES),
            "spin_multiplicities": list(_MULTIPLICITIES),
        },
        "pool": {
            "cell": _CELL.tolist(),
            "maximum_condition_number": _MAXIMUM_CONDITION,
            "copy_basis_transform": _TRANSFORM.tolist(),
            "copy_jitter_angstrom": 0.004,
            "minimum_initial_separation_angstrom": _MINIMUM_SEPARATION,
            "initial_partner_distance_angstrom": _PARTNER_DISTANCE,
            "maximum_site_draws": _MAXIMUM_DRAWS,
        },
        "chart": "AtomisticSymmetryChart(AtomicSymmetryPlan(P1), vary_cell=False)",
        "method": {"kind": "TrustScaledLBFGS", "maximum_step": _LBFGS_MAXIMUM_STEP},
        "convergence": {
            "projected_tolerance": settings.projected_tolerance,
            "maximum_evaluations": settings.maximum_evaluations,
            "convergence_id": _convergence(settings).convergence_id,
        },
        "maximum_rounds": settings.maximum_rounds,
    }


def _json_float(value: float, /) -> float | str:
    """Finite floats as numbers; a non-finite residual keeps its IEEE name."""
    return value if math.isfinite(value) else repr(value)


def _candidate_record(
    candidate: CrystalRankingCandidate,
    item: _Seed,
    relaxation_input: CrystalRelaxationInput,
    /,
) -> dict[str, Any]:
    evidence = candidate.relaxation
    if evidence is None:
        raise RuntimeError(
            f"Relaxed candidate {candidate.source_structure_id} carries no evidence."
        )
    initial = _minimum_image_distances(item.fractional, item.vectors)
    upper = initial[np.triu_indices(len(_SPECIES), k=1)]
    structure = candidate.structure
    step = np.asarray(structure.fractional, dtype=np.float64) - np.asarray(
        jax.device_get(relaxation_input.expansion.fractional), dtype=np.float64
    )
    step = (step - np.round(step)) @ np.asarray(structure.vectors, dtype=np.float64)
    return {
        "source_structure_id": candidate.source_structure_id,
        "ordinal": candidate.ordinal,
        "initial_minimum_separation_angstrom": float(upper.min()),
        "initial_interacting_pairs": int(np.count_nonzero(upper < _LJ_CUTOFF)),
        "maximum_displacement_angstrom": float(np.linalg.norm(step, axis=-1).max()),
        "energy": candidate.energy,
        "status": evidence.status,
        "converged": evidence.converged,
        "full_converged": evidence.full_converged,
        "evaluation_successful": evidence.evaluation_successful,
        "evaluations": evidence.evaluations,
        "projected_residual": _json_float(evidence.projected_residual),
        "full_force_residual": _json_float(evidence.full_force_residual),
        "full_stress_residual": (
            None
            if evidence.full_stress_residual is None
            else _json_float(evidence.full_stress_residual)
        ),
        "symmetry_deviation": _json_float(evidence.symmetry_deviation),
    }


def _exclusion_reasons(ranking: CrystalRanking, /) -> dict[str, int]:
    return dict(
        Counter(
            "duplicate-of" if item.reason.startswith("duplicate-of:") else item.reason
            for item in ranking.excluded
        )
    )


def _kernel_evidence(
    policy: RadialScreenPolicy, chunk: int, pair_chunk: int, /
) -> dict[str, Any]:
    atoms, images, capacity = policy.max_atoms, policy.max_images, policy.max_neighbors
    fingerprint_arguments = (
        jnp.zeros((chunk, atoms, 3), dtype=jnp.float64),
        jnp.zeros((chunk, atoms), dtype=jnp.bool_),
        jnp.zeros((chunk, atoms), dtype=jnp.int32),
        jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (chunk, 3, 3)),
        jnp.zeros((chunk, images, 3), dtype=jnp.float64),
        jnp.zeros((chunk, images), dtype=jnp.bool_),
    )
    pool = (
        jnp.zeros((1, atoms, capacity), dtype=jnp.float64),
        jnp.zeros((1, atoms, capacity), dtype=jnp.int32),
        jnp.zeros((1, atoms), dtype=jnp.int32),
        jnp.zeros((1, atoms), dtype=jnp.bool_),
    )
    lanes = jnp.zeros((pair_chunk,), dtype=jnp.int32)
    kernels = {
        "fingerprint_chunk": partial(
            _fingerprint_chunk.lower,
            *fingerprint_arguments,
            cutoff=policy.cutoff,
            capacity=capacity,
            typed=policy.mode == "neighbor-typed",
        ),
        "score_pairs": partial(
            _score_pairs.lower,
            pool,
            pool,
            lanes,
            lanes,
            cutoff=policy.cutoff,
            mode=policy.mode,
        ),
    }
    records = {}
    for name, lower in kernels.items():
        compiled, phases = measure_lower_and_compile(lower, methodcaller("compile"))
        cost = compiled.cost_analysis()
        cost = cost[0] if isinstance(cost, list) else cost
        records[name] = {
            "phases": asdict(phases),
            "compiler": asdict(
                compiler_evidence(cost, compiled.memory_analysis(), source="xla")
            ),
        }
    return records


def _commit_bytes(root: Path, /) -> int:
    return sum(path.stat().st_size for path in (root / "commits").iterdir())


def _relax(
    pool: list[_Seed], settings: _Relaxation, /
) -> tuple[tuple[CrystalRankingCandidate, ...], dict[str, Any]]:
    """Physical preparation: inputs, prepared program and the batched relaxation."""
    plan = _asymmetric_unit()
    inputs, inputs_seconds = measure_synchronized(
        lambda: [
            _relaxation_input(plan, item, ordinal) for ordinal, item in enumerate(pool)
        ]
    )
    program, program_seconds = measure_synchronized(
        lambda: _program(np.asarray(jax.device_get(inputs[0].expansion.cell.vectors)))
    )
    atoms = np.arange(plan.atom_count)
    candidates, relax_seconds = measure_host(
        lambda: relax_crystal_candidates(
            plan,
            inputs,
            program,
            AtomisticSymmetryChart(AtomicSymmetryPlan(plan), atoms, vary_cell=False),
            system_atoms=atoms,
            method=TrustScaledLBFGS(maximum_step=_LBFGS_MAXIMUM_STEP),
            convergence=_convergence(settings),
            maximum_rounds=settings.maximum_rounds,
        )
    )
    records = [
        _candidate_record(candidate, item, relaxation_input)
        for candidate, item, relaxation_input in zip(
            candidates, pool, inputs, strict=True
        )
    ]
    first = candidates[0]
    return candidates, {
        "inputs": len(inputs),
        "inputs_seconds": inputs_seconds,
        "program_prepare_seconds": program_seconds,
        "relax_seconds_including_compilation": relax_seconds,
        "hamiltonian_id": first.hamiltonian_id,
        "numeric_revision_id": first.numeric_revision_id,
        "program_prepared_id": first.program_prepared_id,
        "energy_unit": first.energy_unit,
        "statuses": dict(Counter(record["status"] for record in records)),
        "converged": sum(record["converged"] for record in records),
        "evaluation_successful": sum(
            record["evaluation_successful"] for record in records
        ),
        "with_energy": sum(record["energy"] is not None for record in records),
        "evaluations": sum(record["evaluations"] for record in records),
        "converged_after_iteration": sum(
            record["converged"] and record["evaluations"] > 1 for record in records
        ),
        "smallest_maximum_displacement_angstrom": min(
            record["maximum_displacement_angstrom"] for record in records
        ),
        "program_logical_bytes": logical_array_bytes(program),
        "candidates": records,
    }


def _case(
    size: int,
    policy: RadialScreenPolicy,
    duplicate_fraction: float,
    chunk: int,
    pair_chunk: int,
    repeats: int,
    seed: int,
    settings: _Relaxation,
    scratch: Path,
    /,
) -> dict[str, Any]:
    pool = _pool(size, duplicate_fraction, seed)
    candidates, relaxation = _relax(pool, settings)
    ranking_policy = CrystalRankingPolicy()
    ranking, admission_seconds = measure_host(
        partial(rank_crystal_candidates, candidates, policy=ranking_policy)
    )
    entries = ranking.entries
    record: dict[str, Any] = {
        "pool": size,
        "duplicates": sum(item.structure_id.startswith("copy-") for item in pool),
        "relaxation": relaxation,
        "admission": {
            "seconds": admission_seconds,
            "qualified": len(entries),
            "qualified_duplicates": sum(
                pool[entry.candidate.ordinal].structure_id.startswith("copy-")
                for entry in entries
            ),
            "excluded": _exclusion_reasons(ranking),
        },
    }
    if not entries:
        skipped = {"skipped": "no candidate passed the ranking admission"}
        record |= {
            "fingerprint": skipped,
            "screen": skipped,
            "certified_bank": skipped,
            "composed": skipped,
            "process_peak_rss_bytes": process_peak_rss_bytes(),
        }
        return record
    structures: list[CrystalStructure] = [entry.candidate.structure for entry in entries]
    build = partial(radial_fingerprints, structures, policy, chunk_size=chunk)
    fingerprints, first_fingerprint = measure_host(build)
    _, fingerprint_warm = measure_repeated(build, warmup=0, repeats=repeats)
    screen = partial(screen_radial, fingerprints, pair_chunk_size=pair_chunk)
    radial, first_screen = measure_host(screen)
    _, screen_warm = measure_repeated(screen, warmup=0, repeats=repeats)
    dedup = DedupPolicy(
        mode="greedy", priority_order="ascending", equivalence=_equivalence()
    )
    submissions = []
    for repeat in range(repeats):
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory) / f"bank-{repeat}"
            bank = create_dedup_bank(root, name="benchmark", policy=dedup, limits=_LIMITS)
            batch = [
                DedupCandidate(
                    entry.candidate.structure,
                    priority=entry.value,
                    ordinal=entry.candidate.ordinal,
                )
                for entry in entries
            ]
            submission, seconds = measure_host(partial(submit_to_dedup_bank, bank, batch))
            submissions.append(
                {
                    "seconds": seconds,
                    "published_bytes": _commit_bytes(root),
                    "certified_comparisons": sum(
                        len(decision.comparisons) for decision in submission.decisions
                    ),
                    "representatives": sum(
                        decision.role == "representative"
                        for decision in submission.decisions
                    ),
                }
            )
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        bank = create_dedup_bank(
            Path(directory) / "composed", name="composed", policy=dedup, limits=_LIMITS
        )
        (outcome, host_peak), composed_seconds = measure_host(
            lambda: host_allocation_peak(
                partial(
                    screen_dedup_rank,
                    candidates,
                    bank,
                    screen=policy,
                    ranking=ranking_policy,
                    fingerprint_chunk_size=chunk,
                    pair_chunk_size=pair_chunk,
                )
            )
        )
    return record | {
        "fingerprint": {
            "first_seconds": first_fingerprint,
            "warm": fingerprint_warm.to_dict(unit="seconds"),
            "logical_bytes": logical_array_bytes(fingerprints),
        },
        "screen": {
            "first_seconds": first_screen,
            "warm": screen_warm.to_dict(unit="seconds"),
            "pairs_scored": radial.pairs_scored,
            "representatives": len(radial.representative_ids),
        },
        "certified_bank": submissions,
        "composed": {
            "seconds": composed_seconds,
            "host_allocation_peak_bytes": host_peak,
            "tracing_included": True,
            "radial_pairs": outcome.screen.pairs_scored,
            "radial_pairs_match_screen_phase": (
                outcome.screen.pairs_scored == radial.pairs_scored
            ),
            "certified_comparisons": outcome.certified_comparisons,
            "ranked": len(outcome.ranking.entries),
            "excluded": _exclusion_reasons(outcome.ranking),
            "screen_misses": len(outcome.screen_misses),
            "unconfirmed_proposals": len(outcome.unconfirmed_proposals),
        },
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--duplicate-fraction", type=float, default=0.25)
    parser.add_argument("--mode", default="neighbor-typed")
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--pair-chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--relax-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--relax-evaluations", type=int, default=3000)
    parser.add_argument("--relax-rounds", type=int, default=3000)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(size < 1 for size in args.pools) or args.repeats < 1:
        raise ValueError("Pool sizes and repeats must be positive.")
    if not 0.0 <= args.duplicate_fraction < 1.0:
        raise ValueError("duplicate-fraction must lie in [0, 1).")
    if (
        not math.isfinite(args.relax_tolerance)
        or args.relax_tolerance <= 0.0
        or args.relax_evaluations < 1
        or args.relax_rounds < 1
    ):
        raise ValueError(
            "The relaxation tolerance, evaluation and round budgets must be positive."
        )
    settings = _Relaxation(
        args.relax_tolerance, args.relax_evaluations, args.relax_rounds
    )
    policy = RadialScreenPolicy(
        cutoff=args.cutoff,
        mode=args.mode,
        include_hydrogens=True,
        threshold=args.threshold,
        max_atoms=4,
        max_images=343,
        max_neighbors=64,
    )
    scratch = args.output.resolve().parent
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    owners = (
        "phydrax/atomistic/crystal/_ranking.py",
        "phydrax/atomistic/crystal/_dedup.py",
        "phydrax/atomistic/crystal/_equivalence.py",
        "phydrax/atomistic/crystal/_asu.py",
        "phydrax/atomistic/_relaxation.py",
        "phydrax/atomistic/_classical.py",
    )
    record = {
        "benchmark": "atomistic-crystal-radial-screen-and-certified-dedup",
        "label": args.label,
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__),
            (
                "cases",
                "environment",
                "kernels",
                "policy",
                "relaxation_definition",
                "source_sha256",
            ),
        ).to_dict(),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in owners
        },
        "policy": {
            "policy_id": policy.policy_id,
            "mode": policy.mode,
            "cutoff": policy.cutoff,
            "threshold": policy.threshold,
            "equivalence_policy_id": _equivalence().policy_id,
            "ranking_policy_id": CrystalRankingPolicy().policy_id,
            "ranking_nonclaim": RANKING_NONCLAIM,
            "chunk": args.chunk,
            "pair_chunk": args.pair_chunk,
        },
        "relaxation_definition": _relaxation_definition(settings),
        "kernels": _kernel_evidence(policy, args.chunk, args.pair_chunk),
        "cases": [
            _case(
                size,
                policy,
                args.duplicate_fraction,
                args.chunk,
                args.pair_chunk,
                args.repeats,
                args.seed,
                settings,
                scratch,
            )
            for size in args.pools
        ],
    }
    write_json_atomic(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
