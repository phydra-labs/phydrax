# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real native prepare -> fit -> sample -> qualify microbenchmark.

Original analytic four-atom coordinate fixture, not a complete protein residue,
experimental corpus, pretrained predictor, equilibrium ensemble, or accuracy claim.
Run: python -m benchmarks.biophysical_coordinate_generation --steps 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import ValuePort
from phydrax.applications.protein_folding._construct import (
    ProteinAtomKey,
    ProteinConstruct,
)
from phydrax.applications.protein_folding.generation import (
    CoordinateGeometryPolicy,
    fit_coordinate_model,
    prepare_coordinate_sampler,
    prepare_coordinate_training_data,
    prepare_protein_coordinate_support,
    sample_coordinate_proposals,
)
from phydrax.atomistic import (
    AtomisticBatch,
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticScaleContract,
)
from phydrax.atomistic._generation import (
    AtomisticMemoryGenerationSink,
    proposal_address,
    proposal_keys,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM, DIMENSIONLESS, ELECTRONVOLT


CAMPAIGN = "original-analytic-coordinate-benchmark"
CONDITIONING = AtomisticConditioningSpec(
    (
        AtomisticConditionField(
            "analytic_deformation",
            ValuePort(
                "benchmark.coordinate-generation.analytic-deformation",
                event_shape=(),
                component_ids=("value",),
                dimensions=(DIMENSIONLESS,),
                representation="declared-dimensionless-deformation",
                normalization_id="identity",
            ),
        ),
    )
)


def fixture() -> Any:
    construct = ProteinConstruct(("A",), ("A",))
    keys = tuple(
        ProteinAtomKey(construct.residue_keys[0], name) for name in ("CA", "N", "C", "CB")
    )
    ids = (101, 809, 405, 222)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    template = AtomisticBatch(
        np.array([[6, 7, 6, 6]]),
        positions[None],
        np.array([[12.0, 14.0, 12.0, 12.0]]),
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        particle_ids=np.array([ids]),
    )
    geometry = CoordinateGeometryPolicy(
        ((101, 809), (101, 405), (101, 222)),
        ((0.2, 2.0), (0.2, 2.0), (0.2, 2.0)),
        ((101, 809, 405, 222),),
        (1,),
        0.01,
        "original-analytic-four-atom-screen-not-protein-chemistry",
    )
    support = prepare_protein_coordinate_support(
        construct,
        template,
        dict(zip(keys, ids, strict=True)),
        gauge_atom_ids=(101, 809, 405),
        geometry=geometry,
    )
    context = np.linspace(-1.0, 1.0, 20)
    coordinates = np.broadcast_to(positions, (20, 4, 3)).copy()
    coordinates[:, 3, 2] += 0.25 * context
    payload = coordinates.tobytes() + context.tobytes()
    manifest = ReferenceArtifactManifest(
        "original-analytic-coordinate-fixture",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Phydrax-OriginalSyntheticFixture",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted-original-synthetic-data",
        nondimensionalization={"coordinate_angstrom": 1.0},
        uncertainty=None,
        lineage_ids=("analytic-four-atom-deformation",),
    )
    groups = tuple("validation" if i % 4 == 0 else "training" for i in range(20))
    data = prepare_coordinate_training_data(
        support,
        coordinates,
        {"analytic_deformation": context},
        conditioning=CONDITIONING,
        record_ids=tuple(f"original-{i}" for i in range(20)),
        source_manifest_ids=(manifest.manifest_id,) * 20,
        split_group_ids=groups,
        validation_groups=("validation",),
        rights=(manifest,),
        corpus_description=(
            "Original analytic coordinate deformation with distinct held-out "
            "parameter values; numerical ABI only."
        ),
    )
    return data


def run(*, steps: Any = 200, repeats: Any = 3) -> Any:
    data = fixture()
    fit, fit_seconds = measure_synchronized(
        lambda: fit_coordinate_model(
            data,
            key=jr.key(4),
            steps=steps,
            pairs_per_step=32,
            width=32,
            depth=2,
            learning_rate=3e-3,
        )
    )
    sampler = prepare_coordinate_sampler(fit)
    context = jnp.linspace(-0.8, 0.8, 8)
    conditioning = AtomisticConditioning(
        fit.conditioning,
        {"analytic_deformation": context},
        case_count=8,
        atom_capacity=4,
    )
    ordinals = jnp.arange(8, dtype=jnp.int64)
    # The stream's own ordinal-addressed lane keys, so the timed numeric boundary
    # computes exactly the proposals the stream publishes below.
    keys = proposal_keys(
        jr.key(9),
        proposal_address(
            CAMPAIGN, sampler.support.support_id, sampler.sampler_id, role="proposal"
        ),
        ordinals,
    )
    compiled, compilation = measure_lower_and_compile(
        lambda: eqx.filter_jit(sampler.propose).lower(keys, ordinals, conditioning),
        lambda lower: lower.compile(),
    )
    _, timings = measure_repeated(
        lambda: compiled(keys, ordinals, conditioning), warmup=1, repeats=repeats
    )
    batch, stream_seconds = measure_synchronized(
        lambda: sample_coordinate_proposals(
            fit,
            jr.key(9),
            {"analytic_deformation": context},
            campaign_id=CAMPAIGN,
            start_ordinal=0,
            lanes_per_member=8,
            sink=AtomisticMemoryGenerationSink("benchmark-memory"),
        )
    )
    canonical = batch.canonical_positions
    target = np.broadcast_to(np.asarray(data.raw_positions[0]), (8, 4, 3)).copy()
    target[:, 3, 2] = 1.0 + 0.25 * np.asarray(context)
    target, _ = fit.support.canonicalize(jnp.asarray(target))
    error = jnp.sqrt(jnp.mean((canonical - target) ** 2))
    return {
        "claim": "small actually trained analytic coordinate model; no protein-folding or calibrated accuracy claim",
        "qualification_gates": [
            "source-pinned scientific corpus",
            "biologically independent split and chemistry coverage",
            "documented compute campaign",
            "held-out predictive and confidence calibration",
        ],
        "environment": capture_environment().to_dict(),
        "active_atoms": 4,
        "atom_capacity": 4,
        "training_records": len(data.train_indices),
        "validation_records": len(data.validation_indices),
        "training_steps": steps,
        "fit_seconds_including_training_compilation": fit_seconds,
        "sampling_lowering_seconds": compilation.lowering_seconds,
        "sampling_compilation_seconds": compilation.compilation_seconds,
        "sampling_steady_seconds": timings.to_dict(),
        "stream_publication_seconds_including_host_materialization": stream_seconds,
        "model_logical_array_bytes": logical_array_bytes(fit.model),
        "initial_training_loss": fit.initial_training_loss,
        "final_training_loss": fit.final_training_loss,
        "heldout_flow_loss": fit.validation_loss,
        "heldout_sample_coordinate_rmse_angstrom": float(error),
        "accepted_fraction": float(jnp.mean(batch.qualification.accepted)),
        "solver_status": np.asarray(batch.solver_status).tolist(),
        "proposal_status_bits": np.asarray(batch.proposal_status).tolist(),
        "all_samples_retained": len(batch.sample_ids),
        "confidence": "uncalibrated",
        "likelihood": "unavailable",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    print(json.dumps(run(steps=arguments.steps, repeats=arguments.repeats), indent=2))
