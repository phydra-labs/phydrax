#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tiny governed calorimeter corpus through native flow fitting and sampling."""

import hashlib

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def main() -> None:
    calorimetry = phx.applications.detector.calorimetry
    geometry = calorimetry.CalorimeterGeometry(
        cell_ids=jnp.asarray([0, 1]),
        channel_ids=jnp.asarray([0, 1]),
        layer_ids=jnp.asarray([0, 1]),
        subdetector_ids=jnp.zeros(2, dtype=jnp.int32),
        material_ids=jnp.zeros(2, dtype=jnp.int32),
        readout_ids=jnp.zeros(2, dtype=jnp.int32),
        centroids=jnp.asarray([[0.0, 0.0, 1.0], [0.5, 0.0, 2.0]]),
        volumes=jnp.ones(2),
        active=jnp.ones(2, dtype=bool),
        dead=jnp.zeros(2, dtype=bool),
        senders=jnp.asarray([0, 1]),
        receivers=jnp.asarray([1, 0]),
        conditions_id="fast-sim-example",
    )
    payload = b"tiny-calorimeter-flow-corpus"
    manifest = phx.qualification.ReferenceArtifactManifest(
        "tiny-calorimeter-flow-corpus",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="example",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted-example",
        nondimensionalization={"energy": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic-example",),
    )
    showers = jnp.asarray(
        [[0.8, 0.2], [1.4, 0.6], [2.1, 0.9], [2.7, 1.3], [3.2, 1.8], [3.8, 2.2]]
    )
    corpus = calorimetry.prepare_calorimeter_corpus(
        geometry,
        showers,
        jnp.arange(1.0, 7.0)[:, None],
        condition_names=("incident_energy",),
        ledger=jnp.concatenate(
            (jnp.sum(showers, axis=1)[:, None], jnp.zeros((6, 4))), axis=1
        ),
        record_ids=tuple(f"record-{index}" for index in range(6)),
        source_manifest_ids=(manifest.manifest_id,) * 6,
        split_group_ids=(
            "train-a",
            "train-b",
            "train-c",
            "train-d",
            "validation",
            "test",
        ),
        validation_groups=("validation",),
        test_groups=("test",),
        rights=(manifest,),
    )
    fit = calorimetry.fit_calorimeter_flow(
        corpus,
        key=jr.key(1),
        steps=4,
        pairs_per_step=2,
        width=8,
        depth=1,
    )
    sampler = calorimetry.prepare_calorimeter_sampler(
        fit,
        maximum_steps=128,
        maximum_samples=4,
    )
    sample = calorimetry.sample_calorimeter_showers(
        sampler,
        jr.key(2),
        jnp.asarray([[3.0]]),
    )
    if not np.isfinite(
        (fit.initial_training_loss, fit.final_training_loss, fit.validation_loss)
    ).all() or not bool(jnp.all(sample.valid)):
        raise RuntimeError("Calorimeter flow fitting or physical sampling failed.")
    print(
        {
            "initial_loss": fit.initial_training_loss,
            "final_loss": fit.final_training_loss,
            "solver_valid": sample.solver_valid.tolist(),
            "physical_valid": sample.valid.tolist(),
        }
    )


if __name__ == "__main__":
    main()
