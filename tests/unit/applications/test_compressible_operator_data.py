import jax.numpy as jnp

from phydrax.applications.compressible_flow import (
    CompressibleOperatorCase,
    CompressibleOperatorDatasetPlan,
)
from phydrax.nn.operator.training import split_operator_dataset
from phydrax.qualification import ReferenceArtifactManifest


def _manifest():
    return ReferenceArtifactManifest(
        "native-cfd-artifact",
        checksum_algorithm="sha256",
        checksum="1" * 64,
        size_bytes=1,
        license_id="native-test",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty=None,
        lineage_ids=("qualified-cfd",),
    )


def test_compressible_operator_adapter_builds_group_safe_dataset():
    manifest = _manifest()
    cases = tuple(
        CompressibleOperatorCase(
            f"case-{index}",
            f"geometry-{index // 2}",
            "system",
            "method",
            f"family-{index // 2}",
            "qualification",
            manifest,
            0.7 + 0.01 * index,
            1.0e6,
            float(index),
        )
        for index in range(6)
    )
    geometry = jnp.asarray(((0.0, 0.0), (0.5, 0.1), (1.0, 0.0), (0.5, -0.1)))
    query = jnp.stack((jnp.linspace(0.0, 1.0, 5), jnp.zeros((5,))), axis=-1)
    conditions = jnp.asarray(
        tuple(
            (case.mach_number, case.reynolds_number, case.angle_of_attack)
            for case in cases
        )
    )
    pressure = jnp.arange(30, dtype=float).reshape((6, 5))
    dataset = CompressibleOperatorDatasetPlan(
        ("mach_number", "reynolds_number", "angle_of_attack")
    ).build(cases, conditions, geometry, query, {"pressure_coefficient": pressure})
    split = split_operator_dataset(
        dataset,
        train_fraction=1.0 / 3.0,
        validation_fraction=1.0 / 3.0,
        policy=CompressibleOperatorDatasetPlan.default_split_policy(seed=4),
    )

    assert dataset.size == 6
    assert dataset.batch.input("geometry").values.shape == (6, 4, 2)
    assert dataset.targets.fields["pressure_coefficient"].values.shape == (6, 5)
    groups = []
    for partition in (split.train, split.validation, split.test):
        groups.append(
            {record.identities["geometry_id"] for record in partition.provenance}
        )
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
