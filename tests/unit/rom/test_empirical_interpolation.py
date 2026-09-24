import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import read_array_archive, write_array_archive


def _artifact(basis, *, role="roq"):
    values = jnp.asarray(basis)
    space = phx.linalg.ArraySpace(
        (values.shape[0],),
        dtype=values.dtype,
        space_id=f"{role}-test-space",
    )
    return phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            space,
            values,
            orthonormal=False,
            subspace_id=f"{role}-test-subspace",
        ),
        role=role,
        state_contract_id=f"{role}-test-state",
        support_id=f"{role}-test-support",
        measure_id=f"{role}-test-measure",
        geometry_id=f"{role}-test-geometry",
        source_artifact_ids=(f"{role}-test-source",),
    )


def test_empirical_interpolation_reproduces_source_basis():
    basis = jnp.eye(4)
    artifact = _artifact(basis)
    interpolation = phx.rom.prepare_empirical_interpolation(artifact)
    prepared = interpolation.prepare()
    values = np.asarray([2.0, -1.0, 0.5, 3.0])
    node_values = values[np.asarray(interpolation.node_indices)]

    np.testing.assert_allclose(prepared.interpolate(node_values), values, atol=1e-12)
    assert interpolation.source_artifact_id == artifact.artifact_id
    assert interpolation.support_id == artifact.support_id
    assert interpolation.condition_number <= 1.0e10
    assert interpolation.maximum_reproduction_error <= 1.0e-12


def test_empirical_interpolation_preserves_complex_basis_algebra():
    basis = jnp.asarray(
        [
            [1.0 + 1.0j, 0.5 - 0.5j],
            [0.25j, 1.0 - 0.25j],
            [0.5, -0.75j],
        ]
    )
    coefficients = jnp.asarray([0.5 - 0.2j, -1.0 + 0.3j])
    values = basis @ coefficients
    interpolation = phx.rom.prepare_empirical_interpolation(_artifact(basis))
    prepared = interpolation.prepare()

    np.testing.assert_allclose(
        prepared.interpolate(values[prepared.node_indices]),
        values,
        atol=1e-6,
    )
    assert jnp.issubdtype(prepared.reconstruction_matrix.dtype, jnp.complexfloating)


def test_empirical_interpolation_archive_binds_canonical_numeric_revision(tmp_path):
    interpolation = phx.rom.prepare_empirical_interpolation(_artifact(jnp.eye(4)))
    path = phx.rom.write_empirical_interpolation_artifact(
        tmp_path / "eim.phx", interpolation, analysis_plan_id="eim-analysis"
    )
    restored = phx.rom.read_empirical_interpolation_artifact(path)
    assert restored.artifact_id == interpolation.artifact_id

    manifest, arrays = read_array_archive(path)
    manifest.pop("arrays")
    relabeled = write_array_archive(
        tmp_path / "relabeled.phx",
        manifest={**manifest, "support_id": "another-support"},
        arrays=arrays,
    )
    with pytest.raises(ValueError, match="numeric revision mismatch"):
        phx.rom.read_empirical_interpolation_artifact(relabeled)
    altered = write_array_archive(
        tmp_path / "altered.phx",
        manifest=manifest,
        arrays={**arrays, "interpolation_matrix": 2.0 * arrays["interpolation_matrix"]},
    )
    with pytest.raises(ValueError, match="numeric revision mismatch"):
        phx.rom.read_empirical_interpolation_artifact(altered)
