from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _grid():
    address = phx.discretization.MortonAddressPlan((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 3)
    coordinates = np.stack(
        np.meshgrid(np.arange(8), np.arange(8), np.arange(8), indexing="ij"),
        axis=-1,
    ).reshape((-1, 3))
    return phx.discretization.SparseVoxelGridPlan(
        address,
        brick_size=2,
        brick_capacity=64,
    ).prepare(coordinates)


def test_voxel_geometry_sampling_downgrades_field_certificate() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.5, feature_id="sampled-sphere"
    ).compile()
    enclosure = phx.geometry.ExactSDFEnclosureCertificate(geometry.field_certificate)
    plan = phx.geometry.VoxelGeometrySamplingPlan(
        _grid(), enclosure=enclosure, narrow_band_width=0.2
    )
    samples = plan.sample(geometry)
    assert bool(samples.evidence.successful)
    assert (
        samples.certificate.zero_set_accuracy is phx.geometry.ZeroSetAccuracy.APPROXIMATE
    )
    assert samples.certificate.sign_reliability is phx.geometry.SignReliability.LOCAL
    assert not samples.certificate.is_signed_distance
    assert int(samples.evidence.certified_sign_samples) > 0
    assert int(jnp.sum(samples.narrow_band)) > 0
    assert bool(
        jnp.all(samples.lower_bounds[samples.sign_certified] > 0.0)
        or jnp.any(samples.upper_bounds[samples.sign_certified] < 0.0)
    )


def test_voxel_geometry_sampling_jits_and_tracks_parameters() -> None:
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.5, feature_id="dynamic-sphere"
    ).compile()
    plan = phx.geometry.VoxelGeometrySamplingPlan(_grid())
    sample = eqx.filter_jit(plan.sample)
    first = sample(geometry)
    radius_index = geometry.schema.index(
        phx.geometry.ParameterId("dynamic-sphere", "radius")
    )
    moved_geometry = eqx.tree_at(
        lambda value: value.state,
        geometry,
        geometry.state.replace_at(radius_index, jnp.asarray(0.6)),
    )
    second = sample(moved_geometry)
    assert bool(first.evidence.successful)
    assert bool(second.evidence.successful)
    assert not bool(jnp.allclose(first.field.values, second.field.values))
