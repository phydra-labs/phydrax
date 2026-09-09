#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.discretization.particle import (
    ParticleInternalGeometry,
    prepare_radial_species_transport,
    RadialShellMeshPlan,
    RadialSpeciesTransportPlan,
)


def _prepared(shell_count=3, species_count=2, *, reference_faces=None):
    mesh = RadialShellMeshPlan(
        ParticleInternalGeometry.SPHERE,
        shell_count,
        reference_faces=reference_faces,
    ).prepare()
    return prepare_radial_species_transport(
        RadialSpeciesTransportPlan(species_count), mesh
    )


def test_nonuniform_spherical_shell_fluxes_are_conservative_and_origin_regular():
    prepared = _prepared(reference_faces=jnp.asarray([0.0, 0.1, 0.45, 1.0]))
    outer_scale = jnp.asarray([2.0])
    metrics = prepared.mesh.metrics(outer_scale)
    concentration = jnp.asarray([[[4.0, 1.0], [2.0, 3.0], [1.0, 2.0]]])
    amounts = concentration * metrics.cell_measures[..., None]
    diffusivity = jnp.asarray([[[2.0, 1.0], [4.0, 3.0], [1.0, 6.0]]])
    outer_flux = jnp.asarray([[0.3, -0.2]])

    result = prepared.evaluate(
        amounts,
        outer_scale=outer_scale,
        storage_measure=metrics.cell_measures,
        cell_diffusivity=diffusivity,
        outer_molar_flux=outer_flux,
        active_mask=jnp.asarray([True]),
    )

    diffusivity_sum = diffusivity[:, :-1, :] + diffusivity[:, 1:, :]
    face_diffusivity = (
        2.0 * diffusivity[:, :-1, :] * diffusivity[:, 1:, :] / diffusivity_sum
    )
    expected_interior_flux = (
        face_diffusivity
        * (concentration[:, :-1, :] - concentration[:, 1:, :])
        / metrics.center_distances[..., None]
    )
    conductance = (
        face_diffusivity
        * metrics.face_measures[:, 1:-1, None]
        / metrics.center_distances[..., None]
    )
    degree = jnp.zeros_like(amounts)
    degree = degree.at[:, :-1, :].add(conductance)
    degree = degree.at[:, 1:, :].add(conductance)
    expected_dt_limit = jnp.min(
        jnp.where(
            degree > 0.0,
            metrics.cell_measures[..., None] / degree,
            jnp.inf,
        ),
        axis=(-2, -1),
    )
    assert jnp.array_equal(result.face_molar_flux[:, 0, :], jnp.zeros((1, 2)))
    assert jnp.allclose(result.face_molar_flux[:, 1:-1, :], expected_interior_flux)
    assert jnp.allclose(result.face_molar_flux[:, -1, :], outer_flux)
    assert jnp.allclose(result.concentrations, concentration)
    assert jnp.allclose(
        result.outer_amount_rate,
        outer_flux * metrics.surface_measure[:, None],
    )
    assert jnp.allclose(
        jnp.sum(result.amount_rate, axis=-2) + result.outer_amount_rate,
        0.0,
        atol=1.0e-12,
    )
    assert jnp.allclose(result.conservation_defect, 0.0, atol=1.0e-12)
    assert jnp.allclose(result.explicit_dt_limit, expected_dt_limit)
    assert jnp.all(result.successful)


def test_runtime_diffusivity_and_inactive_entries_are_masked_without_failure():
    prepared = _prepared()
    outer_scale = jnp.asarray([1.0, 0.0])
    active = jnp.asarray([True, False])
    metrics = prepared.mesh.metrics(jnp.asarray([1.0, 1.0]))
    concentration = jnp.asarray(
        [
            [[3.0, 1.0], [2.0, 2.0], [1.0, 4.0]],
            [[8.0, 7.0], [6.0, 5.0], [4.0, 3.0]],
        ]
    )
    storage = metrics.cell_measures.at[1].set(-1.0)
    amounts = concentration * jnp.where(storage > 0.0, storage, 1.0)[..., None]
    diffusivity = jnp.ones_like(amounts).at[1].set(-2.0)
    outer_flux = jnp.asarray([[0.0, 0.0], [5.0, -4.0]])

    base = prepared.evaluate(
        amounts,
        outer_scale=outer_scale,
        storage_measure=storage,
        cell_diffusivity=diffusivity,
        outer_molar_flux=outer_flux,
        active_mask=active,
    )
    faster = prepared.evaluate(
        amounts,
        outer_scale=outer_scale,
        storage_measure=storage,
        cell_diffusivity=diffusivity.at[0].multiply(2.0),
        outer_molar_flux=outer_flux,
        active_mask=active,
    )
    immobile = prepared.evaluate(
        amounts,
        outer_scale=outer_scale,
        storage_measure=storage,
        cell_diffusivity=jnp.zeros_like(diffusivity),
        outer_molar_flux=jnp.zeros_like(outer_flux),
        active_mask=active,
    )

    assert jnp.allclose(faster.amount_rate[0], 2.0 * base.amount_rate[0])
    assert jnp.array_equal(base.concentrations[1], jnp.zeros((3, 2)))
    assert jnp.array_equal(base.face_molar_flux[1], jnp.zeros((4, 2)))
    assert jnp.array_equal(base.amount_rate[1], jnp.zeros((3, 2)))
    assert jnp.array_equal(base.outer_amount_rate[1], jnp.zeros((2,)))
    assert jnp.array_equal(base.conservation_defect[1], jnp.zeros((2,)))
    assert jnp.isinf(base.explicit_dt_limit[1])
    assert jnp.all(base.successful)
    assert jnp.isinf(immobile.explicit_dt_limit[0])
    assert jnp.all(immobile.successful)


def test_ambiguous_runtime_broadcasting_is_rejected():
    prepared = _prepared()
    amounts = jnp.ones((2, 3, 2))
    with pytest.raises(ValueError, match="cell_diffusivity"):
        prepared.evaluate(
            amounts,
            outer_scale=jnp.ones((2,)),
            storage_measure=jnp.ones((2, 3)),
            cell_diffusivity=jnp.ones((2, 2)),
            outer_molar_flux=jnp.zeros((2, 2)),
            active_mask=jnp.ones((2,), dtype=bool),
        )


def test_radial_species_transport_is_jittable_and_vmappable():
    prepared = _prepared()
    scale = jnp.asarray([0.8, 1.2])
    metrics = prepared.mesh.metrics(scale)
    concentration = jnp.asarray(
        [
            [[1.0, 3.0], [2.0, 2.0], [4.0, 1.0]],
            [[2.0, 1.0], [3.0, 2.0], [5.0, 4.0]],
        ]
    )
    amounts = concentration * metrics.cell_measures[..., None]
    diffusivity = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        ]
    )
    outer_flux = jnp.asarray([[0.1, -0.2], [-0.3, 0.4]])
    active = jnp.asarray([True, True])

    eager = prepared.evaluate(
        amounts,
        outer_scale=scale,
        storage_measure=metrics.cell_measures,
        cell_diffusivity=diffusivity,
        outer_molar_flux=outer_flux,
        active_mask=active,
    )
    compiled = eqx.filter_jit(prepared.evaluate)(
        amounts,
        outer_scale=scale,
        storage_measure=metrics.cell_measures,
        cell_diffusivity=diffusivity,
        outer_molar_flux=outer_flux,
        active_mask=active,
    )
    mapped = jax.vmap(
        lambda amount, radius, storage, coefficient, flux, enabled: prepared.evaluate(
            amount,
            outer_scale=radius,
            storage_measure=storage,
            cell_diffusivity=coefficient,
            outer_molar_flux=flux,
            active_mask=enabled,
        )
    )(
        amounts,
        scale,
        metrics.cell_measures,
        diffusivity,
        outer_flux,
        active,
    )

    assert jnp.allclose(compiled.amount_rate, eager.amount_rate)
    assert jnp.allclose(mapped.amount_rate, eager.amount_rate)
    assert jnp.allclose(mapped.face_molar_flux, eager.face_molar_flux)
    assert jnp.all(mapped.successful)
