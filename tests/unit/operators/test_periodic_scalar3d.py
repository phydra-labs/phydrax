import math

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.discretization import PeriodicCell
from phydrax.operators.integral.layer_potential._periodic_core3d import (
    periodic_bloch_phase_3d,
    periodic_lattice_translation_3d,
    periodic_reciprocal_vectors_3d,
    PeriodicEwaldPolicy3D,
    PeriodicScalarResourceError,
)
from phydrax.operators.integral.layer_potential._periodic_helmholtz3d import (
    periodic_helmholtz_green_3d,
    PeriodicHelmholtzWoodAnomalyError,
    prepare_periodic_helmholtz_single_layer_dp0_3d,
)
from phydrax.operators.integral.layer_potential._periodic_laplace3d import (
    PeriodicLaplaceNeutralityError,
    prepare_periodic_laplace_single_layer_dp0_3d,
)
from phydrax.operators.integral.layer_potential._periodic_modified_helmholtz3d import (
    direct_periodic_modified_helmholtz_image_sum_3d,
    periodic_modified_helmholtz_green_3d,
    prepare_periodic_modified_helmholtz_single_layer_dp0_3d,
)


_VERTICES = jnp.asarray(
    [
        [0.9, 0.9, 0.9],
        [1.3, 0.9, 0.9],
        [0.9, 1.3, 0.9],
        [0.9, 0.9, 1.3],
    ]
)
_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


def _cell():
    return PeriodicCell(3.0 * jnp.eye(3))


def _region():
    return phx.geometry.MeshRegion(_VERTICES, _FACES)


def _small_policy(**overrides):
    arguments = dict(
        splitting_parameter=1.4,
        real_cutoff=1,
        reciprocal_cutoff=2,
        exact_image_cutoff=1,
        quadrature_order=2,
        absolute_tolerance=2.0e-3,
        relative_tolerance=2.0e-3,
        max_matrix_entries=64,
        max_preparation_workspace_bytes=16 * 1024 * 1024,
        max_resident_bytes=16 * 1024 * 1024,
    )
    arguments.update(overrides)
    return PeriodicEwaldPolicy3D(**arguments)


def test_affine_lattice_reciprocal_vectors_and_bloch_phase_are_consistent():
    lattice = jnp.asarray([[2.0, 0.2, 0.0], [0.0, 1.7, 0.1], [0.0, 0.0, 1.4]])
    cell = PeriodicCell(lattice, origin=jnp.asarray([-0.1, 0.2, 0.3]))
    assert cell.ambient_dimension == 3
    assert jnp.allclose(
        cell.vectors @ periodic_reciprocal_vectors_3d(cell).T,
        2.0 * jnp.pi * jnp.eye(3),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    alpha = jnp.asarray([0.17, -0.08, 0.11])
    index = jnp.asarray([1, -2, 1])
    assert jnp.allclose(
        periodic_bloch_phase_3d(cell, index, alpha),
        jnp.exp(1j * alpha @ (index @ lattice)),
    )
    partially_periodic = PeriodicCell(jnp.eye(3), periodic_axes=(True, True, False))
    with pytest.raises(ValueError, match="fully periodic 3D"):
        periodic_reciprocal_vectors_3d(partially_periodic)


def test_absolutely_convergent_yukawa_matches_direct_images_and_bloch_character():
    cell = PeriodicCell(2.0 * jnp.eye(3))
    alpha = jnp.asarray([0.19, -0.11, 0.07])
    displacement = jnp.asarray([0.31, -0.27, 0.22])
    policy = PeriodicEwaldPolicy3D(
        splitting_parameter=1.2,
        real_cutoff=5,
        reciprocal_cutoff=6,
        exact_image_cutoff=1,
    )
    ewald = periodic_modified_helmholtz_green_3d(
        displacement,
        cell,
        screening=1.3,
        bloch_wavevector=alpha,
        policy=policy,
    )
    direct = direct_periodic_modified_helmholtz_image_sum_3d(
        displacement,
        cell,
        screening=1.3,
        image_cutoff=7,
        bloch_wavevector=alpha,
    )
    translated = periodic_modified_helmholtz_green_3d(
        displacement + periodic_lattice_translation_3d(cell, jnp.asarray([1, 0, 0])),
        cell,
        screening=1.3,
        bloch_wavevector=alpha,
        policy=policy,
    )
    assert jnp.allclose(ewald, direct, rtol=3.0e-5, atol=3.0e-6)
    assert jnp.allclose(
        translated,
        periodic_bloch_phase_3d(cell, jnp.asarray([1, 0, 0]), alpha) * ewald,
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    with pytest.raises(PeriodicScalarResourceError, match="max_image_count"):
        direct_periodic_modified_helmholtz_image_sum_3d(
            displacement,
            cell,
            screening=1.3,
            image_cutoff=2,
            max_image_count=10,
        )


def test_modified_helmholtz_ewald_split_is_invariant_at_fixed_convergence():
    cell = PeriodicCell(2.0 * jnp.eye(3))
    displacement = jnp.asarray([0.29, 0.18, -0.24])
    alpha = jnp.asarray([0.13, 0.09, -0.05])
    broad_real = PeriodicEwaldPolicy3D(
        splitting_parameter=0.9,
        real_cutoff=5,
        reciprocal_cutoff=5,
        exact_image_cutoff=1,
    )
    broad_reciprocal = PeriodicEwaldPolicy3D(
        splitting_parameter=1.5,
        real_cutoff=4,
        reciprocal_cutoff=7,
        exact_image_cutoff=1,
    )
    first = periodic_modified_helmholtz_green_3d(
        displacement,
        cell,
        screening=0.8,
        bloch_wavevector=alpha,
        policy=broad_real,
    )
    second = periodic_modified_helmholtz_green_3d(
        displacement,
        cell,
        screening=0.8,
        bloch_wavevector=alpha,
        policy=broad_reciprocal,
    )
    assert jnp.allclose(first, second, rtol=3.0e-5, atol=3.0e-6)


def test_laplace_neutrality_gauge_and_fail_closed_allocations():
    cell = _cell()
    policy = _small_policy()
    operator = prepare_periodic_laplace_single_layer_dp0_3d(
        _region(),
        cell,
        certified_fractional_clearance=0.25,
        policy=policy,
    )
    assert operator.require_neutrality
    assert "zero reciprocal mode" in operator.report.gauge
    assert not operator.report.truncation_error_certified
    assert not operator.report.continuum_discretization_error_certified
    with pytest.raises(PeriodicLaplaceNeutralityError, match="zero total DP0 charge"):
        operator.mv(jnp.ones((operator.face_count,)))

    neutral = jnp.asarray([0.4, -0.2, 0.1, 0.0])
    neutral = neutral.at[-1].set(
        -(operator.face_areas[:-1] @ neutral[:-1]) / operator.face_areas[-1]
    )
    assert jnp.all(jnp.isfinite(operator.mv(neutral)))
    target = jnp.asarray([-0.3, 0.2, 0.5, -0.1])
    assert jnp.allclose(
        jnp.sum(target * operator.mv(neutral)),
        jnp.sum(neutral * operator.transpose_mv(target)),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert operator.report.resident_bytes <= policy.max_resident_bytes
    assert (
        operator.report.preparation_workspace_bytes
        <= policy.max_preparation_workspace_bytes
    )

    with pytest.raises(PeriodicScalarResourceError, match="max_matrix_entries"):
        prepare_periodic_laplace_single_layer_dp0_3d(
            _region(),
            cell,
            certified_fractional_clearance=0.25,
            policy=_small_policy(max_matrix_entries=15),
        )


def test_helmholtz_wood_mode_and_unsearched_tail_fail_with_typed_evidence():
    cell = _cell()
    policy = _small_policy(wood_tolerance=1.0e-8)
    with pytest.raises(PeriodicHelmholtzWoodAnomalyError) as caught:
        periodic_helmholtz_green_3d(
            jnp.asarray([0.2, 0.1, -0.15]),
            cell,
            wavenumber=2.0 * math.pi / 3.0,
            policy=policy,
        )
    assert caught.value.minimum_denominator <= caught.value.denominator_tolerance
    assert caught.value.closest_mode_index != (0, 0, 0)

    with pytest.raises(PeriodicHelmholtzWoodAnomalyError, match="unsearched"):
        periodic_helmholtz_green_3d(
            jnp.asarray([0.2, 0.1, -0.15]),
            cell,
            wavenumber=5.0,
            policy=_small_policy(reciprocal_cutoff=1),
        )


def test_all_family_operators_have_exact_algebraic_transposes_and_bounded_actions():
    cell = _cell()
    policy = _small_policy()
    modified = prepare_periodic_modified_helmholtz_single_layer_dp0_3d(
        _region(),
        cell,
        screening=0.7,
        certified_fractional_clearance=0.25,
        bloch_wavevector=jnp.asarray([0.12, -0.05, 0.08]),
        policy=policy,
    )
    laplace = prepare_periodic_laplace_single_layer_dp0_3d(
        _region(),
        cell,
        certified_fractional_clearance=0.25,
        bloch_wavevector=jnp.asarray([0.12, -0.05, 0.08]),
        policy=policy,
    )
    helmholtz = prepare_periodic_helmholtz_single_layer_dp0_3d(
        _region(),
        cell,
        wavenumber=0.8,
        certified_fractional_clearance=0.25,
        bloch_wavevector=jnp.asarray([0.12, -0.05, 0.08]),
        policy=policy,
    )
    x = jnp.asarray([0.3 + 0.1j, -0.2, 0.4 - 0.3j, 0.15])
    y = jnp.asarray([-0.1, 0.25 + 0.2j, 0.3, -0.35j])
    for operator in (modified, laplace, helmholtz):
        forward = operator.mv(x)
        transposed = operator.transpose_mv(y)
        assert jnp.allclose(
            jnp.sum(y * forward),
            jnp.sum(x * transposed),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        assert operator.report.action_workspace_bytes_per_rhs > 0
        assert operator.report.real_image_count == 27
        assert operator.report.reciprocal_mode_count == 125
        assert operator.report.ambient_dimension == 3
        assert operator.cell.cell_id == cell.cell_id
        assert operator.policy.policy_id == policy.policy_id
        assert operator.report.pde
        assert operator.report.geometry
        assert operator.report.formulation
        assert operator.report.precision
        assert operator.report.gauge
        assert "JAX" in operator.report.provider
        assert "vector" in " ".join(operator.report.non_goals)
