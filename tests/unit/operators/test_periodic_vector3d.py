import math

import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax.discretization import ParticleCell
from phydrax.discretization.bem._rwg import RWGSurfaceCurrentSpace3D
from phydrax.discretization.bem._surface_complex import (
    OrientedTriangleSurfaceComplex3D,
)
from phydrax.operators.integral.layer_potential._periodic_core3d import (
    periodic_bloch_phase_3d,
    periodic_lattice_translation_3d,
    PeriodicEwaldPolicy3D,
)
from phydrax.operators.integral.layer_potential._periodic_helmholtz3d import (
    PeriodicHelmholtzWoodAnomalyError,
)
from phydrax.operators.integral.layer_potential._periodic_vector3d import (
    PeriodicVectorCompatibilityError,
    PeriodicVectorResourceError,
    prepare_periodic_maxwell_electric_field_action_3d,
)
from phydrax.solver._periodic_vector_boundary import (
    periodic_vector_boundary_support_3d,
    PeriodicVectorBoundarySolveUnsupportedError,
    require_periodic_vector_boundary_solve_3d,
)


_VERTICES = jnp.asarray(
    [
        [2.45, 2.0, 2.0],
        [1.55, 2.0, 2.0],
        [2.0, 2.45, 2.0],
        [2.0, 1.55, 2.0],
        [2.0, 2.0, 2.45],
        [2.0, 2.0, 1.55],
    ]
)
_FACES = jnp.asarray(
    [
        [4, 0, 2],
        [4, 2, 1],
        [4, 1, 3],
        [4, 3, 0],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3],
    ],
    dtype=jnp.int32,
)
_TARGETS = jnp.asarray([[3.25, 2.0, 2.1], [2.1, 3.2, 1.9]])
_ALPHA = jnp.asarray([0.13, -0.07, 0.05])


def _cell():
    return ParticleCell(4.0 * jnp.eye(3))


def _space():
    return RWGSurfaceCurrentSpace3D(OrientedTriangleSurfaceComplex3D(_VERTICES, _FACES))


def _policy(**overrides):
    arguments = dict(
        splitting_parameter=1.2,
        real_cutoff=1,
        reciprocal_cutoff=2,
        exact_image_cutoff=1,
        action_block_size=8,
        max_exception_pairs=8192,
        max_matrix_entries=512,
        max_preparation_workspace_bytes=8 * 1024 * 1024,
        max_resident_bytes=8 * 1024 * 1024,
    )
    arguments.update(overrides)
    return PeriodicEwaldPolicy3D(**arguments)


def _action(targets=_TARGETS, **overrides):
    return prepare_periodic_maxwell_electric_field_action_3d(
        _space(),
        targets,
        _cell(),
        wavenumber=0.7,
        wave_impedance=1.3,
        bloch_wavevector=_ALPHA,
        certified_fractional_clearance=0.35,
        policy=_policy(**overrides),
    )


def test_periodic_maxwell_off_surface_action_has_bloch_character_and_evidence():
    action = _action()
    coefficients = (
        jnp.linspace(-0.3, 0.5, action.current_space.size)
        + 1j * jnp.linspace(0.2, -0.4, action.current_space.size)
    ).astype(action.operator.matrix.dtype)
    result = action.evaluate(coefficients)

    assert bool(result.successful)
    assert result.electric_field.shape == (2, 3)
    assert jnp.all(jnp.isfinite(result.electric_field))
    assert not jnp.allclose(result.electric_field, 0.0)
    assert result.support.family.startswith("non-Wood quasi-periodic Maxwell")
    assert result.support.charge_neutrality_enforced
    assert result.support.minimum_target_distance > 0.0
    assert result.support.real_image_count == 27
    assert result.support.reciprocal_mode_count == 125
    assert jnp.isfinite(result.support.dyadic_real_shell_indicator)
    assert jnp.isfinite(result.support.dyadic_reciprocal_shell_indicator)
    assert not result.support.truncation_error_certified
    assert not result.support.continuum_discretization_error_certified

    index = jnp.asarray([1, 0, 0], dtype=jnp.int32)
    translated_targets = _TARGETS + periodic_lattice_translation_3d(_cell(), index)
    translated = _action(translated_targets).electric_field(coefficients)
    phase = periodic_bloch_phase_3d(_cell(), index, _ALPHA)
    assert jnp.allclose(
        translated,
        phase * result.electric_field,
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_periodic_maxwell_complex_transpose_and_adjoint_are_exact():
    action = _action()
    coefficients = (
        jnp.linspace(0.1, 0.7, action.current_space.size)
        + 1j * jnp.linspace(-0.4, 0.2, action.current_space.size)
    ).astype(action.operator.matrix.dtype)
    probe = jnp.asarray(
        [[0.2 + 0.1j, -0.3j, 0.4], [-0.1, 0.5 + 0.2j, 0.3j]],
        dtype=action.operator.matrix.dtype,
    )
    field = action.electric_field(coefficients)

    assert jnp.allclose(
        action.transpose_mv(probe), action.operator.matrix.T @ probe.reshape(-1)
    )
    assert jnp.allclose(
        action.adjoint_mv(probe),
        action.operator.matrix.conj().T @ probe.reshape(-1),
    )
    assert jnp.allclose(
        probe.reshape(-1) @ field.reshape(-1),
        coefficients @ action.transpose_mv(probe),
    )
    assert jnp.allclose(
        jnp.vdot(probe.reshape(-1), field.reshape(-1)),
        jnp.vdot(action.adjoint_mv(probe), coefficients),
    )
    assert not jnp.allclose(action.transpose_mv(probe), action.adjoint_mv(probe))
    assert action.support.exact_transpose
    assert action.support.exact_adjoint


def test_periodic_maxwell_rejects_wood_neutrality_and_resource_violations():
    with pytest.raises(PeriodicHelmholtzWoodAnomalyError):
        prepare_periodic_maxwell_electric_field_action_3d(
            _space(),
            _TARGETS,
            _cell(),
            wavenumber=2.0 * math.pi / 4.0,
            bloch_wavevector=jnp.zeros((3,)),
            certified_fractional_clearance=0.35,
            policy=_policy(wood_tolerance=1.0e-8),
        )

    space = _space()
    incompatible_divergence = space.divergence_matrix.at[0, 0].add(1.0)
    incompatible_space = eqx.tree_at(
        lambda value: value.divergence_matrix,
        space,
        incompatible_divergence,
    )
    with pytest.raises(PeriodicVectorCompatibilityError, match="charge neutral"):
        prepare_periodic_maxwell_electric_field_action_3d(
            incompatible_space,
            _TARGETS,
            _cell(),
            wavenumber=0.7,
            bloch_wavevector=_ALPHA,
            certified_fractional_clearance=0.35,
            policy=_policy(),
        )

    with pytest.raises(PeriodicVectorResourceError, match="max_matrix_entries"):
        _action(max_matrix_entries=71)
    with pytest.raises(
        PeriodicVectorResourceError, match="max_preparation_workspace_bytes"
    ):
        _action(max_preparation_workspace_bytes=1024)


def test_periodic_vector_field_action_never_implies_boundary_solve_support():
    action = _action()
    support = periodic_vector_boundary_support_3d(action)

    assert support.off_surface_field_action_supported
    assert not support.boundary_trace_supported
    assert not support.boundary_self_action_supported
    assert not support.boundary_solve_supported
    assert not support.continuum_certified
    assert "no right-hand-side assembly" in support.non_goals
    assert not action.support.boundary_self_action_supported
    assert not action.support.boundary_solve_supported
    with pytest.raises(
        PeriodicVectorBoundarySolveUnsupportedError,
        match="explicitly absent",
    ):
        require_periodic_vector_boundary_solve_3d(action)
