import jax.numpy as jnp
import pytest

from phydrax.geometry import MeshRegion
from phydrax.linalg import LinearCapabilityError, MaterializationPolicy, materialize
from phydrax.operators.integral.layer_potential._galerkin3d import (
    LaplaceSingleLayerDP0GalerkinPolicy3D,
)
from phydrax.operators.integral.layer_potential._scalar_calderon3d import (
    prepare_scalar_calderon_dp0_3d,
    prepare_scalar_hypersingular_dp0_3d,
    ScalarKernelFamily3D,
)
from phydrax.operators.integral.layer_potential._scalar_trace import (
    SCALAR_TRACE_CONVENTION_3D,
    UnsupportedScalarBoundarySpaceError,
)


_VERTICES = jnp.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


def _policy():
    return LaplaceSingleLayerDP0GalerkinPolicy3D(
        regular_order=3,
        singular_order=3,
        near_order=3,
        near_ratio=1.0,
        absolute_tolerance=5.0e-2,
        relative_tolerance=5.0e-2,
        target_block_size=3,
        source_block_size=2,
    )


def _prepared(kernel=None):
    return prepare_scalar_calderon_dp0_3d(
        MeshRegion(_VERTICES, _FACES), kernel=kernel, policy=_policy()
    )


def test_outward_trace_convention_and_constant_harmonic_jump():
    convention = SCALAR_TRACE_CONVENTION_3D
    assert convention.ambient_dimension == 3
    assert convention.boundary_dimension == 2
    assert convention.normal_orientation == "interior-to-exterior"
    assert convention.double_layer_dirichlet_jump("interior") == -0.5
    assert convention.double_layer_dirichlet_jump("exterior") == 0.5
    assert convention.single_layer_neumann_jump("interior") == 0.5
    assert convention.single_layer_neumann_jump("exterior") == -0.5

    prepared = _prepared()
    constant = jnp.ones((prepared.face_count,), dtype=prepared.space.dtype)
    principal_trace = prepared.double_layer.mv(constant)
    exterior_trace = principal_trace + 0.5 * constant
    interior_trace = principal_trace - 0.5 * constant

    assert jnp.allclose(exterior_trace, 0.0, rtol=1.5e-1, atol=1.5e-1)
    assert jnp.allclose(interior_trace, -1.0, rtol=1.5e-1, atol=1.5e-1)


def test_weak_kprime_is_exact_transpose_and_every_strong_action_transposes():
    prepared = _prepared()
    x = jnp.asarray([0.2, -0.4, 0.7, 0.1], dtype=prepared.space.dtype)
    y = jnp.asarray([-0.3, 0.5, 0.9, -0.2], dtype=prepared.space.dtype)

    assert jnp.allclose(
        x @ prepared.double_layer_weak.mv(y),
        y @ prepared.adjoint_double_layer_weak.mv(x),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    for operator in (
        prepared.single_layer,
        prepared.double_layer,
        prepared.adjoint_double_layer,
    ):
        assert jnp.allclose(
            y @ operator.mv(x),
            x @ operator.transpose_mv(y),
            rtol=1.0e-11,
            atol=1.0e-11,
        )


def test_kernel_metadata_precision_resources_and_complex_helmholtz_actions():
    laplace = _prepared()
    report = laplace.assembly_report
    assert report.pde == "-Delta(u)=0"
    assert report.geometry == (
        "closed-oriented-watertight-piecewise-planar-triangle-mesh-with-"
        "strictly-separated-component-bounding-boxes"
    )
    assert report.formulation == (
        "DP0-Galerkin-V-K-Kprime-with-diagonal-mass-strong-form"
    )
    assert report.provider.startswith("jax-blocked-actions")
    assert report.precision_policy_id
    assert report.resident_bytes > 0
    assert report.action_workspace_bytes_per_rhs > 0
    assert report.quadrature_maximum_errors.shape == (2,)
    assert bool(report.finite & report.accuracy_supported)
    assert not report.materializable
    assert not report.continuum_discretization_error_estimated
    assert not report.hypersingular_supported
    with pytest.raises(LinearCapabilityError):
        materialize(
            laplace.single_layer,
            MaterializationPolicy(max_entries=100, max_bytes=4096),
        )

    helmholtz = _prepared(ScalarKernelFamily3D.outgoing_helmholtz(0.4))
    value = jnp.asarray([1.0, 0.5j, -0.2, 0.3j], dtype=helmholtz.space.dtype)
    image = helmholtz.single_layer.mv(value)
    transposed = helmholtz.single_layer.transpose_mv(value)
    probe = jnp.asarray([-0.2j, 0.4, 0.1j, -0.7], dtype=helmholtz.space.dtype)
    assert jnp.allclose(
        jnp.sum(probe * image),
        jnp.sum(value * helmholtz.single_layer.transpose_mv(probe)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert jnp.allclose(
        jnp.vdot(probe, image),
        jnp.vdot(helmholtz.single_layer.adjoint_mv(probe), value),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert jnp.issubdtype(image.dtype, jnp.complexfloating)
    assert jnp.all(jnp.isfinite(image))
    assert jnp.all(jnp.isfinite(transposed))
    assert helmholtz.kernel.radiation_condition == "outgoing-Sommerfeld"

    screened = _prepared(ScalarKernelFamily3D.modified_helmholtz(0.3))
    assert screened.kernel.pde == "(-Delta+kappa^2)u=0"
    assert jnp.all(jnp.isfinite(screened.single_layer.mv(jnp.ones((4,)))))


def test_hypersingular_open_surface_and_frequency_envelope_fail_closed():
    with pytest.raises(UnsupportedScalarBoundarySpaceError, match=r"H\^1/2"):
        prepare_scalar_hypersingular_dp0_3d(None)

    open_faces = _FACES[:3]
    with pytest.raises(ValueError, match="watertight"):
        prepare_scalar_calderon_dp0_3d(
            MeshRegion(_VERTICES, open_faces), policy=_policy()
        )

    with pytest.raises(ValueError, match="panel-frequency envelope"):
        _prepared(ScalarKernelFamily3D.outgoing_helmholtz(10.0))
