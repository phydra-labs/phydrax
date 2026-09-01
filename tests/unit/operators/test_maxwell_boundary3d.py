import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.bem._rwg import RWGSurfaceCurrentSpace3D
from phydrax.discretization.bem._surface_complex import (
    OrientedTriangleSurfaceComplex3D,
)
from phydrax.linalg import ArraySpace
from phydrax.operators.integral.layer_potential._maxwell3d import (
    MaxwellEFIEPolicy3D,
    prepare_maxwell_efie_3d,
    prepare_maxwell_electric_field_action_3d,
)
from phydrax.solver._maxwell_boundary import solve_pec_efie_3d


_OCTAHEDRON_VERTICES = jnp.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
_OCTAHEDRON_FACES = jnp.asarray(
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


def _space():
    surface = OrientedTriangleSurfaceComplex3D(_OCTAHEDRON_VERTICES, _OCTAHEDRON_FACES)
    return RWGSurfaceCurrentSpace3D(surface)


def _policy(**kwargs):
    values = dict(
        regular_order=2,
        singular_order=2,
        near_order=2,
        near_ratio=1.0,
        absolute_tolerance=2.0e-3,
        relative_tolerance=2.0e-3,
        max_edges=32,
        max_condition_number=1.0e14,
    )
    values.update(kwargs)
    return MaxwellEFIEPolicy3D(**values)


def _prepared():
    return prepare_maxwell_efie_3d(_space(), 0.7, policy=_policy())


def _torus_surface(major_count=3, minor_count=3):
    vertices = []
    for major in range(major_count):
        u = 2.0 * np.pi * major / major_count
        for minor in range(minor_count):
            v = 2.0 * np.pi * minor / minor_count
            radius = 2.0 + 0.6 * np.cos(v)
            vertices.append([radius * np.cos(u), radius * np.sin(u), 0.6 * np.sin(v)])

    def index(major, minor):
        return (major % major_count) * minor_count + (minor % minor_count)

    faces = []
    for major in range(major_count):
        for minor in range(minor_count):
            a = index(major, minor)
            b = index(major + 1, minor)
            c = index(major + 1, minor + 1)
            d = index(major, minor + 1)
            faces.extend(((a, b, c), (a, c, d)))
    return OrientedTriangleSurfaceComplex3D(
        np.asarray(vertices), np.asarray(faces, dtype=np.int32)
    )


def test_oriented_complex_boundary_and_rwg_divergence_identities():
    space = _space()
    surface = space.surface
    tolerance = 50.0 * jnp.finfo(surface.vertices.dtype).eps

    boundary_one = surface.topology.incidences[0].scipy_boundary()
    boundary_two = surface.topology.incidences[1].scipy_boundary()
    assert (boundary_one @ boundary_two).nnz == 0
    assert surface.topology_report.closed
    assert surface.topology_report.consistently_oriented
    assert surface.topology_report.euler_characteristic == 2
    assert surface.topology_report.genus == 0
    assert surface.topology_report.harmonic_dimension == 0

    integrated_divergence = surface.face_areas @ space.divergence_matrix
    assert jnp.allclose(integrated_divergence, 0.0, atol=tolerance)
    assert jnp.allclose(
        space.divergence_matrix[
            jnp.arange(surface.face_count)[:, None], surface.face_edges
        ],
        surface.face_edge_signs
        * surface.edge_lengths[surface.face_edges]
        / surface.face_areas[:, None],
    )


def test_rwg_is_edge_based_tangential_and_conforming_not_a_scalar_face_space():
    space = _space()
    surface = space.surface
    tolerance = 50.0 * jnp.finfo(surface.vertices.dtype).eps

    assert space.layout.entity_set_id == surface.topology.entities(1).entity_set_id
    assert space.layout.global_dof_count == surface.edge_count == 12
    assert space.size != surface.face_count
    assert space.trace_pairing.conformity.startswith("H(div_Gamma)")
    assert any("BC/RBC" in goal for goal in space.trace_pairing.non_goals)
    normal_components = jnp.sum(
        space.centroid_basis * surface.face_normals[:, None, :], axis=2
    )
    assert jnp.allclose(normal_components, 0.0, atol=tolerance)
    assert jnp.allclose(space.tangential_conformity_defect(), 0.0, atol=tolerance)

    with pytest.raises(TypeError, match="scalar spaces are not accepted"):
        prepare_maxwell_efie_3d(
            ArraySpace((surface.face_count,), dtype=jnp.complex128),
            0.7,
            policy=_policy(),
        )


def test_maxwell_efie_complex_transpose_and_adjoint_are_distinct_and_exact():
    prepared = _prepared()
    operator = prepared.operator
    x = (
        jnp.linspace(0.1, 0.8, operator.source.size)
        + 1j * jnp.linspace(-0.4, 0.3, operator.source.size)
    ).astype(operator.matrix.dtype)
    y = (
        jnp.linspace(-0.7, 0.2, operator.target.size)
        + 1j * jnp.linspace(0.6, -0.1, operator.target.size)
    ).astype(operator.matrix.dtype)

    assert jnp.allclose(operator.transpose_mv(y), operator.matrix.T @ y)
    assert jnp.allclose(operator.adjoint_mv(y), operator.matrix.conj().T @ y)
    assert jnp.allclose(y @ operator.mv(x), x @ operator.transpose_mv(y))
    assert jnp.allclose(jnp.vdot(y, operator.mv(x)), jnp.vdot(operator.adjoint_mv(y), x))
    assert not jnp.allclose(operator.transpose_mv(y), operator.adjoint_mv(y))

    divergence = prepared.current_space.divergence_operator
    face_probe = (
        jnp.linspace(-0.3, 0.5, divergence.target.size)
        + 1j * jnp.linspace(0.2, -0.4, divergence.target.size)
    ).astype(divergence.matrix.dtype)
    assert jnp.allclose(
        divergence.transpose_mv(face_probe), divergence.matrix.T @ face_probe
    )
    assert jnp.allclose(
        divergence.adjoint_mv(face_probe), divergence.matrix.conj().T @ face_probe
    )


def test_off_surface_maxwell_green_dyadic_action_and_duals_are_finite():
    prepared = _prepared()
    action = prepare_maxwell_electric_field_action_3d(
        prepared,
        jnp.asarray([[3.0, 0.5, 0.25], [-2.5, 1.5, 0.75]]),
    )
    coefficients = (
        jnp.linspace(-0.2, 0.4, prepared.current_space.size)
        + 1j * jnp.linspace(0.3, -0.1, prepared.current_space.size)
    ).astype(prepared.operator.matrix.dtype)
    probe = jnp.asarray(
        [[0.2 + 0.1j, -0.3j, 0.4], [-0.1, 0.5 + 0.2j, 0.3j]],
        dtype=prepared.operator.matrix.dtype,
    )

    field = action.electric_field(coefficients)
    assert field.shape == (2, 3)
    assert jnp.all(jnp.isfinite(field))
    assert jnp.allclose(
        probe.reshape(-1) @ field.reshape(-1),
        coefficients @ action.transpose_mv(probe),
    )
    assert jnp.allclose(
        jnp.vdot(probe.reshape(-1), field.reshape(-1)),
        jnp.vdot(action.adjoint_mv(probe), coefficients),
    )
    assert action.report.minimum_distance > 0.0
    assert "on-surface traces" in action.report.non_goals
    assert action.report.dense_bytes == 2 * 3 * prepared.current_space.size * 16
    with pytest.raises(ValueError, match="max_targets"):
        prepare_maxwell_electric_field_action_3d(
            prepared,
            jnp.asarray([[3.0, 0.5, 0.25], [-2.5, 1.5, 0.75]]),
            max_targets=1,
        )


def test_bounded_sphere_like_pec_efie_solve_is_finite():
    prepared = _prepared()
    centers = prepared.current_space.surface.face_centroids
    incident = jnp.exp(1j * prepared.wavenumber * centers[:, 2])[:, None] * jnp.asarray(
        [[1.0, 0.0, 0.0]], dtype=prepared.operator.matrix.dtype
    )
    result = solve_pec_efie_3d(prepared, incident)

    assert bool(result.valid)
    assert result.coefficients.shape == (prepared.current_space.size,)
    assert result.surface_current_at_centroids.shape == (
        prepared.current_space.surface.face_count,
        3,
    )
    assert jnp.all(jnp.isfinite(result.coefficients))
    tolerance = 500.0 * jnp.finfo(result.relative_residual.dtype).eps
    assert result.relative_residual < tolerance
    assert not result.assembly_report.continuum_discretization_error_estimated
    assert "CFIE or interior-resonance removal" in result.assembly_report.non_goals


def test_low_frequency_orientation_and_topology_risks_fail_closed():
    space = _space()
    with pytest.raises(ValueError, match="Low-frequency EFIE rejected"):
        prepare_maxwell_efie_3d(space, 1.0e-3, policy=_policy())

    inconsistent = np.asarray(_OCTAHEDRON_FACES).copy()
    inconsistent[0] = inconsistent[0, ::-1]
    with pytest.raises(ValueError, match="orientations disagree"):
        OrientedTriangleSurfaceComplex3D(_OCTAHEDRON_VERTICES, inconsistent)

    torus = _torus_surface()
    assert torus.topology_report.genus == 1
    assert torus.topology_report.harmonic_dimension == 2
    with pytest.raises(ValueError, match="genus-zero"):
        prepare_maxwell_efie_3d(
            RWGSurfaceCurrentSpace3D(torus), 0.3, policy=_policy(max_edges=64)
        )
