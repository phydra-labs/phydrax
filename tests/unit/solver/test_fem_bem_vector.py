import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    LinearSolvePolicy,
    OperatorProperties,
    transpose,
)
from phydrax.operators.integral.layer_potential._elasticity3d import (
    ElasticitySingleLayerDP0Policy3D,
    prepare_elasticity_single_layer_dp0_3d,
)
from phydrax.solver._fem_bem_vector import (
    ElasticityFEMBEMInterfaceQualification3D,
    prepare_elasticity_fem_bem_3d,
    vector_fem_bem_support_report,
)


_VERTICES = jnp.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


@pytest.fixture(scope="module")
def elasticity_bem():
    return prepare_elasticity_single_layer_dp0_3d(
        phx.geometry.MeshRegion(_VERTICES, _FACES),
        shear_modulus=2.0,
        poisson_ratio=0.25,
        policy=ElasticitySingleLayerDP0Policy3D(
            regular_order=3,
            singular_order=3,
            absolute_tolerance=1.0,
            relative_tolerance=1.0,
            max_face_count=8,
            max_matrix_bytes=4096,
            max_preparation_workspace_bytes=1024 * 1024,
        ),
    )


def _qualified_blocks(elasticity_bem, *, orientation="outward-from-fem-interior"):
    boundary = elasticity_bem.weak_operator.source
    interior = ArraySpace(
        (5,),
        dtype=boundary.structure().dtype,
        space_id="synthetic-qualified-vector-h1-interior",
    )
    interior_operator = DenseLinearOperator(
        8.0 * jnp.eye(interior.size, dtype=boundary.structure().dtype),
        source=interior,
        target=interior,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id="synthetic-qualified-symmetric-elasticity-interior",
    )
    trace_matrix = jnp.arange(
        boundary.size * interior.size, dtype=boundary.structure().dtype
    ).reshape((boundary.size, interior.size)) / (50.0 * boundary.size * interior.size)
    trace_operator = DenseLinearOperator(
        trace_matrix,
        source=interior,
        target=boundary,
        operator_id="synthetic-exact-signed-calderon-trace",
    )
    conormal_operator = transpose(trace_operator)
    qualification = ElasticityFEMBEMInterfaceQualification3D(
        "synthetic-matching-tetrahedron-interface",
        interior.space_id,
        boundary.space_id,
        trace_operator.operator_id,
        conormal_operator.operator_id,
        elasticity_bem.weak_operator.operator_id,
        orientation=orientation,
        provider_ids=(
            "synthetic-qualified-vector-H1-provider",
            "exact-signed-interface-map-provider",
        ),
        precision_evidence=(str(boundary.structure().dtype), "synthetic-exact-input"),
        resource_evidence=(
            ("interior_unknowns", interior.size),
            ("interface_unknowns", boundary.size),
        ),
        error_evidence=(
            "synthetic maps are supplied exactly in canonical coordinates",
            "no continuum discretization error estimate",
        ),
    )
    return interior_operator, trace_operator, conormal_operator, qualification


def test_symmetric_elasticity_block_executes_and_solves_manufactured_state(
    elasticity_bem,
):
    blocks = _qualified_blocks(elasticity_bem)
    linear = LinearSolvePolicy(
        DenseLU(),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )
    prepared = prepare_elasticity_fem_bem_3d(
        *blocks[:3], elasticity_bem, blocks[3], linear=linear
    )
    interior_exact = jnp.asarray([0.2, -0.1, 0.3, -0.25, 0.15])
    traction_exact = jnp.linspace(
        -0.15,
        0.2,
        elasticity_bem.weak_operator.source.size,
        dtype=interior_exact.dtype,
    )
    right_hand_side = prepared.operator.mv((interior_exact, traction_exact))

    result = prepared.solve(*right_hand_side)

    assert bool(result.valid)
    assert float(result.relative_block_residual) < 2.0e-5
    assert float(result.symmetry_defect) < 1.0e-6
    assert jnp.allclose(
        result.interior_displacement, interior_exact, rtol=2e-4, atol=2e-5
    )
    assert jnp.allclose(result.boundary_traction, traction_exact, rtol=2e-4, atol=2e-5)
    assert result.normal_convention == "outward-from-fem-interior"
    assert result.continuum_certified is False
    assert "Costabel symmetric" in result.formulation


def test_symmetric_elasticity_block_preserves_exact_transpose(elasticity_bem):
    blocks = _qualified_blocks(elasticity_bem)
    prepared = prepare_elasticity_fem_bem_3d(*blocks[:3], elasticity_bem, blocks[3])
    vector = (
        jnp.linspace(-0.3, 0.4, prepared.interior_operator.source.size),
        jnp.linspace(0.2, -0.1, prepared.bem.weak_operator.source.size),
    )

    forward = prepared.operator.mv(vector)
    transposed = prepared.operator.transpose_mv(vector)

    assert prepared.operator.properties.certifies("self_adjoint")
    assert jnp.allclose(transposed[0], forward[0], rtol=1e-6, atol=1e-7)
    assert jnp.allclose(transposed[1], forward[1], rtol=1e-6, atol=1e-7)


def test_elasticity_coupling_rejects_space_mismatch(elasticity_bem):
    interior_operator, trace_operator, _, qualification = _qualified_blocks(
        elasticity_bem
    )
    boundary = elasticity_bem.weak_operator.source
    wrong_boundary = ArraySpace(
        boundary.structure().shape,
        dtype=boundary.structure().dtype,
        space_id="wrong-interface-space-with-the-same-dimension",
    )
    wrong_trace = DenseLinearOperator(
        trace_operator.matrix,
        source=interior_operator.source,
        target=wrong_boundary,
        operator_id=trace_operator.operator_id,
    )
    wrong_conormal = transpose(wrong_trace)
    wrong_qualification = ElasticityFEMBEMInterfaceQualification3D(
        qualification.interface_id,
        qualification.interior_space_id,
        wrong_boundary.space_id,
        wrong_trace.operator_id,
        wrong_conormal.operator_id,
        qualification.bem_operator_id,
        orientation=qualification.orientation,
        provider_ids=qualification.provider_ids,
        precision_evidence=qualification.precision_evidence,
        resource_evidence=qualification.resource_evidence,
        error_evidence=qualification.error_evidence,
    )

    with pytest.raises(ValueError, match="Boundary space"):
        prepare_elasticity_fem_bem_3d(
            interior_operator,
            wrong_trace,
            wrong_conormal,
            elasticity_bem,
            wrong_qualification,
        )


def test_elasticity_coupling_rejects_orientation_mismatch(elasticity_bem):
    blocks = _qualified_blocks(elasticity_bem, orientation="outward-from-bem-exterior")

    with pytest.raises(ValueError, match="orientation"):
        prepare_elasticity_fem_bem_3d(*blocks[:3], elasticity_bem, blocks[3])


def test_elasticity_coupling_rejects_unpaired_conormal_map(elasticity_bem):
    interior_operator, trace_operator, _, qualification = _qualified_blocks(
        elasticity_bem
    )
    independent_conormal = DenseLinearOperator(
        trace_operator.matrix.T,
        source=trace_operator.target,
        target=trace_operator.source,
        operator_id="independent-conormal-copy",
    )
    independent_qualification = ElasticityFEMBEMInterfaceQualification3D(
        qualification.interface_id,
        qualification.interior_space_id,
        qualification.boundary_space_id,
        qualification.trace_operator_id,
        independent_conormal.operator_id,
        qualification.bem_operator_id,
        orientation=qualification.orientation,
        provider_ids=qualification.provider_ids,
        precision_evidence=qualification.precision_evidence,
        resource_evidence=qualification.resource_evidence,
        error_evidence=qualification.error_evidence,
    )

    with pytest.raises(ValueError, match="exact PHYDRAX algebraic transpose pair"):
        prepare_elasticity_fem_bem_3d(
            interior_operator,
            trace_operator,
            independent_conormal,
            elasticity_bem,
            independent_qualification,
        )


def test_vector_support_report_explicitly_rejects_maxwell_interface():
    report = vector_fem_bem_support_report()

    assert len(report.implemented) == 1
    assert "static isotropic elasticity 3D" in report.implemented[0]
    assert any("Maxwell FEM-BEM is unavailable" in reason for reason in report.rejected)
    assert any("H(curl)-to-RWG" in reason for reason in report.rejected)
    assert report.continuum_certified is False
