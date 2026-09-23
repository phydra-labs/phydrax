#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import diagrammatic_field as df


def _phi4_diagram(*, vertex_label="interaction", line_label="loop", order=1):
    phi = df.FieldSpec("phi", statistics="boson", mass_dimension=1.0)
    propagator = df.PropagatorSpec(phi, mass=2.0)
    rule = df.VertexRule(
        "phi4",
        (phi, phi, phi, phi),
        -2.0,
        perturbative_order=order,
    )
    momentum = df.MomentumRoute(jnp.asarray([1.0]), 0.5)
    return df.DiagramGraph(
        (df.VertexInsertion(vertex_label, rule),),
        (
            df.PropagatorLine(
                line_label,
                propagator,
                vertex_label,
                vertex_label,
                df.MomentumRoute(jnp.asarray([0.0]), 0.0),
            ),
        ),
        (
            df.ExternalLeg(
                "incoming",
                phi,
                vertex_label,
                momentum,
                incoming=True,
            ),
            df.ExternalLeg(
                "outgoing",
                phi,
                vertex_label,
                momentum,
                incoming=False,
            ),
        ),
    )


def test_diagram_graph_is_canonical_conserving_and_has_exact_symmetry_factor():
    first = _phi4_diagram()
    relabeled = _phi4_diagram(vertex_label="z", line_label="anything")

    assert first.graph_id == relabeled.graph_id
    assert first.vertices[0].label == "v0"
    assert first.lines[0].label == "p0"
    assert first.evidence.successful
    np.testing.assert_allclose(first.evidence.conservation_residual, 0.0)
    assert first.automorphism_count == 2
    assert first.symmetry_factor == 0.5
    assert first.order == 1
    assert first.loop_order == 1


def test_canonical_graph_identity_is_invariant_to_boson_orientation_and_labels():
    phi = df.FieldSpec("canonical-phi")
    propagator = df.PropagatorSpec(phi, mass=1.0)
    rule = df.VertexRule("canonical-phi3", (phi, phi, phi), 1.0)

    def build(left, right, source, target, internal_momentum):
        return df.DiagramGraph(
            (df.VertexInsertion(left, rule), df.VertexInsertion(right, rule)),
            (
                df.PropagatorLine(
                    "internal",
                    propagator,
                    source,
                    target,
                    df.MomentumRoute([internal_momentum]),
                ),
            ),
            (
                df.ExternalLeg("i1", phi, left, df.MomentumRoute([0.4]), incoming=True),
                df.ExternalLeg("i2", phi, left, df.MomentumRoute([0.6]), incoming=True),
                df.ExternalLeg("o1", phi, right, df.MomentumRoute([0.2]), incoming=False),
                df.ExternalLeg("o2", phi, right, df.MomentumRoute([0.8]), incoming=False),
            ),
        )

    forward = build("left", "right", "left", "right", 1.0)
    reversed_route = build(
        "renamed-left", "renamed-right", "renamed-right", "renamed-left", -1.0
    )

    assert forward.graph_id == reversed_route.graph_id
    assert forward.evidence.successful
    assert reversed_route.evidence.successful


def test_fixed_order_and_skeleton_generation_are_complete_and_duplicate_free():
    reference = _phi4_diagram()
    phi = reference.vertices[0].rule.fields[0]
    propagator = reference.lines[0].propagator
    incoming = df.ExternalState(
        "in",
        phi,
        df.MomentumRoute([1.0], 0.5),
        incoming=True,
    )
    outgoing = df.ExternalState(
        "out",
        phi,
        df.MomentumRoute([1.0], 0.5),
        incoming=False,
    )
    prepared = df.DiagramGeneratorPlan(
        (propagator,),
        (reference.vertices[0].rule,),
        maximum_vertices=1,
        maximum_candidates=128,
    ).prepare((incoming, outgoing))

    fixed = prepared.generate_fixed_order(1)
    skeleton = prepared.generate_skeletons(1)

    assert fixed.evidence.complete
    assert fixed.evidence.diagrams_generated == 1
    assert skeleton.evidence.diagrams_generated == 1
    np.testing.assert_array_equal(fixed.order_histogram(), [0, 1])
    assert fixed.diagrams[0].graph_id == reference.graph_id


def test_diagram_lowering_executes_the_graphir_product_with_complex_sign_evidence():
    diagram = _phi4_diagram()
    prepared = df.DiagramLoweringPlan().prepare(diagram)
    result = prepared.evaluate()

    assert prepared.ir.num_nodes == 7
    assert prepared.ir.num_edges == 6
    np.testing.assert_allclose(result.amplitude, -0.25 + 0.0j)
    np.testing.assert_allclose(result.evidence.phase, -1.0 + 0.0j)
    assert result.evidence.real_sign == -1
    assert result.evidence.successful


def test_closed_fermion_loop_contributes_its_explicit_minus_sign():
    fermion = df.FieldSpec("psi", statistics="fermion", mass_dimension=1.5)
    propagator = df.PropagatorSpec(fermion, mass=1.0)
    rule = df.VertexRule("fermion-bilinear", (fermion, fermion), 1.0)
    diagram = df.DiagramGraph(
        (df.VertexInsertion("insertion", rule),),
        (
            df.PropagatorLine(
                "fermion-loop",
                propagator,
                "insertion",
                "insertion",
                df.MomentumRoute([0.0]),
            ),
        ),
    )
    result = df.DiagramLoweringPlan().prepare(diagram).evaluate()

    assert diagram.evidence.fermion_loop_count == 1
    assert diagram.evidence.fermion_sign == -1
    np.testing.assert_allclose(result.amplitude, -1.0 + 0.0j)
    np.testing.assert_allclose(result.evidence.phase, -1.0 + 0.0j)


def test_bphz_polynomial_subtraction_annihilates_declared_taylor_conditions():
    scheme = df.MomentumSubtractionScheme("MOM", 2.0, [1.0], 1)
    prepared = df.BPHZSubtractionPlan(scheme, (4,)).prepare()
    result = prepared.subtract(jnp.asarray([3.0, 2.0, 1.0, 4.0]))

    assert result.evidence.successful
    np.testing.assert_allclose(result.evidence.subtraction_conditions, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.evaluate(jnp.asarray([1.0])), 0.0, atol=1e-12)
    x = 1.7
    polynomial = 3.0 + 2.0 * x + x**2 + 4.0 * x**3
    value_at_point = 10.0
    derivative_at_point = 16.0
    expected = polynomial - value_at_point - derivative_at_point * (x - 1.0)
    np.testing.assert_allclose(result.evaluate(jnp.asarray([x])), expected, atol=1e-12)
    batched = result.evaluate(jnp.asarray([[x], [2.0 * x]]))
    second_x = 2.0 * x
    second_polynomial = 3.0 + 2.0 * second_x + second_x**2 + 4.0 * second_x**3
    expected_second = (
        second_polynomial - value_at_point - derivative_at_point * (second_x - 1.0)
    )
    np.testing.assert_allclose(
        batched,
        jnp.asarray([expected, expected_second]),
        atol=1e-12,
    )


def test_native_weighted_quadrature_preserves_mass_and_reports_sampling_error():
    prepared = df.DiagramQuadraturePlan(
        jnp.asarray([[0.0], [1.0]]),
        jnp.asarray([1.0, 3.0]),
    ).prepare()
    result = prepared.integrate(jnp.asarray([1.0, 3.0]))

    np.testing.assert_allclose(result.value, 10.0)
    assert result.statistical_uncertainty > 0.0
    assert result.successful


def test_resource_guards_reject_before_dense_or_dag_allocation():
    scheme = df.MomentumSubtractionScheme("guarded-MOM", 1.0, [0.0], 0)
    with pytest.raises(ValueError, match="maximum_matrix_elements"):
        df.BPHZSubtractionPlan(
            scheme,
            (100,),
            maximum_terms=100,
            maximum_matrix_elements=1_000,
        )
    with pytest.raises(ValueError, match="node or edge capacity"):
        df.DiagramLoweringPlan(maximum_nodes=2, maximum_edges=2).prepare(_phi4_diagram())
