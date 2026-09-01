#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _density(fields, geometry, context):
    del geometry, context
    return fields["u"].value


def test_functional_preserves_order_identity_and_signed_terms():
    first = phx.variational.LocalIntegralTerm(
        "body",
        region="domain",
        fields=(phx.variational.FieldJetSpec("u", value=True),),
        density=_density,
        density_id="identity",
    )
    second = phx.variational.LocalIntegralTerm(
        "load",
        region="boundary",
        fields=(phx.variational.FieldJetSpec("u", value=True),),
        density=_density,
        density_id="identity",
        weight=-2.0,
    )
    functional = phx.variational.Functional(
        "energy",
        (first, second),
        variable_fields=("u",),
    )

    assert functional.field_names == ("u",)
    assert functional.region_names == ("domain", "boundary")
    assert functional.terms == (first, second)
    assert first.term_id != second.term_id


def test_functional_rejects_ambiguous_or_unused_declarations():
    with pytest.raises(ValueError, match="value, gradient"):
        phx.variational.FieldJetSpec("u")

    term = phx.variational.LocalIntegralTerm(
        "body",
        region="domain",
        fields=(phx.variational.FieldJetSpec("u", value=True),),
        density=_density,
        density_id="identity",
    )
    with pytest.raises(ValueError, match="unique"):
        phx.variational.Functional(
            "duplicate",
            (term, term),
            variable_fields=("u",),
        )
    with pytest.raises(ValueError, match="missing"):
        phx.variational.Functional(
            "unused",
            (term,),
            variable_fields=("v",),
        )


def test_functional_evaluation_requires_real_scalar_components():
    evaluation = phx.variational.FunctionalEvaluation(
        jnp.asarray(1.0),
        (jnp.asarray(1.0),),
        functional_id="functional",
        binding_id="binding",
    )
    assert evaluation.value.shape == ()

    with pytest.raises(TypeError, match="real"):
        phx.variational.FunctionalEvaluation(
            jnp.asarray(1.0j),
            (jnp.asarray(1.0j),),
            functional_id="functional",
            binding_id="binding",
        )
