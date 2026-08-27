#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _batch(source, points):
    return phx.nn.operator.OperatorBatch(
        inputs={"input": phx.nn.operator.FunctionSamples(values=source)},
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=points,
            )
        },
    )


def _constraint_operator():
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(3)
    functionals = (
        phx.equations.HolomorphicPointFunctional.value(-1.0),
        phx.equations.HolomorphicPointFunctional.value(1.0),
    )
    return frame, phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        functionals,
    ).prepare()


def test_fixed_target_holomorphic_deeponet_preserves_query_constraints_and_jets():
    frame, operator = _constraint_operator()
    coefficient_map = operator.affine_map(jnp.zeros((2,)))
    trunk = phx.nn.operator.architectures.HolomorphicBasisTrunk(
        frame,
        coefficient_map=coefficient_map,
    )
    branch = phx.nn.models.MLP(
        in_size=3,
        out_size=trunk.latent_size,
        width_size=5,
        depth=1,
        key=jr.key(0),
    )
    deep = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=trunk,
        coord_dim=2,
        latent_size=trunk.latent_size,
        use_bias=False,
    )
    model = phx.nn.operator.architectures.ConditionalHolomorphicDeepONet(deep)
    source = jnp.asarray([0.2, -0.1, 0.3])
    points = jnp.asarray([[-1.0, 0.0], [0.2, -0.15], [1.0, 0.0]])
    values = model((source, points))
    assert jnp.allclose(
        jnp.real(values[jnp.asarray([0, 2])]),
        0.0,
        atol=2e-12,
    )
    assert model.conditional_holomorphic_certificate().trunk_mode == "fixed-target"

    batch = _batch(source, points)
    coordinate = 0.2 - 0.15j
    jet = model.query_jet(batch, coordinate, 2)

    def evaluate(real_coordinates):
        query = jnp.asarray([[real_coordinates[0], real_coordinates[1]]])
        return model((source, query))[0]

    jacobian = jax.jacfwd(evaluate)(jnp.asarray([0.2, -0.15]))
    assert jnp.allclose(jet.derivative(1), jacobian[0], atol=2e-11)
    assert jnp.allclose(jacobian[0] + 1j * jacobian[1], 0.0, atol=2e-11)


def test_variable_target_encoder_preserves_supplied_boundary_values():
    frame, operator = _constraint_operator()
    trunk = phx.nn.operator.architectures.HolomorphicBasisTrunk(
        frame,
        constraint_operator=operator,
    )
    free_model = phx.nn.models.MLP(
        in_size=4,
        out_size=operator.evidence.nullity,
        width_size=5,
        depth=1,
        key=jr.key(1),
    )
    free_encoder = phx.nn.operator.architectures.FixedBranchEncoder(
        free_model,
        operator.evidence.nullity,
    )
    branch = phx.nn.operator.architectures.TargetAugmentedBranchEncoder(
        free_encoder,
        (0, 1),
    )
    deep = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=trunk,
        coord_dim=2,
        latent_size=trunk.latent_size,
        use_bias=False,
    )
    model = phx.nn.operator.architectures.ConditionalHolomorphicDeepONet(deep)
    source = jnp.asarray([0.75, -0.4, 0.2, 0.1])
    points = jnp.asarray([[-1.0, 0.0], [1.0, 0.0]])
    values = model((source, points))
    assert jnp.allclose(jnp.real(values), source[:2], atol=2e-12)
    certificate = model.conditional_holomorphic_certificate()
    assert certificate.trunk_mode == "variable-target"
    assert certificate.coefficient_layout == "target-plus-nullspace"


def test_constrained_holomorphic_deeponet_rejects_free_decoder_bias():
    frame, operator = _constraint_operator()
    coefficient_map = operator.affine_map(jnp.zeros((2,)))
    trunk = phx.nn.operator.architectures.HolomorphicBasisTrunk(
        frame,
        coefficient_map=coefficient_map,
    )
    branch = phx.nn.models.MLP(
        in_size=2,
        out_size=trunk.latent_size,
        width_size=3,
        depth=1,
        key=jr.key(2),
    )
    deep = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=trunk,
        coord_dim=2,
        latent_size=trunk.latent_size,
        use_bias=True,
    )
    with pytest.raises(ValueError, match="free decoder bias"):
        phx.nn.operator.architectures.ConditionalHolomorphicDeepONet(deep)
