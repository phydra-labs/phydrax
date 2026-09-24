#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._admissibility import AdmissibilityReason, DOMAIN_REASON_SHIFT


_DIMENSIONLESS = phx.units.parse_unit("1").dimension
_PASCAL = phx.units.parse_unit("Pa").dimension


def _port(name, size, dimension):
    return phx.ValuePort(
        f"test.{name}",
        event_shape=(size,),
        component_ids=tuple(f"{name}[{index}]" for index in range(size)),
        representation="engineering",
        dimensions=(dimension,) * size,
    )


class _Cubic(phx.AbstractArrayModel):
    """Smooth learned response `a * x + b * x**3` with a declared contract."""

    modulus: jax.Array
    stiffening: jax.Array
    precision: phx.ComponentPrecisionContract
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, modulus, stiffening, *, precision=None):
        self.modulus = jnp.asarray(modulus)
        self.stiffening = jnp.asarray(stiffening)
        self.precision = (
            phx.ComponentPrecisionContract.native("float64")
            if precision is None
            else precision
        )
        self.in_size = 1
        self.out_size = 1

    def __call__(self, x, /, *, key=None):
        return self.modulus * x + self.stiffening * x**3

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=self.precision,
            randomness=phx.RandomnessContract("deterministic"),
        )


def _mlp(activation, in_size, out_size):
    return phx.nn.models.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=8,
        depth=1,
        activation=activation,
        key=jr.key(0),
    )


def _stateful_law(model):
    return phx.equations.LearnedConstitutiveModel(
        model,
        _port("strain", 2, _DIMENSIONLESS),
        _port("stress", 2, _PASCAL),
        lower=[-1.0, -1.0, -1.0],
        upper=[1.0, 1.0, 1.0],
        state_port=_port("history", 1, _DIMENSIONLESS),
        model_id="learned-history-law",
    )


def _uniaxial_law(model):
    return phx.equations.LearnedConstitutiveModel(
        model,
        _port("strain", 1, _DIMENSIONLESS),
        _port("stress", 1, _PASCAL),
        lower=[-0.2],
        upper=[0.2],
        model_id="learned-uniaxial-law",
    )


def _parameter_leaves(tree):
    parameters, _, _ = phx.partition_parameters(tree)
    return jax.tree_util.tree_leaves(parameters)


def test_integration_plan_trains_only_the_learned_law():
    fixed = phx.equations.ConstitutiveModel(
        lambda strain, state, parameters, time, dt: phx.equations.ConstitutiveResponse(
            parameters * strain, state
        ),
        state_shape=(1,),
        response_shape=(1,),
        model_id="linear-elastic",
    )
    learned = _uniaxial_law(_Cubic(2.0, 0.5))
    plan = phx.equations.MaterialIntegrationPlan(
        (
            (phx.equations.MaterialSiteId("fixed"), fixed),
            (phx.equations.MaterialSiteId("learned"), learned),
        )
    )

    parameters = _parameter_leaves(plan)
    assert len(parameters) == 2
    assert {float(leaf) for leaf in parameters} == {2.0, 0.5}
    assert phx.equations.AbstractConstitutiveModel.slot_contract().authority is (
        phx.ComponentAuthority.MODEL
    )
    with pytest.raises(TypeError, match="AbstractConstitutiveModel"):
        phx.equations.MaterialIntegrationPlan(
            ((phx.equations.MaterialSiteId("raw"), _Cubic(1.0, 0.0)),)
        )


def test_learned_law_tangent_agrees_with_jvp_and_vjp():
    law = _stateful_law(_mlp(jnp.tanh, 3, 3))
    strain = jnp.asarray([[0.1, -0.2], [0.3, 0.05]])
    history = jnp.asarray([[0.2], [-0.1]])

    def stress(value):
        return law.evaluate(value, history, None, 0.0, 1.0).response

    response = law.evaluate(strain, history, None, 0.0, 1.0)
    assert response.trial_state.shape == history.shape
    assert bool(jnp.all(response.valid))
    assert bool(jnp.all(response.header.eligible))
    tangent = response.consistent_tangent
    assert tangent.shape == (2, 2, 2)

    direction = jnp.asarray([[1.0, 0.5], [-0.25, 2.0]])
    _, forward = jax.jvp(stress, (strain,), (direction,))
    np.testing.assert_allclose(
        forward,
        phx.ein.contract("sij,sj->si", tangent, direction, backend="jax"),
        rtol=1e-12,
    )
    cotangent = jnp.asarray([[0.3, -1.0], [2.0, 0.7]])
    _, pullback = jax.vjp(stress, strain)
    (reverse,) = pullback(cotangent)
    np.testing.assert_allclose(
        reverse,
        phx.ein.contract("sij,si->sj", tangent, cotangent, backend="jax"),
        rtol=1e-12,
    )


def test_out_of_support_site_keeps_primal_but_poisons_its_derivatives():
    law = _uniaxial_law(_Cubic(2.0, 0.5))
    strain = jnp.asarray([[0.1], [0.5]])
    state = jnp.zeros((2, 0))
    response = law.evaluate(strain, state, None, 0.0, 1.0)

    np.testing.assert_allclose(
        response.response[:, 0], 2.0 * strain[:, 0] + 0.5 * strain[:, 0] ** 3
    )
    np.testing.assert_array_equal(response.valid, (True, False))
    np.testing.assert_array_equal(
        response.header.reason_bits, (0, int(AdmissibilityReason.OUTSIDE_SUPPORT))
    )
    assert bool(jnp.isfinite(response.consistent_tangent[0]).all())
    assert bool(jnp.isnan(response.consistent_tangent[1]).all())

    jacobian = jax.jacfwd(
        lambda value: law.evaluate(value, state, None, 0.0, 1.0).response
    )(strain)
    assert bool(jnp.isfinite(jacobian[0]).all())
    assert bool(jnp.isnan(jacobian[1]).all())

    def site_stress(model, site):
        law = _uniaxial_law(model)
        return law.evaluate(strain[site : site + 1], state[:1], None, 0.0, 1.0).response[
            0, 0
        ]

    valid_gradient = eqx.filter_grad(site_stress)(_Cubic(2.0, 0.5), 0)
    invalid_gradient = eqx.filter_grad(site_stress)(_Cubic(2.0, 0.5), 1)
    assert bool(jnp.isfinite(valid_gradient.modulus))
    assert bool(jnp.isnan(invalid_gradient.modulus))


def test_learned_law_admission_requires_c1_regularity_and_declared_units():
    with pytest.raises(ValueError, match="implicit-requires-c1"):
        _uniaxial_law(_mlp(jax.nn.relu, 1, 1))
    with pytest.raises(ValueError, match="regularity-undeclared"):
        _uniaxial_law(_mlp(lambda x: x * jnp.tanh(x), 1, 1))
    unitless = phx.ValuePort(
        "test.stress",
        event_shape=(1,),
        component_ids=("stress[0]",),
        representation="engineering",
    )
    with pytest.raises(ValueError, match="declare component dimensions"):
        phx.equations.LearnedConstitutiveModel(
            _Cubic(1.0, 0.0),
            _port("strain", 1, _DIMENSIONLESS),
            unitless,
            lower=[-1.0],
            upper=[1.0],
            model_id="unitless",
        )


def _inverse_residual(model, state, target):
    return model(state) - target


def _identity_response(model, state, target):
    del model, target
    return phx.equations.ConstitutiveResponse(state, state)


def _root_material(model, **options):
    return phx.equations.fem.LearnedLocalImplicitMaterial(
        model,
        _inverse_residual,
        _identity_response,
        state_shape=(1,),
        model_id="learned-inverse-response",
        **options,
    )


def test_learned_local_root_converges_with_finite_difference_parameter_gradient():
    target = jnp.asarray([0.5])
    material = _root_material(_Cubic(1.0, 2.0))
    result = material.evaluate(jnp.zeros(1), target)

    assert bool(result.valid) and bool(result.diagnostics["converged"])
    root = result.response[0]
    assert abs(float(root + 2.0 * root**3 - 0.5)) <= 1e-10

    def root_of(parameters):
        model = _Cubic(parameters[0], parameters[1])
        return _root_material(model).evaluate(jnp.zeros(1), target).response[0]

    parameters = jnp.asarray([1.0, 2.0])
    gradient = jax.grad(root_of)(parameters)
    step = 1e-6
    finite_difference = jnp.asarray(
        [
            (root_of(parameters + step * basis) - root_of(parameters - step * basis))
            / (2.0 * step)
            for basis in jnp.eye(2)
        ]
    )
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-6)
    assert len(_parameter_leaves(material)) == 2


def test_unresolved_learned_local_root_is_invalid_and_poisons_its_derivative():
    def root_of(parameters):
        model = _Cubic(parameters[0], parameters[1])
        material = _root_material(model, max_steps=1)
        return material.evaluate(jnp.zeros(1), jnp.asarray([0.5]))

    result = root_of(jnp.asarray([1.0, 2.0]))
    assert not bool(result.valid)
    assert int(result.header.reason_bits) & (1 << DOMAIN_REASON_SHIFT)
    assert bool(jnp.isfinite(result.response).all())
    gradient = jax.grad(lambda parameters: root_of(parameters).response[0])(
        jnp.asarray([1.0, 2.0])
    )
    assert bool(jnp.isnan(gradient).all())


def test_learned_local_root_tolerance_respects_the_declared_error_floor():
    coarse = phx.ComponentPrecisionContract(
        input_dtype="float64",
        parameter_dtype="float64",
        compute_dtype="float64",
        accumulation_dtype="float64",
        output_dtype="float64",
        absolute_error_floor=1e-6,
    )
    with pytest.raises(ValueError, match="below the declared component"):
        _root_material(_Cubic(1.0, 2.0, precision=coarse), tolerance=1e-10)
    material = _root_material(_Cubic(1.0, 2.0, precision=coarse), tolerance=1e-6)
    assert bool(material.evaluate(jnp.zeros(1), jnp.asarray([0.5])).valid)
