#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

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


def _port(name: Any, size: Any, dimension: Any) -> Any:
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

    def __init__(self, modulus: Any, stiffening: Any, *, precision: Any = None) -> None:
        self.modulus = jnp.asarray(modulus)
        self.stiffening = jnp.asarray(stiffening)
        self.precision = (
            phx.ComponentPrecisionContract.native("float64")
            if precision is None
            else precision
        )
        self.in_size = 1
        self.out_size = 1

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        return self.modulus * x + self.stiffening * x**3

    def model_execution_contract(self) -> Any:
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=self.precision,
            randomness=phx.RandomnessContract("deterministic"),
        )


def _mlp(activation: Any, in_size: Any, out_size: Any) -> Any:
    return phx.nn.models.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=8,
        depth=1,
        activation=activation,
        key=jr.key(0),
    )


def _stateful_law(model: Any) -> Any:
    return phx.equations.LearnedConstitutiveModel(
        model,
        _port("strain", 2, _DIMENSIONLESS),
        _port("stress", 2, _PASCAL),
        # ty: ignore[invalid-argument-type]
        lower=[-1.0, -1.0, -1.0],
        # ty: ignore[invalid-argument-type]
        upper=[1.0, 1.0, 1.0],
        state_port=_port("history", 1, _DIMENSIONLESS),
        model_id="learned-history-law",
    )


def _uniaxial_law(model: Any) -> Any:
    return phx.equations.LearnedConstitutiveModel(
        model,
        _port("strain", 1, _DIMENSIONLESS),
        _port("stress", 1, _PASCAL),
        # ty: ignore[invalid-argument-type]
        lower=[-0.2],
        # ty: ignore[invalid-argument-type]
        upper=[0.2],
        model_id="learned-uniaxial-law",
    )


def _parameter_leaves(tree: Any) -> Any:
    parameters, _, _ = phx.partition_parameters(tree)
    return jax.tree_util.tree_leaves(parameters)


def test_learned_constitutive_scenario_1() -> None:
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
            # ty: ignore[invalid-argument-type]
            ((phx.equations.MaterialSiteId("raw"), _Cubic(1.0, 0.0)),)
        )
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
            # ty: ignore[invalid-argument-type]
            lower=[-1.0],
            # ty: ignore[invalid-argument-type]
            upper=[1.0],
            model_id="unitless",
        )
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
    relative = phx.ComponentPrecisionContract(
        input_dtype="float64",
        parameter_dtype="float64",
        compute_dtype="float64",
        accumulation_dtype="float64",
        output_dtype="float64",
        relative_error_floor=1e-3,
    )
    with pytest.raises(ValueError, match="residual_scale"):
        _root_material(_Cubic(1.0, 2.0, precision=relative), tolerance=1e-10)
    with pytest.raises(ValueError, match="below the declared component"):
        _root_material(
            _Cubic(1.0, 2.0, precision=relative), tolerance=1e-10, residual_scale=1.0
        )
    material = _root_material(
        _Cubic(1.0, 2.0, precision=relative), tolerance=1e-3, residual_scale=1.0
    )
    assert material.residual_scale == 1.0
    assert bool(material.evaluate(jnp.zeros(1), jnp.asarray([0.5])).valid)
    ports = phx.ModelPorts(
        inputs=(_port("state", 1, _DIMENSIONLESS),),
        outputs=(_port("response", 1, _PASCAL),),
    )
    model = _PortedCubic(_Cubic(1.0, 2.0), ports)
    with pytest.raises(ValueError, match="local-implicit-material'.*owner_ports"):
        _root_material(model)

    mapping = phx.PortMapping(
        # ty: ignore[invalid-argument-type]
        inputs=[(ports.inputs[0].port_id,) * 2],
        # ty: ignore[invalid-argument-type]
        outputs=[(ports.outputs[0].port_id,) * 2],
    )
    material = _root_material(model, ports=ports, port_mapping=mapping)
    evidence = material.binding.contract().port_binding
    assert evidence.inputs == ((ports.inputs[0].port_id,) * 2,)
    assert evidence.dimensions_verified
    assert bool(material.evaluate(jnp.zeros(1), jnp.asarray([0.5])).valid)


def test_learned_law_tangent_agrees_with_jvp_and_vjp() -> None:
    law = _stateful_law(_mlp(jnp.tanh, 3, 3))
    strain = jnp.asarray([[0.1, -0.2], [0.3, 0.05]])
    history = jnp.asarray([[0.2], [-0.1]])

    def stress(value: Any) -> Any:
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


def test_out_of_support_site_keeps_primal_but_poisons_its_derivatives() -> None:
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

    def site_stress(model: Any, site: Any) -> Any:
        law = _uniaxial_law(model)
        return law.evaluate(strain[site : site + 1], state[:1], None, 0.0, 1.0).response[
            0, 0
        ]

    valid_gradient = eqx.filter_grad(site_stress)(_Cubic(2.0, 0.5), 0)
    invalid_gradient = eqx.filter_grad(site_stress)(_Cubic(2.0, 0.5), 1)
    assert bool(jnp.isfinite(valid_gradient.modulus))
    assert bool(jnp.isnan(invalid_gradient.modulus))


def _inverse_residual(model: Any, state: Any, target: Any) -> Any:
    return model(state) - target


def _identity_response(model: Any, state: Any, target: Any) -> Any:
    del model, target
    return phx.equations.ConstitutiveResponse(state, state)


def _root_material(model: Any, **options: Any) -> Any:
    return phx.equations.fem.LearnedLocalImplicitMaterial(
        model,
        _inverse_residual,
        _identity_response,
        state_shape=(1,),
        model_id="learned-inverse-response",
        **options,
    )


def test_learned_local_root_converges_with_finite_difference_parameter_gradient() -> None:
    target = jnp.asarray([0.5])
    material = _root_material(_Cubic(1.0, 2.0))
    result = material.evaluate(jnp.zeros(1), target)

    assert bool(result.valid) and bool(result.diagnostics["converged"])
    root = result.response[0]
    assert abs(float(root + 2.0 * root**3 - 0.5)) <= 1e-10

    def root_of(parameters: Any) -> Any:
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


def test_unresolved_learned_local_root_is_invalid_and_poisons_its_derivative() -> None:
    def root_of(parameters: Any) -> Any:
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


class _PortedCubic(phx.AbstractArrayModel):
    cubic: _Cubic
    ports: phx.ModelPorts
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, cubic: Any, ports: Any) -> None:
        self.cubic = cubic
        self.ports = ports
        self.in_size = cubic.in_size
        self.out_size = cubic.out_size

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        return self.cubic(x, key=key)

    def model_execution_contract(self) -> Any:
        contract = self.cubic.model_execution_contract()
        return phx.ModelExecutionContract(
            derivative=contract.derivative,
            execution=contract.execution,
            precision=contract.precision,
            randomness=contract.randomness,
            ports=self.ports,
        )

    def model_ports(self) -> Any:
        return self.ports


def test_material_site_identity_uses_cell_rule_and_reference_site() -> None:
    site_id = phx.equations.MaterialSiteId.for_cell_quadrature
    site = site_id(42, "left", domain_id="body", quadrature_rule_id="gauss-2")
    assert (
        site.site_id
        == site_id(42, "left", domain_id="body", quadrature_rule_id="gauss-2").site_id
    )
    alternatives = (
        site_id(43, "left", domain_id="body", quadrature_rule_id="gauss-2"),
        site_id(42, "right", domain_id="body", quadrature_rule_id="gauss-2"),
        site_id(42, "left", domain_id="other", quadrature_rule_id="gauss-2"),
        site_id(42, "left", domain_id="body", quadrature_rule_id="gauss-3"),
    )
    assert all(other.site_id != site.site_id for other in alternatives)
    with pytest.raises(TypeError, match="identity"):
        site_id(True, "left", domain_id="body", quadrature_rule_id="gauss-2")
    with pytest.raises(TypeError, match="identity"):
        site_id(42, 0, domain_id="body", quadrature_rule_id="gauss-2")  # ty: ignore[invalid-argument-type]


def test_material_owner_placement_coverage_and_transactions() -> None:
    MaterialSiteOwnership = phx.equations.MaterialSiteOwnership

    make_site = phx.equations.MaterialSiteId.for_cell_quadrature
    a = make_site(7, "center", domain_id="body", quadrature_rule_id="g2")
    b = make_site(8, "center", domain_id="body", quadrature_rule_id="g2")
    source = MaterialSiteOwnership(((a, 0), (b, 1)), part_count=3)
    target = MaterialSiteOwnership(((b, 0), (a, 1)), part_count=3)
    assert source.owner(a) == 0
    assert target.owner(a) == 1
    assert source.ownership_id != target.ownership_id
    assert (
        source.ownership_id
        == MaterialSiteOwnership(((b, 1), (a, 0)), part_count=3).ownership_id
    )
    assert (
        source.ownership_id
        != MaterialSiteOwnership(((a, 0), (b, 1)), part_count=4).ownership_id
    )
    assert source.for_owner(2) == ()
    with pytest.raises(KeyError):
        source.owner("missing")
    with pytest.raises(ValueError, match="unique"):
        MaterialSiteOwnership(((a, 0), (a, 1)), part_count=3)
    with pytest.raises(ValueError, match="non-negative"):
        MaterialSiteOwnership(((a, True),), part_count=3)
    with pytest.raises(ValueError, match="non-empty"):
        MaterialSiteOwnership((), part_count=3)
    with pytest.raises(ValueError, match="part_count"):
        MaterialSiteOwnership(((a, 3),), part_count=3)
    with pytest.raises(ValueError, match="part_count"):
        source.for_owner(3)
    with pytest.raises(ValueError, match="part_count"):
        phx.equations.MaterialIntegrationPlan((), ownership=source, owner_rank=999)
    with pytest.raises(ValueError, match="part_count"):
        MaterialSiteOwnership(((a, 0),), part_count=0)

    law = phx.equations.ConstitutiveModel(
        lambda strain, state, parameters, time, dt: phx.equations.ConstitutiveResponse(
            state, state
        ),
        state_shape=(1,),
        response_shape=(1,),
        model_id="law",
    )
    plan = phx.equations.MaterialIntegrationPlan(
        ((a, law),), ownership=source, owner_rank=0
    )
    state = plan.initialize((jnp.asarray([1.0]),))
    trial = state.with_trials({a.key: jnp.asarray([2.0])})
    np.testing.assert_array_equal(trial.state(a).committed, [1.0])
    np.testing.assert_array_equal(trial.rollback().state(a).trial, [1.0])
    response = plan.evaluate(a, jnp.asarray([0.0]), trial, None, 0.0, 1.0)
    np.testing.assert_array_equal(response.response, [1.0])
    np.testing.assert_array_equal(response.trial_state, [1.0])
    committed = trial.commit()
    assert committed.state(a).state_version == 1
    np.testing.assert_array_equal(committed.state(a).committed, [2.0])
    restored = plan.restore_payload(plan.checkpoint_payload(trial))
    np.testing.assert_array_equal(restored.state(a).committed, [1.0])
    np.testing.assert_array_equal(restored.state(a).trial, [2.0])
    assert restored.state(a).state_version == 0

    migrated = phx.equations.MaterialTransaction(
        committed.states, ownership=target, owner_rank=1
    )
    destination = phx.equations.MaterialIntegrationPlan(
        ((a, law),), ownership=target, owner_rank=1
    )
    destination.validate(migrated)
    assert migrated.state(a).state_version == 1
    migrated_trial = phx.equations.MaterialTransaction(
        trial.states, ownership=target, owner_rank=1
    )
    destination.validate(migrated_trial)
    np.testing.assert_array_equal(migrated_trial.state(a).committed, [1.0])
    np.testing.assert_array_equal(migrated_trial.state(a).trial, [2.0])
    assert migrated_trial.state(a).state_version == 0
    with pytest.raises(ValueError, match="ownership"):
        destination.validate(committed)
    with pytest.raises(ValueError, match="cover"):
        phx.equations.MaterialIntegrationPlan((), ownership=source, owner_rank=0)
    with pytest.raises(ValueError, match="cover"):
        phx.equations.MaterialIntegrationPlan(((a, law),), ownership=source, owner_rank=1)
    with pytest.raises(ValueError, match="cover"):
        phx.equations.MaterialTransaction(
            (state.state(a),), ownership=source, owner_rank=1
        )
    with pytest.raises(ValueError, match="one or more"):
        phx.equations.MaterialIntegrationPlan(())
    with pytest.raises(TypeError, match="one or more"):
        phx.equations.MaterialTransaction(())

    empty = phx.equations.MaterialIntegrationPlan((), ownership=source, owner_rank=2)
    empty_state = empty.initialize()
    assert empty.evaluate_all((), empty_state, (), 0.0, 1.0) == ()
    assert empty.with_responses(empty_state, ()).states == ()
    assert empty_state.commit().rollback().states == ()
    assert empty.restore_payload(empty.checkpoint_payload(empty_state)).states == ()
    with pytest.raises(ValueError, match="ownership"):
        empty.validate(
            phx.equations.MaterialTransaction((), ownership=target, owner_rank=2)
        )


def test_material_owner_archive_roundtrip_and_rejects_forgery(tmp_path: Any) -> None:
    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
        write_model_artifact,
    )

    site = phx.equations.MaterialSiteId.for_cell_quadrature(
        17, "xi=-sqrt(1/3)", domain_id="body", quadrature_rule_id="gauss-2"
    )
    authority = phx.equations.MaterialSiteOwnership(((site, 0),), part_count=2)
    material = phx.equations.MaterialState(
        site, "law", jnp.asarray([3.0]), trial=jnp.asarray([4.0]), state_version=5
    )
    for rank, states in ((0, (material,)), (1, ())):
        transaction = phx.equations.MaterialTransaction(
            states, ownership=authority, owner_rank=rank
        )
        path = write_model_artifact(
            tmp_path / f"material-{rank}.phx", transaction, limits=None
        )
        restored = read_model_artifact(
            path,
            expected_type=phx.equations.MaterialTransaction,
            restore=restore_model_archive_value,
            limits=None,
        )
        assert restored.layout_id == transaction.layout_id
        assert restored.owner_rank == rank
        restored_site = restored.ownership.site_ids[0]
        assert (
            restored_site.domain_id,
            restored_site.cell_id,
            restored_site.quadrature_rule_id,
            restored_site.reference_site_id,
        ) == ("body", 17, "gauss-2", "xi=-sqrt(1/3)")
        if states:
            assert restored.state(site).state_version == 5
            np.testing.assert_array_equal(restored.state(site).committed, [3.0])
            np.testing.assert_array_equal(restored.state(site).trial, [4.0])
        else:
            assert restored.states == ()
        payload = transaction.checkpoint_payload(plan_id="material-plan")
        payload_path = write_model_artifact(
            tmp_path / f"material-payload-{rank}.phx", payload, limits=None
        )
        restored_payload = read_model_artifact(
            payload_path,
            expected_type=phx.equations.MaterialCheckpointPayload,
            restore=restore_model_archive_value,
            limits=None,
        )
        assert restored_payload.payload_id == payload.payload_id
        restored_state = restored_payload.restore()
        assert restored_state.owner_rank == rank
        assert restored_state.ownership.ownership_id == authority.ownership_id
        if states:
            assert restored_state.state(site).state_version == 5
            np.testing.assert_array_equal(restored_state.state(site).committed, [3.0])
            np.testing.assert_array_equal(restored_state.state(site).trial, [4.0])
        else:
            assert restored_state.states == ()

    forged = phx.equations.MaterialSiteId.for_cell_quadrature(
        17, "xi=-sqrt(1/3)", domain_id="body", quadrature_rule_id="gauss-2"
    )
    object.__setattr__(forged, "cell_id", 18)
    with pytest.raises(ValueError, match="does not match"):
        restore_model_archive_value(forged)
    forged_authority = phx.equations.MaterialSiteOwnership(((site, 0),), part_count=2)
    object.__setattr__(forged_authority, "owners", (1,))
    with pytest.raises(ValueError, match="ownership"):
        restore_model_archive_value(forged_authority)
    payload = phx.equations.MaterialTransaction(
        (material,), ownership=authority, owner_rank=0
    ).checkpoint_payload()
    object.__setattr__(payload.state.states[0], "trial", jnp.asarray([9.0]))
    with pytest.raises(ValueError, match="content identity"):
        payload.restore()


@pytest.mark.parametrize("empty", (False, True))
@pytest.mark.parametrize("field", ("owner_rank", "owners", "part_count", "site_cell"))
@pytest.mark.parametrize("rehash", (False, True))
def test_material_checkpoint_refuses_forged_placement(
    empty: bool, field: str, rehash: bool
) -> None:
    from phydrax.equations._finite_element_material import _material_payload_id

    site = phx.equations.MaterialSiteId.for_cell_quadrature(
        17, "center", domain_id="body", quadrature_rule_id="gauss"
    )
    authority = phx.equations.MaterialSiteOwnership(((site, 0),), part_count=3)
    states = (
        ()
        if empty
        else (
            phx.equations.MaterialState(
                site, "law", jnp.asarray([3.0]), trial=jnp.asarray([4.0]), state_version=5
            ),
        )
    )
    source = phx.equations.MaterialTransaction(
        states, ownership=authority, owner_rank=1 if empty else 0
    )
    payload = source.checkpoint_payload()
    assert payload.state is not source
    assert payload.state.ownership is not None
    assert payload.state.ownership is not source.ownership
    assert payload.state.ownership.site_ids[0] is not site
    if field == "owner_rank":
        object.__setattr__(payload.state, "owner_rank", 0 if empty else 1)
    elif field == "owners":
        object.__setattr__(payload.state.ownership, "owners", (1,))
    elif field == "part_count":
        object.__setattr__(payload.state.ownership, "part_count", 4)
    else:
        object.__setattr__(payload.state.ownership.site_ids[0], "cell_id", 18)
    if rehash:
        object.__setattr__(
            payload, "payload_id", _material_payload_id(payload.state, payload.plan_id)
        )
    with pytest.raises(ValueError):
        payload.restore()


@pytest.mark.parametrize("field", ("site_cell", "model", "version"))
@pytest.mark.parametrize("rehash", (False, True))
def test_material_checkpoint_refuses_forged_state_identity(
    field: str, rehash: bool
) -> None:
    from phydrax.equations._finite_element_material import _material_payload_id

    site = phx.equations.MaterialSiteId.for_cell_quadrature(
        17, "center", domain_id="body", quadrature_rule_id="gauss"
    )
    state = phx.equations.MaterialState(
        site, "law", jnp.asarray([3.0]), trial=jnp.asarray([4.0]), state_version=5
    )
    payload = phx.equations.MaterialTransaction((state,)).checkpoint_payload()
    stored = payload.state.states[0]
    if field == "site_cell":
        object.__setattr__(stored.site_id, "cell_id", 18)
    elif field == "model":
        object.__setattr__(stored, "model_id", "other")
    else:
        object.__setattr__(stored, "state_version", 6)
    if rehash:
        object.__setattr__(
            payload, "payload_id", _material_payload_id(payload.state, payload.plan_id)
        )
    with pytest.raises(ValueError):
        payload.restore()


def test_material_checkpoint_does_not_alias_source_placement() -> None:
    site = phx.equations.MaterialSiteId.for_cell_quadrature(
        17, "center", domain_id="body", quadrature_rule_id="gauss"
    )
    authority = phx.equations.MaterialSiteOwnership(((site, 0),), part_count=2)
    state = phx.equations.MaterialState(
        site, "law", jnp.asarray([3.0]), trial=jnp.asarray([4.0]), state_version=5
    )
    source = phx.equations.MaterialTransaction(
        (state,), ownership=authority, owner_rank=0
    )
    payload = source.checkpoint_payload()
    object.__setattr__(site, "cell_id", 18)
    object.__setattr__(authority, "owners", (1,))
    object.__setattr__(state, "state_version", 42)
    restored = payload.restore()
    assert restored.owner_rank == 0
    assert restored.ownership is not None
    assert restored.ownership.part_count == 2
    assert restored.ownership.owners == (0,)
    assert restored.states[0].site_id.cell_id == 17
    assert restored.states[0].state_version == 5
    np.testing.assert_array_equal(restored.states[0].committed, [3.0])
    np.testing.assert_array_equal(restored.states[0].trial, [4.0])
