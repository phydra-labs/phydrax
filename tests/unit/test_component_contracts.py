#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import functools
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax import (
    AbstractArrayModel,
    AbstractComponentSlot,
    admit_randomness,
    bind_component,
    CapabilityEvidenceKind,
    CapabilityRequirement,
    ComponentAuthority,
    ComponentPrecisionContract,
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    ExecutionCapabilities,
    GradientLevel,
    ModelExecutionContract,
    ModelPorts,
    partition_parameters,
    PortMapping,
    RandomnessContract,
    SemanticProvenance,
    supports_derivative,
    ValuePort,
)
from phydrax.nn.activations import (
    activation_regularity,
    AdaptiveActivation,
    squared_relu,
    Stan,
)
from phydrax.nn.models import InputConvexNetwork, MLP


INPUT = DerivativeSurface.INPUT
PARAMETER = DerivativeSurface.MODEL_PARAMETER
CONSTRUCTED = CapabilityEvidenceKind.CONSTRUCTED


def _position(**overrides):
    fields = dict(event_shape=(2,), component_ids=("x", "y"), representation="cartesian")
    fields.update(overrides)
    return ValuePort("space.position", **fields)


def _temperature(**overrides):
    fields = dict(event_shape=(), component_ids=("T",), representation="physical")
    fields.update(overrides)
    return ValuePort("thermal.temperature", **fields)


class _PortedModel(AbstractArrayModel):
    weight: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.weight = jnp.ones((2,))
        self.in_size = 2
        self.out_size = 1

    def __call__(self, x, /, *, key=None):
        return self.weight @ x

    def model_ports(self) -> ModelPorts:
        return ModelPorts(inputs=(_position(),), outputs=(_temperature(),))


class _ConvexClosureSlot(AbstractComponentSlot):
    component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.MODEL
    slot_semantic_id: ClassVar[str] = "test.convex-closure"
    slot_requirements: ClassVar[tuple[CapabilityRequirement, ...]] = (
        CapabilityRequirement("input-convex", [[CONSTRUCTED]], safety_critical=True),
    )


def _cubic(x):
    return x * x * x


def _undeclared_mlp():
    # An activation outside the regularity table leaves the network undeclared.
    return MLP(
        in_size=2,
        out_size=1,
        width_size=4,
        depth=1,
        activation=_cubic,
        key=jax.random.key(0),
    )


def _execution_contract(**overrides):
    fields = dict(
        derivative=DerivativeContract.smooth((INPUT, PARAMETER)),
        execution=ExecutionCapabilities("native-jax"),
    )
    fields.update(overrides)
    return ModelExecutionContract(**fields)


def test_host_only_disables_jit_and_vmap():
    host = ExecutionCapabilities("host-inference", host_only=True)
    assert (host.jit, host.vmap) == (False, False)
    native = ExecutionCapabilities("native-jax")
    assert (native.jit, native.vmap) == (True, True)
    with pytest.raises(ValueError, match="neither jit nor vmap"):
        ExecutionCapabilities("compiled-inference", host_only=True, jit=True)
    with pytest.raises(ValueError, match="never host-only"):
        ExecutionCapabilities("native-jax", host_only=True)
    with pytest.raises(ValueError, match="is host-only"):
        ExecutionCapabilities("host-inference")
    with pytest.raises(ValueError, match="Unknown execution tier"):
        ExecutionCapabilities("onnx")


def test_host_only_model_cannot_claim_jax_derivatives():
    host = ExecutionCapabilities("host-inference", host_only=True)
    with pytest.raises(ValueError, match="host-only model"):
        _execution_contract(execution=host)
    stopped = DerivativeContract.smooth((INPUT,), route=DerivativeRoute.STOPPED)
    assert _execution_contract(derivative=stopped, execution=host).execution.host_only
    adjoint = DerivativeContract.smooth((INPUT,), route=DerivativeRoute.EXTERNAL_ADJOINT)
    with pytest.raises(ValueError, match="'external-adjoint' tier"):
        _execution_contract(derivative=adjoint)


@pytest.mark.parametrize(
    ("route", "forward", "reverse"),
    [
        (DerivativeRoute.DIRECT, True, True),
        (DerivativeRoute.IMPLICIT, True, True),
        (DerivativeRoute.EXTERNAL_ADJOINT, False, True),
        (DerivativeRoute.STOPPED, False, False),
    ],
)
def test_supports_derivative_follows_contract_route(route, forward, reverse):
    contract = DerivativeContract.smooth((INPUT, PARAMETER), route=route)
    request = DifferentiationRequest((PARAMETER,))
    assert supports_derivative(contract, request, mode="forward") is forward
    assert supports_derivative(contract, request, mode="reverse") is reverse


def test_supports_derivative_follows_admission():
    contract = DerivativeContract.smooth((PARAMETER,))
    assert not supports_derivative(
        contract, DifferentiationRequest((INPUT,)), mode="reverse"
    )
    undeclared = _undeclared_mlp().model_execution_contract().derivative
    eager = DifferentiationRequest((INPUT,))
    physical = DifferentiationRequest((INPUT,), authority=ComponentAuthority.MODEL)
    assert supports_derivative(undeclared, eager, mode="forward")
    assert not supports_derivative(undeclared, physical, mode="forward")
    with pytest.raises(ValueError, match="Unknown derivative mode"):
        supports_derivative(contract, eager, mode="jvp")


_ADMIT = dict(
    implicit=False,
    authoritative=False,
    realization_bound=False,
    inference_state_bound=False,
)


@pytest.mark.parametrize(
    ("contract", "flags", "expected"),
    [
        (None, {}, (True, "randomness-undeclared")),
        (None, {"authoritative": True}, (False, "randomness-undeclared")),
        (None, {"implicit": True}, (False, "randomness-undeclared")),
        (
            RandomnessContract("deterministic"),
            {"implicit": True},
            (True, "deterministic"),
        ),
        (
            RandomnessContract("fixed-realization", realization_id="mask-0"),
            {"implicit": True},
            (False, "realization-unbound"),
        ),
        (
            RandomnessContract("fixed-realization"),
            {"realization_bound": True},
            (False, "realization-unidentified"),
        ),
        (
            RandomnessContract("fixed-realization", realization_id="mask-0"),
            {"implicit": True, "realization_bound": True},
            (True, "fixed-realization"),
        ),
        (RandomnessContract("resampled"), {}, (True, "resampled")),
        (
            RandomnessContract("resampled"),
            {"authoritative": True},
            (False, "resampled-randomness-not-admitted"),
        ),
        (
            RandomnessContract("resampled"),
            {"implicit": True},
            (False, "resampled-randomness-not-admitted"),
        ),
        (
            RandomnessContract("deterministic", requires_inference_state=True),
            {},
            (False, "inference-state-unbound"),
        ),
        (
            RandomnessContract("deterministic", requires_inference_state=True),
            {"implicit": True, "inference_state_bound": True},
            (True, "deterministic"),
        ),
    ],
)
def test_randomness_admission(contract, flags, expected):
    assert admit_randomness(contract, **{**_ADMIT, **flags}) == expected


def test_randomness_realization_only_for_fixed_realization():
    with pytest.raises(ValueError, match="fixed-realization"):
        RandomnessContract("resampled", realization_id="mask-0")


def test_precision_floors_declared_and_undeclared():
    native = ComponentPrecisionContract.native(jnp.float32)
    assert native.compute_dtype == "float32"
    assert native.residual_floor(10.0) is None
    declared = ComponentPrecisionContract(
        input_dtype=jnp.float64,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        accumulation_dtype=jnp.float32,
        output_dtype=jnp.float64,
        absolute_error_floor=1e-6,
        relative_error_floor=1e-5,
        amplification_evidence=("lipschitz-bound",),
    )
    assert declared.compute_dtype == "bfloat16"
    assert declared.residual_floor(100.0) == pytest.approx(1e-3)
    assert declared.residual_floor(0.01) == pytest.approx(1e-6)
    relative = ComponentPrecisionContract(
        input_dtype="float32",
        parameter_dtype="float32",
        compute_dtype="float32",
        accumulation_dtype="float32",
        output_dtype="float32",
        relative_error_floor=1e-4,
    )
    assert relative.residual_floor(2.0) == pytest.approx(2e-4)
    with pytest.raises(TypeError, match="explicit"):
        ComponentPrecisionContract.native(float)
    with pytest.raises(ValueError, match="canonical"):
        ComponentPrecisionContract.native("float")
    with pytest.raises(ValueError, match="non-negative"):
        native.residual_floor(-1.0)


def test_model_execution_contract_fingerprint_is_deterministic():
    certificate = InputConvexNetwork(
        in_size=2, width_size=4, depth=2, key=jax.random.key(1)
    ).input_convex_certificate()
    record = ("verified-bound", "check-7", CapabilityEvidenceKind.RUNTIME_CHECKED)

    def build(certificates, precision=None):
        return _execution_contract(
            precision=precision,
            randomness=RandomnessContract("deterministic"),
            ports=ModelPorts(inputs=(_position(),), outputs=(_temperature(),)),
            certificates=certificates,
            semantic_provenance=SemanticProvenance({"family": "test"}),
        )

    first = build((certificate, record))
    assert build((record, certificate)).contract_id == first.contract_id
    assert first.evidence == (
        ("input-convex", CONSTRUCTED),
        ("verified-bound", CapabilityEvidenceKind.RUNTIME_CHECKED),
    )
    changed = build(
        (certificate, record), precision=ComponentPrecisionContract.native(jnp.float32)
    )
    assert changed.contract_id != first.contract_id
    with pytest.raises(ValueError, match="constructed or checked"):
        build((("claim", "c", CapabilityEvidenceKind.DECLARED),))


def test_default_model_execution_contract_is_conservative():
    contract = _undeclared_mlp().model_execution_contract()
    assert contract.regularity is None
    assert contract.precision is None
    assert contract.randomness is None
    assert contract.ports is None
    assert contract.certificates == ()
    assert contract.derivative.route is DerivativeRoute.DIRECT
    for surface in (INPUT, PARAMETER):
        assert contract.derivative.level(surface) is GradientLevel.CONDITIONAL
    assert {entry.conditions for entry in contract.derivative.surfaces} == {
        ("regularity-undeclared",)
    }
    execution = contract.execution
    assert (execution.tier, execution.jit, execution.vmap) == ("native-jax", True, True)


def test_default_contract_reads_ports_and_certificates():
    assert _PortedModel().model_execution_contract().ports == ModelPorts(
        inputs=(_position(),), outputs=(_temperature(),)
    )
    convex = InputConvexNetwork(in_size=2, width_size=4, depth=2, key=jax.random.key(1))
    assert convex.model_execution_contract().evidence == (("input-convex", CONSTRUCTED),)


def test_component_binding_keeps_model_parameters_and_static_authority():
    model = _undeclared_mlp()
    binding = bind_component(model, ComponentAuthority.SURROGATE)
    parameters, model_state, fixed = partition_parameters(binding)
    model_parameters, _, _ = partition_parameters(model)
    assert jax.tree_util.tree_leaves(parameters.model) == jax.tree_util.tree_leaves(
        model_parameters
    )
    assert jax.tree_util.tree_leaves(model_state) == []
    assert jax.tree_util.tree_leaves(binding) == jax.tree_util.tree_leaves(model)
    other = bind_component(model, ComponentAuthority.MODEL)
    assert jax.tree_util.tree_structure(binding) != jax.tree_util.tree_structure(other)
    contract = binding.contract()
    assert contract.authority is ComponentAuthority.SURROGATE
    assert contract.slot_semantic_id is None
    assert contract.port_binding is None


def test_component_binding_resolves_port_mapping():
    model = _PortedModel()
    owner = ModelPorts(
        inputs=(_position(space_id="plate"),), outputs=(_temperature(space_id="plate"),)
    )
    mapping = PortMapping(
        inputs=[(_position().port_id, owner.inputs[0].port_id)],
        outputs=[(_temperature().port_id, owner.outputs[0].port_id)],
    )
    binding = bind_component(
        model, ComponentAuthority.MODEL, port_mapping=mapping, owner_ports=owner
    )
    evidence = binding.contract().port_binding
    assert evidence.inputs == ((_position().port_id, owner.inputs[0].port_id),)
    assert not evidence.spaces_verified

    with pytest.raises(ValueError, match="explicit PortMapping"):
        bind_component(model, ComponentAuthority.MODEL, owner_ports=owner)
    polar = ModelPorts(
        inputs=(_position(representation="polar"),), outputs=(_temperature(),)
    )
    polar_mapping = PortMapping(
        inputs=[(_position().port_id, polar.inputs[0].port_id)],
        outputs=[(_temperature().port_id, polar.outputs[0].port_id)],
    )
    with pytest.raises(ValueError, match="representation mismatch"):
        bind_component(
            model,
            ComponentAuthority.MODEL,
            port_mapping=polar_mapping,
            owner_ports=polar,
        )
    with pytest.raises(ValueError, match="model without ports"):
        bind_component(
            _undeclared_mlp(),
            ComponentAuthority.MODEL,
            port_mapping=mapping,
            owner_ports=owner,
        )


def test_concrete_slot_must_declare_authority_and_identity():
    with pytest.raises(TypeError, match="Abstract"):

        class _Undeclared(AbstractComponentSlot):
            component_authority: ClassVar[ComponentAuthority] = ComponentAuthority.MODEL


def test_slot_requirements_need_constructed_evidence():
    with pytest.raises(ValueError, match="input-convex"):
        bind_component(_undeclared_mlp(), _ConvexClosureSlot)
    convex = InputConvexNetwork(in_size=2, width_size=4, depth=2, key=jax.random.key(1))
    contract = bind_component(convex, _ConvexClosureSlot).contract()
    assert contract.authority is ComponentAuthority.MODEL
    assert contract.slot_semantic_id == "test.convex-closure"
    assert contract.evidence == (("input-convex", CONSTRUCTED),)


def test_binding_admits_requests_with_the_bound_authority():
    binding = bind_component(_undeclared_mlp(), ComponentAuthority.MODEL)
    admission = binding.contract(
        request=DifferentiationRequest((INPUT,), authority=ComponentAuthority.MODEL)
    ).derivative_admission
    assert not admission.supported
    assert admission.reasons == ("regularity-undeclared",)
    with pytest.raises(ValueError, match="bound authority"):
        binding.contract(
            request=DifferentiationRequest(
                (INPUT,), authority=ComponentAuthority.SURROGATE
            )
        )


_C0_LINEAR = DerivativeRegularity.piecewise_polynomial(continuity=0, degree_bound=1)
_C0_SMOOTH = DerivativeRegularity.piecewise_smooth(continuity=0)
_C1_SMOOTH = DerivativeRegularity.piecewise_smooth(continuity=1)


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (jax.nn.identity, DerivativeRegularity.smooth(degree_bound=1)),
        (jax.nn.tanh, DerivativeRegularity.smooth()),
        (jax.nn.swish, DerivativeRegularity.smooth()),
        (jnp.sin, DerivativeRegularity.smooth()),
        (Stan(), DerivativeRegularity.smooth()),
        (AdaptiveActivation(jax.nn.relu), _C0_LINEAR),
        (jax.nn.relu, _C0_LINEAR),
        (jax.nn.hard_tanh, _C0_LINEAR),
        (
            squared_relu,
            DerivativeRegularity.piecewise_polynomial(continuity=1, degree_bound=2),
        ),
        (jax.nn.elu, _C1_SMOOTH),
        (functools.partial(jax.nn.elu, alpha=0.5), _C0_SMOOTH),
        (functools.partial(jax.nn.celu, alpha=0.5), _C1_SMOOTH),
        (jax.nn.selu, _C0_SMOOTH),
        (functools.partial(jax.nn.leaky_relu, negative_slope=0.2), _C0_LINEAR),
        (lambda x: jnp.tanh(x), None),
        (jnp.abs, None),
    ],
)
def test_activation_regularity(fn, expected):
    assert activation_regularity(fn) == expected


def test_activation_regularity_of_modrelu_is_branchwise():
    from phydrax.nn.models import FeynmaNN

    model = FeynmaNN(in_size=2, out_size=1, width_size=4, depth=1, key=jax.random.key(0))
    assert activation_regularity(model.activs[0]) == _C0_SMOOTH
