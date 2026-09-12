#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.export._discrete_velocity_iree as d2v_iree
from phydrax._identity import SemanticProvenance
from phydrax._model import AbstractArrayModel
from phydrax.backends._types import BackendAvailability, BackendUnavailableError
from phydrax.backends.iree import IREE_CAPABILITIES
from phydrax.closure_data._dataset import NormalizerProvenance, TrainOnlyNormalizer
from phydrax.closure_data._kinetic_equilibrium import (
    energy_equilibrium_numeric_revision,
    EnergyEquilibriumSupportEnvelope,
    LearnedEnergyEquilibriumBindingPlan,
)
from phydrax.closure_data._state import FlowStateSchema
from phydrax.discretization.discrete_velocity._energy_equilibrium import (
    PositiveEnergyEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import d2v17_quadrature
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.export._iree import IREEArtifactManifest, IREEExportPolicy, IREEExportResult


jax.config.update("jax_enable_x64", True)


class _ZeroDualModel(AbstractArrayModel):
    weight: jax.Array
    bias: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, *, offset=0.0):
        self.weight = jnp.zeros((2, 4), dtype=jnp.float64)
        self.bias = jnp.full((2,), offset, dtype=jnp.float64)
        self.in_size = 4
        self.out_size = 2

    def __call__(self, values, /, *, key=None):
        del key
        return self.weight @ values + self.bias


def _runtime_and_binding(*, model_offset=0.0):
    quadrature = d2v17_quadrature()
    material = IdealGasMaterial(1.4, 1.0)
    energy_plan = PositiveEnergyEquilibriumPlan(quadrature)
    method = SmoothCompressibleD2VKineticMethod(
        quadrature,
        material,
        ConstantTransport(0.03, 0.04),
    )
    runtime = PreparedSmoothCompressibleD2V17SpatialDynamics(
        method,
        energy_plan,
        D2V17PeriodicTransportPlan(
            quadrature,
            (5, 6),
            (0.01, 0.01),
            0.01,
        ),
    )
    schema = FlowStateSchema(
        ("density", "momentum_x", "momentum_y", "total_energy"),
        ("kg/m^3", "kg/(m^2*s)", "kg/(m^2*s)", "J/m^3"),
        (1.0, 1.0, 1.0, 1.0),
        density_name="density",
        total_energy_name="total_energy",
    )
    provenance = NormalizerProvenance(
        partition_id="iree-training-partition",
        training_assignment_ids=("assignment",),
        training_sample_ids=("sample",),
        feature_name="conserved-state",
        schema_id=schema.schema_id,
    )
    normalizer = TrainOnlyNormalizer(
        jnp.asarray((1.0, 0.0, 0.0, 1.25), dtype=jnp.float64),
        jnp.ones((4,), dtype=jnp.float64),
        provenance,
        epsilon=1.0e-12,
    )
    support = EnergyEquilibriumSupportEnvelope(
        rho_bounds=(0.5, 2.0),
        u_x_bounds=(-0.5, 0.5),
        u_y_bounds=(-0.5, 0.5),
        temperature_bounds=(0.25, 1.0),
        maximum_mach=1.0,
        minimum_hull_margin=1.0e-8,
        minimum_particle_equilibrium_margin=0.0,
        schema_id=schema.schema_id,
        material_id=material.material_id,
        normalizer_id=normalizer.normalizer_id,
        quadrature_id=quadrature.quadrature_id,
        equilibrium_plan_id=energy_plan.plan_id,
        training_preparation_id="iree-training-preparation",
    )
    semantic = SemanticProvenance(
        {
            "kind": "iree-learned-energy-equilibrium",
            "architecture": "zero-affine-dual",
        },
        resource_ids={
            "material": material.material_id,
            "normalizer": normalizer.normalizer_id,
            "quadrature": quadrature.quadrature_id,
            "support": support.support_id,
        },
    )
    plan = LearnedEnergyEquilibriumBindingPlan(
        energy_plan,
        schema,
        material,
        normalizer,
        support,
        input_component_names=schema.component_names,
        semantic_id=semantic.semantic_id,
        training_preparation_id="iree-training-preparation",
        parent_artifact_id="frozen-training-artifact",
    )
    model = _ZeroDualModel(offset=model_offset)
    binding = plan.prepare(
        model,
        energy_equilibrium_numeric_revision(semantic, model),
    )
    return runtime, binding


def _state(runtime, *, density=1.0):
    conserved = jnp.asarray((density, 0.0, 0.0, 1.25 * density), dtype=jnp.float64)
    equilibrium = runtime.method.equilibrium(conserved)
    shape = runtime.transport.spatial_shape + (17,)
    return SmoothCompressibleKineticState(
        jnp.broadcast_to(equilibrium.particle_populations, shape),
        jnp.broadcast_to(equilibrium.total_energy_populations, shape),
    )


def _available():
    class Available:
        @staticmethod
        def require(capability):
            assert capability == "compiled-inference"

    return Available()


def _install_fake_save(monkeypatch, *, validation_ok=True, target_backend=None):
    calls = []

    def fake_save(function, path, /, **kwargs):
        native = function(*kwargs["inputs"], key=None)
        deployed = tuple(np.asarray(value) for value in native)
        output_names = tuple(kwargs["output_names"])
        policy = kwargs["policy"]
        manifest = IREEArtifactManifest(
            format="phydrax-iree-inference",
            artifact_id="fake-d2v-artifact",
            module_file="module.vmfb",
            module_sha256="0" * 64,
            compiler_version="test-compiler",
            runtime_version="test-runtime",
            target_backend=(
                policy.target_backend if target_backend is None else target_backend
            ),
            runtime_driver=policy.runtime_driver,
            function_name="forward",
            entry_point="main",
            calling_convention_version=10,
            input_names=tuple(kwargs["input_names"]),
            input_shapes=tuple(tuple(value.shape) for value in kwargs["inputs"]),
            input_dtypes=tuple(np.dtype(value.dtype).str for value in kwargs["inputs"]),
            output_names=output_names,
            output_shapes=tuple(tuple(value.shape) for value in deployed),
            output_dtypes=tuple(value.dtype.str for value in deployed),
            vectorized=False,
            has_preprocess=False,
            has_postprocess=False,
            validation_ok=validation_ok,
            maximum_absolute_errors=tuple(0.0 for _ in output_names),
            maximum_relative_errors=tuple(0.0 for _ in output_names),
        )
        calls.append((native, deployed, kwargs))
        return IREEExportResult(Path(path), manifest)

    monkeypatch.setattr(d2v_iree, "save_iree", fake_save)
    monkeypatch.setattr(d2v_iree, "discrete_velocity_iree_availability", _available)
    return calls


def test_d2v_iree_modes_have_one_fixed_ordered_heterogeneous_abi():
    runtime, binding = _runtime_and_binding()
    state = _state(runtime)
    conserved = runtime.method.moments(state).conserved
    policy = IREEExportPolicy(target_backend="llvm-cpu", runtime_driver="local-task")

    equilibrium = d2v_iree.prepare_discrete_velocity_iree_contract(
        runtime,
        binding,
        conserved,
        host_id="host-a",
        mode="frozen-equilibrium",
        policy=policy,
    )
    one_step = d2v_iree.prepare_discrete_velocity_iree_contract(
        runtime,
        binding,
        state,
        host_id="host-a",
        mode="one-step",
        policy=policy,
    )
    horizon = d2v_iree.prepare_discrete_velocity_iree_contract(
        runtime,
        binding,
        state,
        host_id="host-a",
        mode="fixed-horizon",
        step_count=4,
        policy=policy,
    )

    expected_outputs = (
        "accepted_f",
        "accepted_g",
        "successful",
        "rollback_applied",
        "status",
        "first_failure_step",
        "maximum_conservation_residual",
        "maximum_energy_flux_error",
        "minimum_f",
        "minimum_g",
        "minimum_support_margin",
    )
    for contract in (equilibrium, one_step, horizon):
        assert contract.output_names == expected_outputs
        assert contract.output_shapes[:2] == (state.particle_populations.shape,) * 2
        assert contract.output_shapes[2:] == ((),) * 9
        assert contract.output_dtypes[:2] == (np.dtype(np.float64).str,) * 2
        assert contract.output_dtypes[2:4] == (np.dtype(bool).str,) * 2
        assert contract.output_dtypes[4:6] == (np.dtype(np.int32).str,) * 2
        assert contract.output_dtypes[6:] == (np.dtype(np.float64).str,) * 5
        assert not contract.supports_reverse_mode
        assert not contract.supports_training_export
        assert contract.frozen_artifact_id == binding.prepared_id
        assert contract.numeric_revision_id == binding.numeric_revision.revision_id
        assert contract.support_id == binding.plan.support.support_id
    assert equilibrium.input_names == ("conserved",)
    assert one_step.input_names == ("f", "g")
    assert horizon.step_count == 4
    assert horizon.step_size == runtime.required_step_size
    with pytest.raises(ValueError, match="host, backend, or artifact identity"):
        one_step.require_compatible(
            runtime, binding, host_id="another-host", policy=policy
        )
    with pytest.raises(ValueError, match="Only fixed-horizon"):
        d2v_iree.prepare_discrete_velocity_iree_contract(
            runtime,
            binding,
            state,
            host_id="host-a",
            mode="one-step",
            step_count=2,
        )


def test_d2v_iree_native_and_export_boundaries_agree_for_accept_and_reject(
    monkeypatch, tmp_path
):
    runtime, binding = _runtime_and_binding()
    calls = _install_fake_save(monkeypatch)
    accepted_input = _state(runtime)
    rejected_input = _state(runtime, density=3.0)

    equilibrium_bundle = d2v_iree.save_discrete_velocity_iree(
        runtime,
        binding,
        tmp_path / "equilibrium.phxiree",
        example=runtime.method.moments(accepted_input).conserved,
        host_id="host-a",
        mode="frozen-equilibrium",
    )
    accepted_bundle = d2v_iree.save_discrete_velocity_iree(
        runtime,
        binding,
        tmp_path / "accepted.phxiree",
        example=accepted_input,
        host_id="host-a",
        mode="fixed-horizon",
        step_count=3,
    )
    rejected_bundle = d2v_iree.save_discrete_velocity_iree(
        runtime,
        binding,
        tmp_path / "rejected.phxiree",
        example=rejected_input,
        host_id="host-a",
        mode="one-step",
    )

    equilibrium_native, equilibrium_deployed, equilibrium_call = calls[0]
    accepted_native, accepted_deployed, accepted_call = calls[1]
    rejected_native, rejected_deployed, rejected_call = calls[2]
    for native, deployed in (
        (equilibrium_native, equilibrium_deployed),
        (accepted_native, accepted_deployed),
        (rejected_native, rejected_deployed),
    ):
        assert len(native) == len(deployed) == 11
        for expected, actual in zip(native, deployed, strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert bool(equilibrium_native[2])
    assert not bool(equilibrium_native[3])
    assert bool(accepted_native[2])
    assert not bool(accepted_native[3])
    assert int(accepted_native[5]) == -1
    assert not bool(rejected_native[2])
    assert bool(rejected_native[3])
    assert int(rejected_native[5]) == 0
    np.testing.assert_array_equal(rejected_native[0], rejected_input.particle_populations)
    np.testing.assert_array_equal(
        rejected_native[1], rejected_input.total_energy_populations
    )
    assert equilibrium_call["input_names"] == equilibrium_bundle.contract.input_names
    assert accepted_call["output_names"] == accepted_bundle.contract.output_names
    assert rejected_call["output_names"] == rejected_bundle.contract.output_names
    assert accepted_call["validate"] is True
    assert accepted_bundle.vjp is None


def test_d2v_iree_unavailable_backend_fails_before_export_without_fallback(
    monkeypatch, tmp_path
):
    runtime, binding = _runtime_and_binding()
    state = _state(runtime)
    unavailable = BackendAvailability(
        capabilities=IREE_CAPABILITIES,
        available=False,
        requirement="install phydrax[iree]",
        reason="test compiler/runtime unavailable",
    )
    called = False

    def forbidden_save(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("generic exporter must not run after a failed gate")

    monkeypatch.setattr(
        d2v_iree, "discrete_velocity_iree_availability", lambda: unavailable
    )
    monkeypatch.setattr(d2v_iree, "save_iree", forbidden_save)
    with pytest.raises(BackendUnavailableError, match="compiled-inference"):
        d2v_iree.save_discrete_velocity_iree(
            runtime,
            binding,
            tmp_path / "unavailable.phxiree",
            example=state,
            host_id="host-a",
        )

    assert not called
    assert not (tmp_path / "unavailable.phxiree").exists()


@pytest.mark.parametrize(
    ("validation_ok", "target_backend"),
    ((False, None), (True, "cuda")),
)
def test_d2v_iree_refuses_parity_or_backend_identity_changes(
    monkeypatch, tmp_path, validation_ok, target_backend
):
    runtime, binding = _runtime_and_binding()
    calls = _install_fake_save(
        monkeypatch,
        validation_ok=validation_ok,
        target_backend=target_backend,
    )

    with pytest.raises(RuntimeError, match="backend, ABI, or native parity"):
        d2v_iree.save_discrete_velocity_iree(
            runtime,
            binding,
            tmp_path / "mismatch.phxiree",
            example=_state(runtime),
            host_id="host-a",
        )
    assert len(calls) == 1


def test_d2v_iree_refuses_foreign_frozen_artifact_identity():
    runtime, binding = _runtime_and_binding()
    _, foreign_binding = _runtime_and_binding(model_offset=0.1)
    contract = d2v_iree.prepare_discrete_velocity_iree_contract(
        runtime,
        binding,
        _state(runtime),
        host_id="host-a",
    )

    with pytest.raises(ValueError, match="host, backend, or artifact identity"):
        contract.require_compatible(runtime, foreign_binding, host_id="host-a")
