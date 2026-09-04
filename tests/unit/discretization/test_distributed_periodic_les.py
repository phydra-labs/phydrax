#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.spectral._distributed import (
    SpectralMeshTopology,
    SpectralResourceError,
)
from phydrax.discretization.spectral._distributed_les import (
    DistributedPeriodicLESPlan,
    DistributedPeriodicLESStage,
)
from phydrax.discretization.spectral._incompressible import PeriodicLerayProjector
from phydrax.equations._les_closures import (
    LESParameterProvenance,
    ResolvedLESFilter,
    SmagorinskyLESPlan,
)
from phydrax.equations._periodic_les import (
    PeriodicAlgebraicLESPlan,
    PeriodicFourierGridFilterPlan,
)


def _space(count=4):
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in range(3)))


def _scientific(space):
    resolved_filter = ResolvedLESFilter(
        "retained Fourier grid",
        family="sharp-fourier-projection",
        axis_names=("x", "y", "z"),
        topology="tensor-product",
        boundary_class="periodic",
        scale_rule="cutoff-equivalent",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    provenance = LESParameterProvenance(
        resolved_filter,
        space.prepared_id,
        "three-dimensional-periodic-unit-density",
        source_kind="user",
        evidence_ids=(),
    )
    model = SmagorinskyLESPlan(0.16).prepare(provenance)
    plan = PeriodicAlgebraicLESPlan(
        model,
        PeriodicFourierGridFilterPlan(resolved_filter),
        phx.discretization.PseudospectralMethodPlan(
            dealiasing=phx.discretization.OversamplingDealiasingPlan(1.5)
        ),
        energy_tolerance=3e-9,
    )
    projector = PeriodicLerayProjector(space)
    return plan.prepare(space, projector)


def _velocity(space):
    x, y, z = jnp.meshgrid(
        space.axes[0].nodes,
        space.axes[1].nodes,
        space.axes[2].nodes,
        indexing="ij",
    )
    physical = jnp.stack(
        (
            jnp.sin(2.0 * jnp.pi * y),
            jnp.sin(2.0 * jnp.pi * z),
            jnp.sin(2.0 * jnp.pi * x),
        ),
        axis=-1,
    )
    return PeriodicLerayProjector(space).project(space.project(physical))


def _topology(schedule="slab"):
    device = jax.devices("cpu")[0]
    if schedule == "pencil":
        return SpectralMeshTopology(
            (1, 1),
            devices=(device,),
            axis_names=("px", "py"),
        )
    return SpectralMeshTopology(
        (1,),
        devices=(device,),
        axis_names=("spectral",),
    )


def _distributed(scientific, schedule="slab", **kwargs):
    return DistributedPeriodicLESPlan(
        scientific,
        _topology(schedule),
        schedule=schedule,
        **kwargs,
    ).prepare()


def test_distributed_periodic_les_single_device_parity_and_backend_identity():
    space = _space()
    scientific = _scientific(space)
    distributed = _distributed(scientific)
    state = _velocity(space)

    evidence = distributed.parity_evidence(
        state,
        absolute_tolerance=2e-8,
        relative_tolerance=2e-8,
    )
    stage = distributed.evaluate(state)

    assert isinstance(stage, DistributedPeriodicLESStage)
    assert bool(evidence.finite)
    assert bool(evidence.passed)
    assert evidence.qualification_inherited is False
    assert distributed.prepared_id != scientific.prepared_id
    assert distributed.preparation.scientific_prepared_id == scientific.prepared_id
    assert distributed.preparation.qualification_inherited is False
    assert distributed.preparation.host_gather is False
    assert distributed.preparation.resource.closure_bytes > 0
    assert stage.projected_rate.sharding == distributed.execution.modal_layout.sharding(
        distributed.execution.topology
    )


def test_distributed_periodic_les_slab_pencil_layout_invariance_and_global_work():
    space = _space()
    scientific = _scientific(space)
    state = _velocity(space)
    slab = _distributed(scientific, "slab")
    pencil = _distributed(scientific, "pencil")

    slab_stage = slab.evaluate(state)
    pencil_stage = pencil.evaluate(state)
    reference = scientific.evaluate(state)

    np.testing.assert_allclose(
        slab_stage.projected_rate,
        pencil_stage.projected_rate,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        slab_stage.modal_deviatoric_specific_stress,
        reference.modal_deviatoric_specific_stress,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        slab_stage.modeled_dissipation,
        reference.modeled_dissipation,
        rtol=2e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        slab_stage.modal_energy_rate,
        reference.modal_energy_rate,
        rtol=2e-8,
        atol=2e-8,
    )
    assert abs(float(slab_stage.energy_identity_defect)) < 3e-9
    assert abs(float(slab_stage.projection_energy_defect)) < 3e-9
    assert bool(slab_stage.energy_consistent)
    assert slab.preparation.reduction_axes == ("spectral",)
    assert pencil.preparation.reduction_axes == ("py", "px")


def test_distributed_periodic_les_has_no_host_gather_and_restart_is_layout_bound(
    monkeypatch,
):
    space = _space()
    distributed = _distributed(_scientific(space))
    state = _velocity(space)

    with monkeypatch.context() as guard:
        guard.setattr(
            jax,
            "device_get",
            lambda *_args, **_kwargs: pytest.fail("host gather is forbidden"),
        )
        stage = distributed.evaluate(state)
        restart = distributed.restart_evidence(stage.projected_rate)
        restored = distributed.restore(restart)

    np.testing.assert_allclose(restored, stage.projected_rate, rtol=0.0, atol=0.0)
    assert restart.sharding_preserved
    assert restart.layout_id == distributed.execution.modal_layout.layout_id
    assert restart.topology_id == distributed.execution.topology.topology_id
    source = inspect.getsource(type(distributed))
    assert "device_get" not in source
    assert "process_allgather" not in source


def test_distributed_periodic_les_real_multi_device_slab_pencil_when_available():
    devices = tuple(jax.devices("cpu"))
    if len(devices) < 4:
        pytest.skip(
            "Four forced CPU devices are required for distributed LES collectives."
        )
    space = _space(8)
    scientific = _scientific(space)
    state = _velocity(space)
    slab_topology = SpectralMeshTopology(
        (4,),
        devices=devices[:4],
        axis_names=("spectral",),
    )
    pencil_topology = SpectralMeshTopology(
        (2, 2),
        devices=devices[:4],
        axis_names=("px", "py"),
    )
    slab = DistributedPeriodicLESPlan(
        scientific,
        slab_topology,
        schedule="slab",
    ).prepare()
    pencil = DistributedPeriodicLESPlan(
        scientific,
        pencil_topology,
        schedule="pencil",
    ).prepare()

    slab_stage = slab.evaluate(state)
    pencil_stage = pencil.evaluate(state)

    np.testing.assert_allclose(
        slab_stage.projected_rate,
        pencil_stage.projected_rate,
        rtol=3e-8,
        atol=3e-8,
    )
    np.testing.assert_allclose(
        slab_stage.modeled_dissipation,
        pencil_stage.modeled_dissipation,
        rtol=3e-8,
        atol=3e-8,
    )
    assert len(slab_stage.projected_rate.addressable_shards) == 4
    assert len(pencil_stage.projected_rate.addressable_shards) == 4
    assert slab_stage.reduction_axes == ("spectral",)
    assert pencil_stage.reduction_axes == ("py", "px")


def test_distributed_periodic_les_resource_and_support_refusals_are_exact(monkeypatch):
    space = _space()
    scientific = _scientific(space)
    topology = _topology()

    with pytest.raises(SpectralResourceError) as caught:
        DistributedPeriodicLESPlan(
            scientific,
            topology,
            maximum_bytes=128,
        ).prepare()
    assert caught.value.report.closure_bytes > 0
    assert caught.value.report.total_bytes > caught.value.report.maximum_bytes

    with pytest.raises(ValueError, match="only slab and pencil"):
        DistributedPeriodicLESPlan(scientific, topology, schedule="channel")

    unavailable = DistributedPeriodicLESPlan(scientific, topology)
    monkeypatch.setattr(jax, "devices", lambda *_args, **_kwargs: [])
    with pytest.raises(RuntimeError, match="unavailable"):
        unavailable.prepare()


def test_distributed_periodic_les_restriction_jit_and_jvp():
    space = _space()
    distributed = _distributed(_scientific(space))
    state = _velocity(space)

    eager = distributed.evaluate(state).projected_rate
    compiled = jax.jit(lambda value: distributed.evaluate(value).projected_rate)(state)
    direction = jnp.ones_like(state)
    _, tangent = jax.jvp(
        lambda value: distributed.evaluate(value).projected_rate,
        (state,),
        (direction,),
    )
    restriction = distributed.step_restriction(
        state,
        0.01,
        stage=distributed.evaluate(state),
    )

    np.testing.assert_allclose(compiled, eager, rtol=2e-8, atol=2e-8)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.isfinite(restriction.advective)
    assert jnp.isfinite(restriction.combined_diffusive)
    assert bool(restriction.finite)
    assert restriction.backend_id == distributed.prepared_id
