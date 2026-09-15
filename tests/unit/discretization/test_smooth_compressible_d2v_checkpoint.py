#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax.discretization.discrete_velocity._checkpoint import (
    read_smooth_compressible_d2v_checkpoint,
    SmoothCompressibleD2VCheckpointPlan,
    write_smooth_compressible_d2v_checkpoint,
)
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


def _runtime(*, viscosity=0.03):
    quadrature = d2v17_quadrature()
    method = SmoothCompressibleD2VKineticMethod(
        quadrature,
        IdealGasMaterial(1.4, 1.0),
        ConstantTransport(viscosity, 0.04),
    )
    return PreparedSmoothCompressibleD2V17SpatialDynamics(
        method,
        PositiveEnergyEquilibriumPlan(quadrature),
        D2V17PeriodicTransportPlan(
            quadrature,
            (5, 6),
            (0.01, 0.01),
            0.01,
        ),
    )


def _state(runtime):
    conserved = jnp.asarray((1.0, 0.03, -0.02, 1.251))
    moments = runtime.method._equilibrium_fields(conserved)
    target_flux = (moments[3] + moments[5]) * moments[4]
    oracle = runtime.energy_plan.solve(moments[3], target_flux)
    equilibrium, evidence = runtime.method.equilibrium_from_energy_dual_with_evidence(
        conserved, oracle.dual, runtime.energy_plan
    )
    assert bool(evidence.successful)
    shape = runtime.transport.spatial_shape + (17,)
    return SmoothCompressibleKineticState(
        jnp.broadcast_to(equilibrium.particle_populations, shape),
        jnp.broadcast_to(equilibrium.total_energy_populations, shape),
    )


def _plan(runtime, **overrides):
    identities = {
        "topology_id": "periodic-topology",
        "support_id": "learned-support",
        "frozen_artifact_id": "frozen-energy-artifact",
        "numeric_revision_id": "energy-numeric-revision",
    }
    identities.update(overrides)
    return SmoothCompressibleD2VCheckpointPlan(
        runtime,
        identities["topology_id"],
        identities["support_id"],
        identities["frozen_artifact_id"],
        identities["numeric_revision_id"],
        boundary_history_names=("outflow_memory",),
        source_history_names=("source_phase",),
    )


def _histories():
    return {
        "boundary_history": {"outflow_memory": jnp.asarray((0.25, -0.5))},
        "source_history": {"source_phase": jnp.asarray(3, dtype=jnp.int32)},
    }


def test_d2v_checkpoint_continues_exactly_from_accepted_populations(tmp_path):
    runtime = _runtime()
    initial = _state(runtime)
    first, _ = runtime.step_oracle(initial, jnp.asarray(0.01))
    assert bool(first.successful)
    plan = _plan(runtime)
    path = tmp_path / "accepted-d2v.phxcheckpoint"

    written = write_smooth_compressible_d2v_checkpoint(
        path,
        plan,
        jnp.asarray(0.01),
        jnp.asarray(1, dtype=jnp.int32),
        first.accepted_state,
        accepted=first.successful,
        **_histories(),
    )
    restored = read_smooth_compressible_d2v_checkpoint(
        path,
        plan,
        initial,
        boundary_history_template={"outflow_memory": jnp.zeros((2,))},
        source_history_template={"source_phase": jnp.asarray(0, dtype=jnp.int32)},
    )
    uninterrupted, _ = runtime.step_oracle(first.accepted_state, jnp.asarray(0.01))
    continued, _ = runtime.step_oracle(restored.accepted_state, jnp.asarray(0.01))

    assert restored.payload_id == written.payload_id
    assert int(restored.step_index) == 1
    assert float(restored.time) == pytest.approx(0.01)
    np.testing.assert_array_equal(
        restored.accepted_state.particle_populations,
        first.accepted_state.particle_populations,
    )
    np.testing.assert_array_equal(
        restored.accepted_state.total_energy_populations,
        first.accepted_state.total_energy_populations,
    )
    np.testing.assert_array_equal(
        continued.accepted_state.particle_populations,
        uninterrupted.accepted_state.particle_populations,
    )
    np.testing.assert_array_equal(
        continued.accepted_state.total_energy_populations,
        uninterrupted.accepted_state.total_energy_populations,
    )
    np.testing.assert_array_equal(
        restored.boundary_value("outflow_memory"), jnp.asarray((0.25, -0.5))
    )
    assert int(restored.source_value("source_phase")) == 3


def test_d2v_checkpoint_refuses_identity_mismatch_and_payload_tamper(tmp_path):
    runtime = _runtime()
    state = _state(runtime)
    plan = _plan(runtime)
    path = tmp_path / "identity-bound.phxcheckpoint"
    write_smooth_compressible_d2v_checkpoint(
        path,
        plan,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        state,
        accepted=jnp.asarray(True),
        **_histories(),
    )

    incompatible_plans = (
        _plan(runtime, topology_id="another-topology"),
        _plan(runtime, support_id="another-support"),
        _plan(runtime, frozen_artifact_id="another-artifact"),
        _plan(runtime, numeric_revision_id="another-revision"),
        _plan(_runtime(viscosity=0.031)),
    )
    for incompatible in incompatible_plans:
        with pytest.raises(ValueError, match="does not match the runtime"):
            read_smooth_compressible_d2v_checkpoint(
                path,
                incompatible,
                state,
                boundary_history_template={"outflow_memory": jnp.zeros((2,))},
                source_history_template={"source_phase": jnp.asarray(0, dtype=jnp.int32)},
            )

    manifest, arrays = read_array_archive(path)
    rewritten_manifest = dict(manifest)
    rewritten_manifest.pop("arrays")
    population_name = next(
        name
        for name, value in arrays.items()
        if value.shape == state.particle_populations.shape
    )
    corrupted_arrays = dict(arrays)
    corrupted_arrays[population_name] = np.asarray(arrays[population_name]).copy()
    corrupted_arrays[population_name].flat[0] += 1.0
    write_array_archive(path, manifest=rewritten_manifest, arrays=corrupted_arrays)
    with pytest.raises(ValueError, match="payload identity"):
        read_smooth_compressible_d2v_checkpoint(
            path,
            plan,
            state,
            boundary_history_template={"outflow_memory": jnp.zeros((2,))},
            source_history_template={"source_phase": jnp.asarray(0, dtype=jnp.int32)},
        )


def test_d2v_checkpoint_is_accepted_boundary_only_and_minimal(tmp_path):
    runtime = _runtime()
    state = _state(runtime)
    plan = _plan(runtime)
    path = tmp_path / "accepted-only.phxcheckpoint"
    original = write_smooth_compressible_d2v_checkpoint(
        path,
        plan,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        state,
        accepted=jnp.asarray(True),
        **_histories(),
    )
    rejected_candidate = SmoothCompressibleKineticState(
        state.particle_populations + 0.1,
        state.total_energy_populations + 0.2,
    )

    with pytest.raises(ValueError, match="accepted boundary"):
        write_smooth_compressible_d2v_checkpoint(
            path,
            plan,
            jnp.asarray(0.01),
            jnp.asarray(1, dtype=jnp.int32),
            rejected_candidate,
            accepted=jnp.asarray(False),
            **_histories(),
        )
    restored = read_smooth_compressible_d2v_checkpoint(
        path,
        plan,
        state,
        boundary_history_template={"outflow_memory": jnp.zeros((2,))},
        source_history_template={"source_phase": jnp.asarray(0, dtype=jnp.int32)},
    )
    manifest, arrays = read_array_archive(path)

    assert restored.payload_id == original.payload_id
    np.testing.assert_array_equal(
        restored.accepted_state.particle_populations, state.particle_populations
    )
    assert len(arrays) == 6
    assert manifest["checkpoint_fields"] == [
        "particle_populations",
        "total_energy_populations",
    ]
    archived_paths = (*arrays, *manifest["state"]["paths"])
    assert all(
        "model" not in name and "conserved" not in name and "candidate" not in name
        for name in archived_paths
    )
    with pytest.raises(ValueError, match="exactly match"):
        write_smooth_compressible_d2v_checkpoint(
            tmp_path / "extra-history.phxcheckpoint",
            plan,
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int32),
            state,
            accepted=jnp.asarray(True),
            boundary_history={
                "outflow_memory": jnp.zeros((2,)),
                "derived_pressure": jnp.ones((5, 6)),
            },
            source_history={"source_phase": jnp.asarray(0, dtype=jnp.int32)},
        )
