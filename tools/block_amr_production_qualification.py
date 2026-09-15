#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _identity(points, time, args):
    del time, args
    return points


def _static_plane(points, time, args):
    del time, args
    return points[:, 0] - 0.37


def _moving_plane(points, time, args):
    del args
    return points[:, 0] - (0.35 + 0.01 * time)


def _topology(*, x_cells=1):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(x_cells),
            phx.discretization.UniformCellAxisSpec(1),
            phx.discretization.UniformCellAxisSpec(1),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((x_cells, 1, 1), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(signature, 1),),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0, 0), (x_cells, 1, 1)),),
    )
    return phx.discretization.VariablePatchTopologyCompiler(hierarchy).initial_topology()


def _topology_2d():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(1),
            phx.discretization.UniformCellAxisSpec(1),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((1, 1), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(signature, 1),),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0), (1, 1)),),
    )
    return phx.discretization.VariablePatchTopologyCompiler(hierarchy).initial_topology()


def _resources():
    return phx.discretization.BlockAMRResourcePlan(
        maximum_components_per_cell=4,
        maximum_apertures_per_face=96,
        maximum_embedded_faces_per_cell=256,
        maximum_mortars=64,
        maximum_redistribution_routes=128,
        maximum_topology_events=16,
        maximum_communication_peers=64,
    )


def _cut_plan(level_set=_static_plane, field_id="production-plane"):
    return phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "production-identity-map",
        phx.discretization.EmbeddedLevelSetBodySet(
            (phx.discretization.EmbeddedLevelSetBody(level_set, field_id, 7),)
        ),
        _resources(),
    )


def qualify() -> dict[str, object]:
    topology = _topology()
    hierarchy = phx.discretization.canonicalize_patch_hierarchy(topology)
    resource = _resources().preflight(
        hierarchy,
        physical_component_count=5,
        dtype=np.float64,
    )
    mapped = phx.discretization.CanonicalMappedGeometryPlan(
        topology,
        phx.discretization.PatchCoordinateMapSet(
            lambda point, time, args: (1.0 + 0.02 * time) * point,
            "production-dilation",
        ),
        quadrature_order=3,
        tolerance=2.0e-6,
    ).evaluate(0.25, revision=1)
    cut = _cut_plan().prepare()
    multivalued_body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: (points[:, 0] - 0.3) * (points[:, 0] - 0.7),
        "production-slab",
        9,
    )
    multivalued = phx.discretization.MultivaluedCutCellPlan(
        topology,
        _identity,
        "production-identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((multivalued_body,)),
        _resources(),
        subdivision=4,
    ).prepare()
    multivalued_2d = phx.discretization.MultivaluedCutCell2DPlan(
        _topology_2d(),
        _identity,
        "production-identity-map-2d",
        phx.discretization.EmbeddedLevelSetBodySet((multivalued_body,)),
        _resources(),
        subdivision=4,
    ).prepare()

    def slab_interval(lower, upper, time, args):
        del time, args
        midpoint = jnp.clip(0.5, lower[0], upper[0])
        candidates = jnp.asarray(
            (
                (lower[0] - 0.3) * (lower[0] - 0.7),
                (upper[0] - 0.3) * (upper[0] - 0.7),
                (midpoint - 0.3) * (midpoint - 0.7),
            )
        )
        return jnp.min(candidates), jnp.max(candidates)

    adaptive = phx.discretization.AdaptiveImplicitSamplingPlan(
        topology,
        _identity,
        "production-identity-map",
        (
            phx.discretization.CertifiedImplicitBody(
                multivalued_body,
                slab_interval,
                lambda points, values, time, args: (
                    jnp.any(values > 0.0) & jnp.any(values < 0.0)
                ),
                "production-pl-certificate",
            ),
        ),
        maximum_depth=3,
    ).prepare()

    mortar_maps = phx.discretization.PatchCoordinateMapSet(
        lambda point, time, args: point,
        "production-mortar-identity",
    )
    mortar = phx.discretization.MappedMortarPlan(
        "left",
        "right",
        lambda parameter, time, args: jnp.asarray((0.5, parameter[0])),
        lambda parameter, time, args: jnp.asarray((0.5, parameter[0])),
        lambda parameter, time, args: jnp.asarray((0.5, parameter[0])),
        ((0.0, 1.0),),
    ).prepare(mortar_maps)

    system = phx.equations.EulerSystem(3)
    discretization = cut.finite_volume_plan(
        component_names=system.component_names
    ).prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: (
                phx.discretization.SlipWallBoundary()
                if name.startswith("embedded-")
                else phx.discretization.ExtrapolationBoundary()
            )
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "production-block-amr",
        "state",
        system,
        boundaries,
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 0.0, 1.0)),
        discretization.state_shape,
    )
    average = system.primitive_to_conserved(primitive)
    advanced = runtime.advance(runtime.initialize_state(average, 0.0, 1.0e-4))

    diffusion = phx.discretization.MultivaluedCutCellDiffusionPlan(cut, 1.0)
    diffusion_constant_defect = float(
        jnp.max(jnp.abs(diffusion.apply(jnp.ones((diffusion.cell_count,)))))
    )
    cochain = phx.discretization.CutCellCochainPlan(cut).prepare()
    complex_topology = cochain.topology.topology
    edge = jnp.linspace(-0.2, 0.3, complex_topology.entities(1).count)
    divergence_of_curl = (
        complex_topology.incidences[2]
        .exterior_derivative()
        .mv(complex_topology.incidences[1].exterior_derivative().mv(edge))
    )
    chain_defect = float(jnp.max(jnp.abs(divergence_of_curl), initial=0.0))
    cochain_plan = phx.discretization.CutCellCochainPlan(cut)
    cochain_transfer = phx.discretization.CutCellCochainTransferPlan(
        cochain_plan,
        cochain_plan,
        cochain,
        cochain,
    )

    moving_plan = _cut_plan(_moving_plane, "production-moving-plane")
    moving = phx.solver.MovingMultivaluedCutCellPlan(
        moving_plan,
        tolerance=2.0e-7,
    )
    moving_result = moving.advance(
        moving.initialize(jnp.asarray(2.0), 0.0),
        1.0,
        jnp.asarray(2.0),
    )

    def oscillating_plane(points, time, args):
        del args
        return points[:, 0] - (0.5 - 2.8 * (time - 0.5) ** 2)

    localized_plan = _cut_plan(oscillating_plane, "production-oscillating-plane")
    localized_moving = phx.solver.MovingMultivaluedCutCellPlan(
        localized_plan,
        tolerance=3.0e-7,
    )
    localized_result = localized_moving.advance_localized(
        localized_moving.initialize(jnp.asarray(2.0), 0.0),
        1.0,
        jnp.asarray(2.0),
        phx.solver.MovingTopologyLocalizationPlan(
            probe_count=16,
            bisection_iterations=18,
            minimum_event_separation=1.0e-3,
        ),
    )

    transition = phx.discretization.MultivaluedCutCellTransition(cut, cut)
    derivative = phx.discretization.FrozenCutCellTransitionDerivativePlan(
        transition,
        topology_margin=0.1,
    )
    content = jnp.zeros((cut.component_capacity, 1)).at[0, 0].set(1.0)
    tangent = jnp.zeros_like(content).at[0, 0].set(0.25)
    derivative_result = derivative.jvp_content(content, tangent)

    group = phx.execution.ExecutionRuntime.current().root_group
    distributed = phx.discretization.DistributedCutCellPartitionPlan(
        group,
        max(1, cut.component_count),
    ).prepare(cut)
    distributed_state = distributed.pack(content)
    distributed_roundtrip = distributed.unpack(distributed_state)
    local_indices = distributed.local_component_indices()
    process_local_state = distributed.pack_process_local(
        local_indices,
        content[jnp.asarray(local_indices, dtype=jnp.int32)],
    )
    process_local_roundtrip = distributed.unpack_process_local(process_local_state)

    cache_plan = phx.discretization.PatchExecutableCachePlan(_resources())
    signatures = cache_plan.required_signatures(
        hierarchy,
        physical_component_count=2,
        method_id="production-qualification-kernel",
        dtype=np.float64,
    )
    cache_result = cache_plan.install(
        phx.discretization.PatchExecutableCacheState(),
        signatures,
        lambda signature: lambda values: values + 1.0,
        lambda signature: (
            (jnp.zeros(signature.state_shape, dtype=signature.dtype),),
            {},
        ),
    )

    checkpoint_state = phx.solver.MovingMultivaluedCutCellPlan(_cut_plan()).initialize(
        jnp.asarray(((2.0, 3.0),)), 0.25
    )
    with TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "checkpoint"
        checkpoint_plan = phx.solver.MultivaluedBlockAMRCheckpointPlan(
            _cut_plan(),
            ("first", "second"),
        )
        written = phx.solver.write_multivalued_block_amr_checkpoint(
            checkpoint_path,
            checkpoint_plan,
            checkpoint_state,
        )
        restored = phx.solver.read_multivalued_block_amr_checkpoint(
            checkpoint_path,
            phx.solver.CutCellRestartRegistry(
                {"production-identity-map": _identity},
                {"production-plane": _static_plane},
            ),
        )
        checkpoint_exact = restored.payload_id == written.payload_id and bool(
            jnp.array_equal(restored.state.content, checkpoint_state.content)
        )

    mortar_state = phx.equations.EulerSystem(2).primitive_to_conserved(
        jnp.asarray((1.0, 0.0, 0.0, 1.0))
    )
    mortar_flux = phx.discretization.MappedMortarFluxPlan(
        mortar,
        0,
        1,
        2,
    ).evaluate(
        phx.equations.EulerSystem(2),
        phx.discretization.RusanovFluxPlan(),
        mortar_state,
        mortar_state,
    )
    parity = phx.discretization.BlockAMRReferenceParityPlan(
        {
            "cut_centroid_x": np.asarray((0.685,)),
            "cut_volume": np.asarray((0.63,)),
            "mortar_measure": np.asarray((1.0,)),
            "multivalued_2d_areas": np.asarray((0.34, 0.34)),
        },
        provider_id="analytic-piecewise-linear-reference",
        provider_revision="plane-and-slab-closed-form",
        absolute_tolerance=2.0e-6,
        relative_tolerance=2.0e-6,
    ).compare(
        {
            "cut_centroid_x": np.asarray((cut.component_centers[0, 0],)),
            "cut_volume": np.asarray((cut.component_volumes[0],)),
            "mortar_measure": np.asarray((jnp.sum(mortar.quadrature_weights),)),
            "multivalued_2d_areas": np.sort(
                np.asarray(multivalued_2d.component_areas)[
                    np.asarray(multivalued_2d.component_active)
                ]
            ),
        }
    )

    gates = {
        "resource_preflight": resource.valid,
        "mapped_metric_gcl": bool(mapped.evidence.valid),
        "three_dimensional_cut_closure": cut.evidence.valid,
        "multivalued_components": multivalued.component_count == 2,
        "two_dimensional_multivalued": multivalued_2d.component_count == 2,
        "adaptive_implicit_sampling": adaptive.evidence.valid
        and adaptive.evidence.maximum_depth_used >= 1,
        "nonconforming_mortar_flux": bool(mortar_flux.successful),
        "independent_reference_parity": parity.passed,
        "stationary_finite_volume": bool(advanced.accepted),
        "viscous_elliptic_constant": diffusion_constant_defect <= 2.0e-7,
        "compatible_chain": chain_defect <= 2.0e-7,
        "moving_space_time_ledger": bool(moving_result.accepted),
        "multiple_topology_events": bool(localized_result.accepted)
        and len(localized_result.localization.event_times) >= 2,
        "cochain_topology_transfer": cochain_transfer.evidence.valid,
        "frozen_history_derivative": bool(derivative_result.evidence.valid),
        "distributed_roundtrip": bool(jnp.array_equal(distributed_roundtrip, content)),
        "process_local_global_array": bool(
            jnp.array_equal(
                process_local_roundtrip,
                content[jnp.asarray(local_indices, dtype=jnp.int32)],
            )
        ),
        "dynamic_signature_compile": cache_result.changed,
        "portable_restart": checkpoint_exact,
    }
    multihost_available = jax.process_count() > 1
    required_pass = all(gates.values())
    status = (
        "pass"
        if required_pass and multihost_available
        else "inconclusive"
        if required_pass
        else "fail"
    )
    return {
        "status": status,
        "released": status == "pass",
        "gates": gates,
        "hardware_gates": {
            "real_multihost_available": multihost_available,
            "process_count": jax.process_count(),
            "device_count": jax.device_count(),
        },
        "evidence": {
            "resource_evidence_id": resource.evidence_id,
            "mapped_evidence_id": mapped.evidence.evidence_id,
            "cut_evidence_id": cut.evidence.evidence_id,
            "multivalued_evidence_id": multivalued.evidence.evidence_id,
            "moving_evidence_id": moving_result.evidence.evidence_id,
            "adaptive_sampling_evidence_id": adaptive.evidence.evidence_id,
            "cochain_transfer_id": cochain_transfer.plan_id,
            "mortar_flux_plan_id": mortar_flux.plan_id,
            "localized_event_evidence_id": localized_result.localization.evidence_id,
            "cochain_state_id": cochain.state_id,
            "distributed_partition_id": distributed.partition_id,
            "signature_cache_id": cache_result.state.cache_id,
            "maximum_cut_face_closure_defect": cut.evidence.maximum_face_closure_defect,
            "reference_parity_evidence_id": parity.evidence_id,
            "maximum_reference_tolerance_ratio": max(parity.maximum_tolerance_ratios),
            "maximum_chain_defect": chain_defect,
            "moving_volume_balance_defect": float(
                moving_result.evidence.volume_balance_defect
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
    print(payload)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
