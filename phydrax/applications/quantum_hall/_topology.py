#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bulk Chern and time-reversal topology workflows for Hall lattice models."""

from __future__ import annotations

from math import prod
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import ReciprocalConnectivityPlan, ReciprocalMeshPlan
from ...operators.periodic import (
    identity_cross_k_connection,
    PeriodicBandManifold,
    PeriodicChernPlan,
    PeriodicChernResult,
    PeriodicOverlapBundle,
    PeriodicSpectrumPlan,
    PeriodicSpectrumResult,
    PeriodicTimeReversalPlan,
    PeriodicZ2Plan,
    PeriodicZ2Result,
)
from ._lattice import HaldaneModelPlan, HofstadterModelPlan, KaneMeleModelPlan


class HallBulkTopologyResult(StrictModule, NonTrainableState):
    spectrum: PeriodicSpectrumResult
    chern: PeriodicChernResult | None
    z2: PeriodicZ2Result | None
    successful: jnp.ndarray
    model_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _chern_result(
    prepared,
    mesh_shape: tuple[int, int],
    occupied_bands: Sequence[int],
    /,
) -> tuple[PeriodicSpectrumResult, PeriodicChernResult]:
    mesh = ReciprocalMeshPlan.monkhorst_pack(
        prepared.plan.basis.cell,
        mesh_shape,
    )
    spectrum = PeriodicSpectrumPlan(prepared, mesh).evaluate()
    connectivity = ReciprocalConnectivityPlan.regular(mesh).prepare()
    manifold = PeriodicBandManifold(spectrum, occupied_bands)
    bundle = PeriodicOverlapBundle(
        manifold,
        identity_cross_k_connection(prepared, connectivity),
    )
    return spectrum, PeriodicChernPlan(bundle).evaluate()


def evaluate_haldane_topology(
    plan: HaldaneModelPlan,
    /,
    *,
    mesh_shape: tuple[int, int] = (31, 31),
) -> HallBulkTopologyResult:
    if not isinstance(plan, HaldaneModelPlan):
        raise TypeError("plan must be HaldaneModelPlan.")
    spectrum, chern = _chern_result(plan.prepare(), mesh_shape, (0,))
    return HallBulkTopologyResult(
        spectrum,
        chern,
        None,
        chern.successful,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "haldane-bulk-topology-result",
                "model": plan.plan_id,
                "spectrum": spectrum.result_id,
                "chern": chern.result_id,
            }
        ),
    )


def evaluate_hofstadter_topology(
    plan: HofstadterModelPlan,
    occupied_bands: Sequence[int],
    /,
    *,
    mesh_shape: tuple[int, int] = (31, 31),
) -> HallBulkTopologyResult:
    if not isinstance(plan, HofstadterModelPlan):
        raise TypeError("plan must be HofstadterModelPlan.")
    bands = tuple(int(value) for value in occupied_bands)
    if not bands or len(set(bands)) != len(bands):
        raise ValueError("occupied_bands must be nonempty and unique.")
    spectrum, chern = _chern_result(plan.prepare(), mesh_shape, bands)
    return HallBulkTopologyResult(
        spectrum,
        chern,
        None,
        chern.successful,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "hofstadter-bulk-topology-result",
                "model": plan.plan_id,
                "spectrum": spectrum.result_id,
                "chern": chern.result_id,
                "occupied_bands": bands,
            }
        ),
    )


def _time_reversal_mesh(prepared, mesh_shape: tuple[int, int]) -> ReciprocalMeshPlan:
    shape = tuple(int(value) for value in mesh_shape)
    if len(shape) != 2 or any(value < 4 or value % 2 for value in shape):
        raise ValueError(
            "Time-reversal meshes require two even dimensions of at least four."
        )
    axes = tuple(np.arange(size, dtype=np.float64) / size for size in shape)
    indices = np.asarray(
        tuple(np.ndindex(shape)),
        dtype=np.int32,
    )
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 2))
    return ReciprocalMeshPlan(
        prepared.plan.basis.cell,
        points,
        np.full((prod(shape),), 1.0 / prod(shape)),
        mesh_shape=shape,
        shift=(0.0, 0.0),
        mesh_indices=indices,
    )


def evaluate_kane_mele_topology(
    plan: KaneMeleModelPlan,
    /,
    *,
    mesh_shape: tuple[int, int] = (24, 24),
) -> HallBulkTopologyResult:
    if not isinstance(plan, KaneMeleModelPlan):
        raise TypeError("plan must be KaneMeleModelPlan.")
    prepared = plan.prepare()
    mesh = _time_reversal_mesh(prepared, mesh_shape)
    spectrum = PeriodicSpectrumPlan(prepared, mesh).evaluate()
    connectivity = ReciprocalConnectivityPlan.regular(mesh).prepare()
    manifold = PeriodicBandManifold(spectrum, (0, 1))
    bundle = PeriodicOverlapBundle(
        manifold,
        identity_cross_k_connection(prepared, connectivity),
    )
    time_reversal = PeriodicTimeReversalPlan(
        prepared,
        mesh,
        plan.time_reversal_unitary,
    ).evaluate()
    z2 = PeriodicZ2Plan(bundle, time_reversal).evaluate()
    return HallBulkTopologyResult(
        spectrum,
        None,
        z2,
        z2.successful,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "kane-mele-bulk-topology-result",
                "model": plan.plan_id,
                "spectrum": spectrum.result_id,
                "z2": z2.result_id,
            }
        ),
    )


__all__ = [
    "HallBulkTopologyResult",
    "evaluate_haldane_topology",
    "evaluate_hofstadter_topology",
    "evaluate_kane_mele_topology",
]
