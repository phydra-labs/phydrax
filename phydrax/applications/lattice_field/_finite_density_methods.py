#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...integration import (
    ComplexWeightMeasure,
    phase_quenched_reweight,
    PhaseQuenchedReweightingPlan,
    PhaseQuenchedReweightingResult,
    prepare_phase_quenched_reweighting,
)
from ._finite_density import ChemicalChargeConvention


class MultiChargeCanonicalPlan(StrictModule, NonTrainableState):
    convention: ChemicalChargeConvention
    node_shape: tuple[int, int, int] = eqx.field(static=True)
    charge_bounds: tuple[int, int, int] = eqx.field(static=True)
    periodicities: tuple[float, float, float] = eqx.field(static=True)
    volume: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        convention: ChemicalChargeConvention,
        node_shape: Sequence[int],
        charge_bounds: Sequence[int],
        /,
        *,
        periodicities: Sequence[float],
        volume: float,
    ):
        if not isinstance(convention, ChemicalChargeConvention):
            raise TypeError("convention must be ChemicalChargeConvention.")
        nodes = tuple(node_shape)
        bounds = tuple(charge_bounds)
        periods = tuple(float(value) for value in periodicities)
        volume_ = float(volume)
        if len(nodes) != 3 or len(bounds) != 3 or len(periods) != 3:
            raise ValueError("Canonical plans require one B/Q/S axis each.")
        if any(
            node < 2 or bound < 0 or node < 2 * bound + 1
            for node, bound in zip(nodes, bounds, strict=True)
        ):
            raise ValueError(
                "Fourier node support does not resolve the requested charge bounds."
            )
        if (
            any(not np.isfinite(value) or value <= 0.0 for value in periods)
            or not np.isfinite(volume_)
            or volume_ <= 0.0
        ):
            raise ValueError("Periodicities and volume must be finite and positive.")
        self.convention = convention
        self.node_shape = nodes
        self.charge_bounds = bounds
        self.periodicities = periods
        self.volume = volume_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multi-charge-canonical-fourier-plan",
                "convention": convention.convention_id,
                "node_shape": list(nodes),
                "charge_bounds": list(bounds),
                "periodicities": list(periods),
                "volume": volume_,
            }
        )


class CanonicalSectorResult(StrictModule, NonTrainableState):
    sectors: Array
    charge_axes: tuple[Array, Array, Array]
    active: Array
    reconstruction_residual: Array
    conjugation_residual: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


def canonical_sector_transform(
    plan: MultiChargeCanonicalPlan,
    grand_partition_imaginary_mu: ArrayLike,
    /,
    *,
    reconstruction_holdout: ArrayLike | None = None,
    tolerance: float = 1.0e-10,
) -> CanonicalSectorResult:
    """Finite B/Q/S Fourier transform with explicit support and reconstruction evidence."""
    if not isinstance(plan, MultiChargeCanonicalPlan):
        raise TypeError("plan must be MultiChargeCanonicalPlan.")
    values = jnp.asarray(grand_partition_imaginary_mu)
    if values.shape != plan.node_shape or not jnp.iscomplexobj(values):
        raise ValueError(
            "grand_partition_imaginary_mu must be one complex value per B/Q/S node."
        )
    sectors = jnp.fft.fftshift(jnp.fft.fftn(values) / np.prod(plan.node_shape))
    axes = tuple(
        jnp.arange(-(node // 2), node - node // 2, dtype=jnp.int32)
        for node in plan.node_shape
    )
    active = (
        (jnp.abs(axes[0])[:, None, None] <= plan.charge_bounds[0])
        & (jnp.abs(axes[1])[None, :, None] <= plan.charge_bounds[1])
        & (jnp.abs(axes[2])[None, None, :] <= plan.charge_bounds[2])
    )
    retained = jnp.where(active, sectors, 0.0)
    reconstructed = jnp.fft.ifftn(jnp.fft.ifftshift(retained) * np.prod(plan.node_shape))
    target = (
        values if reconstruction_holdout is None else jnp.asarray(reconstruction_holdout)
    )
    if target.shape != plan.node_shape:
        raise ValueError("reconstruction_holdout must match node_shape.")
    reconstruction_residual = jnp.max(jnp.abs(reconstructed - target))
    reflected = values
    for axis, node_count in enumerate(plan.node_shape):
        reflected = jnp.take(
            reflected,
            (-jnp.arange(node_count, dtype=jnp.int32)) % node_count,
            axis=axis,
        )
    conjugation_residual = jnp.max(jnp.abs(values - jnp.conj(reflected)))
    finite = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(sectors))
    qualified = (
        finite
        & (reconstruction_residual <= tolerance)
        & (conjugation_residual <= tolerance)
    )
    return CanonicalSectorResult(
        sectors,
        axes,
        active,
        reconstruction_residual,
        conjugation_residual,
        finite,
        qualified,
        plan.plan_id,
    )


class QCDReweightingPlan(StrictModule, NonTrainableState):
    reference_theory_point_id: str = eqx.field(static=True)
    target_theory_point_id: str = eqx.field(static=True)
    determinant_prescription_id: str = eqx.field(static=True)
    chain_evidence_id: str = eqx.field(static=True)
    numerical: PhaseQuenchedReweightingPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reference_theory_point_id: str,
        target_theory_point_id: str,
        determinant_prescription_id: str,
        chain_evidence_id: str,
        numerical: PhaseQuenchedReweightingPlan | None = None,
    ):
        labels = tuple(
            str(value).strip()
            for value in (
                reference_theory_point_id,
                target_theory_point_id,
                determinant_prescription_id,
                chain_evidence_id,
            )
        )
        if any(not value for value in labels) or labels[0] == labels[1]:
            raise ValueError(
                "Distinct reference/target theory points and evidence are required."
            )
        numerical_ = PhaseQuenchedReweightingPlan() if numerical is None else numerical
        if not isinstance(numerical_, PhaseQuenchedReweightingPlan):
            raise TypeError("numerical must be PhaseQuenchedReweightingPlan.")
        (
            self.reference_theory_point_id,
            self.target_theory_point_id,
            self.determinant_prescription_id,
            self.chain_evidence_id,
        ) = labels
        self.numerical = numerical_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "qcd-phase-quenched-reweighting-plan",
                "reference": labels[0],
                "target": labels[1],
                "determinant": labels[2],
                "chain_evidence": labels[3],
                "numerical": numerical_.plan_id,
            }
        )


class QCDReweightingResult(StrictModule, NonTrainableState):
    numerical: PhaseQuenchedReweightingResult
    qualified: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def evaluate_qcd_reweighting(
    plan: QCDReweightingPlan,
    measure: ComplexWeightMeasure,
    observable_values: ArrayLike,
    /,
) -> QCDReweightingResult:
    if not isinstance(plan, QCDReweightingPlan) or not isinstance(
        measure, ComplexWeightMeasure
    ):
        raise TypeError("plan and measure must use QCD/generic complex-weight types.")
    prepared = prepare_phase_quenched_reweighting(measure, plan.numerical)
    result = phase_quenched_reweight(prepared, observable_values)
    return QCDReweightingResult(
        result,
        result.successful,
        plan.plan_id,
        "finite target estimate conditional on overlap and chain evidence; no generic sign-problem solution",
    )


__all__ = [
    "CanonicalSectorResult",
    "MultiChargeCanonicalPlan",
    "QCDReweightingPlan",
    "QCDReweightingResult",
    "canonical_sector_transform",
    "evaluate_qcd_reweighting",
]
