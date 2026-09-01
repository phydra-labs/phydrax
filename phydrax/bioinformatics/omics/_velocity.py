#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


VELOCITY_SUCCESS = 0
VELOCITY_NONFINITE = 1
VELOCITY_NEGATIVE_COUNT = 2
VELOCITY_INVALID_RATE = 3
VELOCITY_LAYER_ASSUMPTION_FAILED = 4
VELOCITY_PROVENANCE_MISMATCH = 5


def velocity_status_name(status: int, /) -> str:
    """Return the stable name of a kinetic RNA-velocity status code."""
    names = (
        "success",
        "nonfinite_count_or_rate",
        "negative_count",
        "invalid_kinetic_rate",
        "spliced_unspliced_assumption_failed",
        "fitted_provenance_mismatch",
    )
    code = int(status)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown RNA-velocity status {code}.")
    return names[code]


def _velocity_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "kinetic_spliced_unspliced_rna_velocity",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.EXACT_AD,
        OutputKind.ARRAY,
        conditioning_statement=(
            "Velocity conditions on observed spliced/unspliced count layers and "
            "supplied transcription, splicing, and degradation rates."
        ),
        truncation_statement="No cells or genes are truncated.",
        capacity_semantics="Cell and gene axes are complete array dimensions.",
        assumptions=(
            "The two inputs are distinct, aligned spliced and unspliced count layers.",
            "Rates are nonnegative and constant at the evaluated state.",
        ),
        nondifferentiable_outputs=("status", "valid"),
    )


class KineticRNAVelocityPlan(StrictModule):
    """Explicit count-layer and fitted-parameter assumptions for kinetic velocity."""

    spliced_layer_name: str = eqx.field(static=True)
    unspliced_layer_name: str = eqx.field(static=True)
    count_layers: bool = eqx.field(static=True)
    shared_cell_gene_order: bool = eqx.field(static=True)
    parameter_origin: str = eqx.field(static=True)
    fitted_on_split_id: str = eqx.field(static=True)
    steady_state_assumption: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        spliced_layer_name: str = "spliced",
        unspliced_layer_name: str = "unspliced",
        count_layers: bool = True,
        shared_cell_gene_order: bool = True,
        parameter_origin: str,
        fitted_on_split_id: str,
        steady_state_assumption: bool = False,
    ):
        if (
            not spliced_layer_name
            or not unspliced_layer_name
            or spliced_layer_name == unspliced_layer_name
        ):
            raise ValueError("Spliced and unspliced layers must be distinct and named.")
        if parameter_origin not in ("externally_calibrated", "training_fitted"):
            raise ValueError("parameter_origin must classify rate provenance.")
        if not fitted_on_split_id:
            raise ValueError("fitted_on_split_id must be non-empty.")
        self.spliced_layer_name = str(spliced_layer_name)
        self.unspliced_layer_name = str(unspliced_layer_name)
        self.count_layers = bool(count_layers)
        self.shared_cell_gene_order = bool(shared_cell_gene_order)
        self.parameter_origin = str(parameter_origin)
        self.fitted_on_split_id = str(fitted_on_split_id)
        self.steady_state_assumption = bool(steady_state_assumption)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-rna-velocity-plan",
                "spliced_layer_name": self.spliced_layer_name,
                "unspliced_layer_name": self.unspliced_layer_name,
                "count_layers": self.count_layers,
                "shared_cell_gene_order": self.shared_cell_gene_order,
                "parameter_origin": self.parameter_origin,
                "fitted_on_split_id": self.fitted_on_split_id,
                "steady_state_assumption": self.steady_state_assumption,
            }
        )


class KineticRNAVelocityEvidence(StrictModule):
    """Per-cell validity plus layer, rate, and training provenance evidence."""

    finite_counts: Array
    nonnegative_counts: Array
    finite_rates: Array
    nonnegative_rates: Array
    spliced_unspliced_assumptions: Array
    fitted_provenance_match: Array
    parameters_training_fitted: Array
    uses_steady_state_assumption: Array
    spliced_layer_name: str = eqx.field(static=True)
    unspliced_layer_name: str = eqx.field(static=True)


class KineticRNAVelocityResult(StrictModule):
    """Kinetic derivatives for unspliced and spliced abundance states."""

    unspliced_velocity: Array
    spliced_velocity: Array
    valid: Array
    status: Array
    evidence: KineticRNAVelocityEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)
    claim_kind: str = eqx.field(static=True)


def kinetic_rna_velocity(
    spliced_counts: ArrayLike,
    unspliced_counts: ArrayLike,
    transcription_rate: ArrayLike,
    splicing_rate: ArrayLike,
    degradation_rate: ArrayLike,
    plan: KineticRNAVelocityPlan,
    /,
    *,
    expected_split_id: str,
    method_contract: BioinformaticsMethodContract | None = None,
) -> KineticRNAVelocityResult:
    """Evaluate du/dt = alpha - beta*u and ds/dt = beta*u - gamma*s."""
    spliced = jnp.asarray(spliced_counts)
    unspliced = jnp.asarray(unspliced_counts)
    alpha = jnp.asarray(transcription_rate)
    beta = jnp.asarray(splicing_rate)
    gamma = jnp.asarray(degradation_rate)
    if not isinstance(plan, KineticRNAVelocityPlan):
        raise TypeError("plan must be KineticRNAVelocityPlan.")
    if spliced.ndim != 2 or unspliced.shape != spliced.shape:
        raise ValueError(
            "Spliced and unspliced counts must be matching cell-gene arrays."
        )
    gene_count = int(spliced.shape[1])
    for name, rate in (
        ("transcription_rate", alpha),
        ("splicing_rate", beta),
        ("degradation_rate", gamma),
    ):
        if rate.ndim > 1 or (rate.ndim == 1 and rate.shape != (gene_count,)):
            raise ValueError(f"{name} must be scalar or contain one value per gene.")
    if not expected_split_id:
        raise ValueError("expected_split_id must be non-empty.")

    alpha = jnp.broadcast_to(alpha, (gene_count,))
    beta = jnp.broadcast_to(beta, (gene_count,))
    gamma = jnp.broadcast_to(gamma, (gene_count,))
    finite_counts = jnp.all(jnp.isfinite(spliced) & jnp.isfinite(unspliced), axis=1)
    nonnegative_counts = jnp.all((spliced >= 0.0) & (unspliced >= 0.0), axis=1)
    finite_rates = jnp.all(jnp.isfinite(alpha) & jnp.isfinite(beta) & jnp.isfinite(gamma))
    nonnegative_rates = jnp.all((alpha >= 0.0) & (beta >= 0.0) & (gamma >= 0.0))
    assumptions = jnp.asarray(plan.count_layers and plan.shared_cell_gene_order)
    provenance_match = jnp.asarray(plan.fitted_on_split_id == str(expected_split_id))
    unspliced_velocity = alpha[None, :] - beta[None, :] * unspliced
    spliced_velocity = beta[None, :] * unspliced - gamma[None, :] * spliced
    valid = (
        finite_counts
        & nonnegative_counts
        & finite_rates
        & nonnegative_rates
        & assumptions
        & provenance_match
    )
    status = jnp.where(
        ~finite_counts | ~finite_rates,
        VELOCITY_NONFINITE,
        jnp.where(
            ~nonnegative_counts,
            VELOCITY_NEGATIVE_COUNT,
            jnp.where(
                ~nonnegative_rates,
                VELOCITY_INVALID_RATE,
                jnp.where(
                    ~assumptions,
                    VELOCITY_LAYER_ASSUMPTION_FAILED,
                    jnp.where(
                        provenance_match, VELOCITY_SUCCESS, VELOCITY_PROVENANCE_MISMATCH
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = KineticRNAVelocityEvidence(
        finite_counts,
        nonnegative_counts,
        finite_rates,
        nonnegative_rates,
        assumptions,
        provenance_match,
        jnp.asarray(plan.parameter_origin == "training_fitted"),
        jnp.asarray(plan.steady_state_assumption),
        plan.spliced_layer_name,
        plan.unspliced_layer_name,
    )
    return KineticRNAVelocityResult(
        unspliced_velocity,
        spliced_velocity,
        valid,
        status,
        evidence,
        method_contract if method_contract is not None else _velocity_contract(),
        "kinetic_model_estimate",
    )


__all__ = [
    "VELOCITY_INVALID_RATE",
    "VELOCITY_LAYER_ASSUMPTION_FAILED",
    "VELOCITY_NEGATIVE_COUNT",
    "VELOCITY_NONFINITE",
    "VELOCITY_PROVENANCE_MISMATCH",
    "VELOCITY_SUCCESS",
    "KineticRNAVelocityEvidence",
    "KineticRNAVelocityPlan",
    "KineticRNAVelocityResult",
    "kinetic_rna_velocity",
    "velocity_status_name",
]
