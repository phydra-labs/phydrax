#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import DifferentiationContract


class GeophysicalCapabilityEvidence(StrictModule, NonTrainableState):
    """Exact physical support and exclusions for one prepared model family."""

    model: str = eqx.field(static=True)
    dimensions: tuple[int, ...] = eqx.field(static=True)
    field_equations: tuple[str, ...] = eqx.field(static=True)
    source_models: tuple[str, ...] = eqx.field(static=True)
    receiver_models: tuple[str, ...] = eqx.field(static=True)
    boundary_models: tuple[str, ...] = eqx.field(static=True)
    material_models: tuple[str, ...] = eqx.field(static=True)
    limitations: tuple[str, ...] = eqx.field(static=True)
    differentiation: DifferentiationContract
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: str,
        dimensions: tuple[int, ...],
        /,
        *,
        field_equations: tuple[str, ...],
        source_models: tuple[str, ...],
        receiver_models: tuple[str, ...],
        boundary_models: tuple[str, ...],
        material_models: tuple[str, ...],
        limitations: tuple[str, ...],
        differentiation: DifferentiationContract,
    ):
        values = (
            str(model).strip(),
            tuple(int(value) for value in dimensions),
            tuple(str(value).strip() for value in field_equations),
            tuple(str(value).strip() for value in source_models),
            tuple(str(value).strip() for value in receiver_models),
            tuple(str(value).strip() for value in boundary_models),
            tuple(str(value).strip() for value in material_models),
            tuple(str(value).strip() for value in limitations),
        )
        if (
            not values[0]
            or not values[1]
            or any(value not in (1, 2, 3) for value in values[1])
            or any(not group or any(not item for item in group) for group in values[2:])
            or not isinstance(differentiation, DifferentiationContract)
        ):
            raise ValueError(
                "Geophysical capability evidence must be explicit and nonempty."
            )
        (
            self.model,
            self.dimensions,
            self.field_equations,
            self.source_models,
            self.receiver_models,
            self.boundary_models,
            self.material_models,
            self.limitations,
        ) = values
        self.differentiation = differentiation
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "geophysical-capability-evidence",
                "model": self.model,
                "dimensions": self.dimensions,
                "field_equations": self.field_equations,
                "source_models": self.source_models,
                "receiver_models": self.receiver_models,
                "boundary_models": self.boundary_models,
                "material_models": self.material_models,
                "limitations": self.limitations,
                "differentiation": differentiation.contract_id,
            }
        )


class GeophysicalResourceEstimate(StrictModule, NonTrainableState):
    """Static retained, temporary, checkpoint, and observation memory estimate."""

    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    checkpoint_bytes: int = eqx.field(static=True)
    observation_bytes: int = eqx.field(static=True)
    source_batch_size: int = eqx.field(static=True)
    maximum_bytes: int | None = eqx.field(static=True)
    total_bytes: int = eqx.field(static=True)
    within_budget: bool = eqx.field(static=True)
    estimate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        retained_bytes: int,
        workspace_bytes: int,
        checkpoint_bytes: int,
        observation_bytes: int,
        source_batch_size: int,
        maximum_bytes: int | None = None,
    ):
        components = tuple(
            int(value)
            for value in (
                retained_bytes,
                workspace_bytes,
                checkpoint_bytes,
                observation_bytes,
            )
        )
        batch = int(source_batch_size)
        budget = None if maximum_bytes is None else int(maximum_bytes)
        if any(value < 0 for value in components) or batch <= 0:
            raise ValueError(
                "Resource components must be nonnegative and source batch positive."
            )
        if budget is not None and budget <= 0:
            raise ValueError("Resource budget must be positive or None.")
        total = sum(components)
        (
            self.retained_bytes,
            self.workspace_bytes,
            self.checkpoint_bytes,
            self.observation_bytes,
        ) = components
        self.source_batch_size = batch
        self.maximum_bytes = budget
        self.total_bytes = total
        self.within_budget = budget is None or total <= budget
        self.estimate_id = canonical_fingerprint(
            {
                "kind": "geophysical-resource-estimate",
                "components": components,
                "source_batch_size": batch,
                "maximum_bytes": budget,
            }
        )

    def require_budget(self) -> None:
        if not self.within_budget:
            raise MemoryError(
                f"Estimated geophysical allocation {self.total_bytes} exceeds "
                f"the declared budget {self.maximum_bytes}."
            )


__all__ = ["GeophysicalCapabilityEvidence", "GeophysicalResourceEstimate"]
