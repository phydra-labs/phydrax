#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


DiagnosticSeverity: TypeAlias = Literal["info", "warning", "error", "fatal"]


class Diagnostic(StrictModule, NonTrainableState):
    """One structured, content-addressed diagnostic emitted by any provider."""

    code: str = eqx.field(static=True)
    severity: DiagnosticSeverity = eqx.field(static=True)
    phase: str = eqx.field(static=True)
    message: str = eqx.field(static=True)
    entity_ids: tuple[str, ...] = eqx.field(static=True)
    parametric_point: Array | None
    physical_point: Array | None
    value: Array | None
    tolerance: Array | None
    unit: str | None = eqx.field(static=True)
    remediation: str | None = eqx.field(static=True)
    parent_code: str | None = eqx.field(static=True)
    run_id: str | None = eqx.field(static=True)
    diagnostic_id: str = eqx.field(static=True)

    def __init__(
        self,
        code: str,
        severity: DiagnosticSeverity,
        phase: str,
        message: str,
        /,
        *,
        entity_ids: Sequence[str] = (),
        parametric_point: ArrayLike | None = None,
        physical_point: ArrayLike | None = None,
        value: ArrayLike | None = None,
        tolerance: ArrayLike | None = None,
        unit: str | None = None,
        remediation: str | None = None,
        parent_code: str | None = None,
        run_id: str | None = None,
    ):
        code_ = str(code).strip()
        severity_ = str(severity).strip()
        phase_ = str(phase).strip()
        message_ = str(message).strip()
        entities = tuple(str(entity_id).strip() for entity_id in entity_ids)
        parametric = None if parametric_point is None else jnp.asarray(parametric_point)
        physical = None if physical_point is None else jnp.asarray(physical_point)
        value_ = None if value is None else jnp.asarray(value)
        tolerance_ = None if tolerance is None else jnp.asarray(tolerance)
        unit_ = _optional_identifier(unit, "unit")
        remediation_ = _optional_text(remediation, "remediation")
        parent_ = _optional_identifier(parent_code, "parent_code")
        run_ = _optional_identifier(run_id, "run_id")
        if not code_ or not phase_ or not message_:
            raise ValueError("Diagnostic code, phase, and message must be non-empty.")
        if severity_ not in ("info", "warning", "error", "fatal"):
            raise ValueError(
                "Diagnostic severity must be info, warning, error, or fatal."
            )
        if any(not entity_id for entity_id in entities) or len(set(entities)) != len(
            entities
        ):
            raise ValueError("Diagnostic entity IDs must be non-empty and unique.")
        for name, point in (
            ("parametric_point", parametric),
            ("physical_point", physical),
        ):
            if point is not None and (point.ndim != 1 or point.size == 0):
                raise ValueError(f"Diagnostic {name} must be a non-empty vector.")
        for name, scalar in (("value", value_), ("tolerance", tolerance_)):
            if scalar is not None and scalar.shape != ():
                raise ValueError(f"Diagnostic {name} must be scalar.")
        if tolerance_ is not None and bool(np.asarray(tolerance_) < 0):
            raise ValueError("Diagnostic tolerance must be nonnegative.")

        self.code = code_
        self.severity = severity_  # type: ignore[assignment]
        self.phase = phase_
        self.message = message_
        self.entity_ids = entities
        self.parametric_point = parametric
        self.physical_point = physical
        self.value = value_
        self.tolerance = tolerance_
        self.unit = unit_
        self.remediation = remediation_
        self.parent_code = parent_
        self.run_id = run_
        self.diagnostic_id = canonical_fingerprint(
            {
                "kind": "diagnostic",
                "code": code_,
                "severity": severity_,
                "phase": phase_,
                "entity_ids": list(entities),
                "parametric_point": _array_identity(parametric),
                "physical_point": _array_identity(physical),
                "value": _array_identity(value_),
                "tolerance": _array_identity(tolerance_),
                "unit": unit_,
                "parent_code": parent_,
                "run_id": run_,
            }
        )


class DiagnosticError(RuntimeError):
    """Failure carrying complete structured diagnostics across an API boundary."""

    diagnostics: tuple[Diagnostic, ...]

    def __init__(self, diagnostics: Sequence[Diagnostic], /):
        values = tuple(diagnostics)
        if not values or not all(isinstance(value, Diagnostic) for value in values):
            raise TypeError("DiagnosticError requires one or more Diagnostic values.")
        self.diagnostics = values
        codes = ", ".join(value.code for value in values)
        super().__init__(f"Diagnostics reported failure: {codes}")


def _array_identity(value: Array | None, /) -> dict[str, object] | None:
    if value is None:
        return None
    return array_tree_fingerprint(np.asarray(value))


def _optional_identifier(value: str | None, name: str, /) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"Diagnostic {name} must be non-empty when provided.")
    return normalized


def _optional_text(value: str | None, name: str, /) -> str | None:
    return _optional_identifier(value, name)


__all__ = ["Diagnostic", "DiagnosticError", "DiagnosticSeverity"]
