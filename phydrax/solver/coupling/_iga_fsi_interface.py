#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import FieldTransfer


def _identifier(value: str, role: str, /) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{role} must be non-empty.")
    return result


def _tree_subtract(left: Any, right: Any, /) -> Any:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _norm(space, value: Any, /) -> Array:
    squared = jnp.real(space.inner(value, value))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def _threshold(absolute: float, relative: float, scale: Array, /) -> Array:
    return (
        jnp.asarray(absolute, dtype=scale.dtype)
        + jnp.asarray(relative, dtype=scale.dtype) * scale
    )


class InterfaceTransferTolerance(StrictModule, NonTrainableState):
    """Physical error budgets for a bidirectional interface transfer."""

    power_absolute: float = eqx.field(static=True)
    power_relative: float = eqx.field(static=True)
    conservation_absolute: float = eqx.field(static=True)
    conservation_relative: float = eqx.field(static=True)
    error_absolute: float = eqx.field(static=True)
    error_relative: float = eqx.field(static=True)
    derivative_absolute: float = eqx.field(static=True)
    derivative_relative: float = eqx.field(static=True)
    tolerance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        power_absolute: float,
        power_relative: float = 0.0,
        conservation_absolute: float,
        conservation_relative: float = 0.0,
        error_absolute: float,
        error_relative: float = 0.0,
        derivative_absolute: float,
        derivative_relative: float = 0.0,
    ):
        values = tuple(
            float(value)
            for value in (
                power_absolute,
                power_relative,
                conservation_absolute,
                conservation_relative,
                error_absolute,
                error_relative,
                derivative_absolute,
                derivative_relative,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Interface transfer tolerances must be finite and non-negative."
            )
        pairs = tuple(zip(values[::2], values[1::2], strict=True))
        if any(absolute == 0.0 and relative == 0.0 for absolute, relative in pairs):
            raise ValueError(
                "Every interface transfer error class needs a positive tolerance."
            )
        (
            self.power_absolute,
            self.power_relative,
            self.conservation_absolute,
            self.conservation_relative,
            self.error_absolute,
            self.error_relative,
            self.derivative_absolute,
            self.derivative_relative,
        ) = values
        self.tolerance_id = canonical_fingerprint(
            {
                "kind": "iga-interface-transfer-tolerance",
                "power": list(values[0:2]),
                "conservation": list(values[2:4]),
                "error": list(values[4:6]),
                "derivative": list(values[6:8]),
            }
        )


class InterfaceTransferProbe(StrictModule):
    """Physical common-quadrature references for one paired transfer check."""

    source_value: Any
    target_effort: Any
    source_measure_functional: Any
    target_measure_functional: Any
    source_direction: Any
    target_direction: Any
    forward_reference: Any
    reverse_reference: Any
    forward_derivative_reference: Any
    reverse_derivative_reference: Any
    probe_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: FieldTransfer,
        source_value: Any,
        target_effort: Any,
        source_measure_functional: Any,
        target_measure_functional: Any,
        source_direction: Any,
        target_direction: Any,
        forward_reference: Any,
        reverse_reference: Any,
        forward_derivative_reference: Any,
        reverse_derivative_reference: Any,
        /,
        *,
        probe_id: str,
    ):
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("transfer must be a FieldTransfer.")
        source = transfer.source.vector_space
        target = transfer.target.vector_space
        self.source_value = source.validate(source_value)
        self.target_effort = target.validate(target_effort)
        self.source_measure_functional = source.validate(source_measure_functional)
        self.target_measure_functional = target.validate(target_measure_functional)
        self.source_direction = source.validate(source_direction)
        self.target_direction = target.validate(target_direction)
        self.forward_reference = target.validate(forward_reference)
        self.reverse_reference = source.validate(reverse_reference)
        self.forward_derivative_reference = target.validate(forward_derivative_reference)
        self.reverse_derivative_reference = source.validate(reverse_derivative_reference)
        self.probe_id = _identifier(probe_id, "Interface transfer probe_id")


class InterfaceTransferEvidence(StrictModule):
    """Power, conservation, reference, and derivative transfer evidence."""

    forward_power: Array
    reverse_power: Array
    power_error: Array
    forward_conservation_error: Array
    reverse_conservation_error: Array
    forward_error: Array
    reverse_error: Array
    forward_derivative_error: Array
    reverse_derivative_error: Array
    power_threshold: Array
    forward_conservation_threshold: Array
    reverse_conservation_threshold: Array
    forward_error_threshold: Array
    reverse_error_threshold: Array
    forward_derivative_threshold: Array
    reverse_derivative_threshold: Array
    finite: Array
    certified: Array


class InterfaceTransferCertificate(StrictModule, NonTrainableState):
    """Identity-bound physical qualification of one paired IGA FSI transfer."""

    transfer: FieldTransfer
    evidence: InterfaceTransferEvidence
    tolerance: InterfaceTransferTolerance
    source_plan_id: str = eqx.field(static=True)
    target_plan_id: str = eqx.field(static=True)
    source_numeric_revision_id: str = eqx.field(static=True)
    target_numeric_revision_id: str = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)
    orientation_id: str = eqx.field(static=True)
    orientation_sign: int = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    common_quadrature_id: str = eqx.field(static=True)
    probe_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: FieldTransfer,
        evidence: InterfaceTransferEvidence,
        tolerance: InterfaceTransferTolerance,
        /,
        *,
        source_plan_id: str,
        target_plan_id: str,
        source_numeric_revision_id: str,
        target_numeric_revision_id: str,
        interface_id: str,
        orientation_id: str,
        orientation_sign: int,
        frame_id: str,
        measure_id: str,
        common_quadrature_id: str,
        probe_id: str,
    ):
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("transfer must be a FieldTransfer.")
        if not isinstance(evidence, InterfaceTransferEvidence):
            raise TypeError("evidence must be InterfaceTransferEvidence.")
        if not isinstance(tolerance, InterfaceTransferTolerance):
            raise TypeError("tolerance must be InterfaceTransferTolerance.")
        sign = int(orientation_sign)
        if sign not in (-1, 1):
            raise ValueError("Interface orientation_sign must be -1 or +1.")
        self.transfer = transfer
        self.evidence = evidence
        self.tolerance = tolerance
        self.source_plan_id = _identifier(source_plan_id, "source_plan_id")
        self.target_plan_id = _identifier(target_plan_id, "target_plan_id")
        self.source_numeric_revision_id = _identifier(
            source_numeric_revision_id, "source_numeric_revision_id"
        )
        self.target_numeric_revision_id = _identifier(
            target_numeric_revision_id, "target_numeric_revision_id"
        )
        self.interface_id = _identifier(interface_id, "interface_id")
        self.orientation_id = _identifier(orientation_id, "orientation_id")
        self.orientation_sign = sign
        self.frame_id = _identifier(frame_id, "frame_id")
        self.measure_id = _identifier(measure_id, "measure_id")
        self.common_quadrature_id = _identifier(
            common_quadrature_id, "common_quadrature_id"
        )
        self.probe_id = _identifier(probe_id, "probe_id")
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "iga-fsi-interface-transfer-certificate",
                "transfer": transfer.transfer_id,
                "source_plan": self.source_plan_id,
                "target_plan": self.target_plan_id,
                "source_numeric_revision": self.source_numeric_revision_id,
                "target_numeric_revision": self.target_numeric_revision_id,
                "interface": self.interface_id,
                "orientation": self.orientation_id,
                "orientation_sign": sign,
                "frame": self.frame_id,
                "measure": self.measure_id,
                "common_quadrature": self.common_quadrature_id,
                "probe": self.probe_id,
                "tolerance": tolerance.tolerance_id,
            }
        )

    def require_certified(self, /) -> None:
        if not bool(self.evidence.certified):
            raise ValueError(
                f"IGA interface transfer certificate {self.certificate_id!r} failed."
            )


def certify_interface_transfer(
    transfer: FieldTransfer,
    probe: InterfaceTransferProbe,
    tolerance: InterfaceTransferTolerance,
    /,
    *,
    source_plan_id: str,
    target_plan_id: str,
    source_numeric_revision_id: str,
    target_numeric_revision_id: str,
    interface_id: str,
    orientation_id: str,
    orientation_sign: int,
    frame_id: str,
    measure_id: str,
    common_quadrature_id: str,
) -> InterfaceTransferCertificate:
    """Evaluate a paired transfer against independent common-quadrature references."""

    if not isinstance(transfer, FieldTransfer):
        raise TypeError("transfer must be a FieldTransfer.")
    if not isinstance(probe, InterfaceTransferProbe):
        raise TypeError("probe must be InterfaceTransferProbe.")
    if not isinstance(tolerance, InterfaceTransferTolerance):
        raise TypeError("tolerance must be InterfaceTransferTolerance.")
    if transfer.adjoint_operator is None or not transfer.properties.adjoint_paired:
        raise ValueError("IGA FSI transfer certification requires a paired adjoint.")
    source = transfer.source.vector_space
    target = transfer.target.vector_space
    forward = transfer.operator.mv(probe.source_value)
    reverse = transfer.adjoint_operator.mv(probe.target_effort)
    forward_power = jnp.real(target.inner(forward, probe.target_effort))
    reverse_power = jnp.real(source.inner(probe.source_value, reverse))
    sign = int(orientation_sign)
    if sign not in (-1, 1):
        raise ValueError("Interface orientation_sign must be -1 or +1.")
    power_error = jnp.abs(forward_power + sign * reverse_power)
    source_integral = jnp.real(
        source.inner(probe.source_measure_functional, probe.source_value)
    )
    target_integral = jnp.real(target.inner(probe.target_measure_functional, forward))
    forward_conservation_error = jnp.abs(target_integral - source_integral)
    target_effort_integral = jnp.real(
        target.inner(probe.target_measure_functional, probe.target_effort)
    )
    source_effort_integral = jnp.real(
        source.inner(probe.source_measure_functional, reverse)
    )
    reverse_conservation_error = jnp.abs(
        target_effort_integral + sign * source_effort_integral
    )
    forward_error = _norm(target, _tree_subtract(forward, probe.forward_reference))
    reverse_error = _norm(source, _tree_subtract(reverse, probe.reverse_reference))
    _, forward_derivative = jax.jvp(
        transfer.operator.mv,
        (probe.source_value,),
        (probe.source_direction,),
    )
    _, reverse_derivative = jax.jvp(
        transfer.adjoint_operator.mv,
        (probe.target_effort,),
        (probe.target_direction,),
    )
    forward_derivative_error = _norm(
        target,
        _tree_subtract(forward_derivative, probe.forward_derivative_reference),
    )
    reverse_derivative_error = _norm(
        source,
        _tree_subtract(reverse_derivative, probe.reverse_derivative_reference),
    )

    power_scale = jnp.maximum(jnp.abs(forward_power), jnp.abs(reverse_power))
    forward_conservation_scale = jnp.maximum(
        jnp.abs(source_integral), jnp.abs(target_integral)
    )
    reverse_conservation_scale = jnp.maximum(
        jnp.abs(target_effort_integral), jnp.abs(source_effort_integral)
    )
    forward_error_scale = jnp.maximum(
        _norm(target, forward), _norm(target, probe.forward_reference)
    )
    reverse_error_scale = jnp.maximum(
        _norm(source, reverse), _norm(source, probe.reverse_reference)
    )
    forward_derivative_scale = jnp.maximum(
        _norm(target, forward_derivative),
        _norm(target, probe.forward_derivative_reference),
    )
    reverse_derivative_scale = jnp.maximum(
        _norm(source, reverse_derivative),
        _norm(source, probe.reverse_derivative_reference),
    )
    thresholds = (
        _threshold(tolerance.power_absolute, tolerance.power_relative, power_scale),
        _threshold(
            tolerance.conservation_absolute,
            tolerance.conservation_relative,
            forward_conservation_scale,
        ),
        _threshold(
            tolerance.conservation_absolute,
            tolerance.conservation_relative,
            reverse_conservation_scale,
        ),
        _threshold(
            tolerance.error_absolute,
            tolerance.error_relative,
            forward_error_scale,
        ),
        _threshold(
            tolerance.error_absolute,
            tolerance.error_relative,
            reverse_error_scale,
        ),
        _threshold(
            tolerance.derivative_absolute,
            tolerance.derivative_relative,
            forward_derivative_scale,
        ),
        _threshold(
            tolerance.derivative_absolute,
            tolerance.derivative_relative,
            reverse_derivative_scale,
        ),
    )
    errors = (
        power_error,
        forward_conservation_error,
        reverse_conservation_error,
        forward_error,
        reverse_error,
        forward_derivative_error,
        reverse_derivative_error,
    )
    finite = jnp.asarray(True)
    for value in (*errors, *thresholds, forward_power, reverse_power):
        finite = finite & jnp.all(jnp.isfinite(value))
    certified = finite
    for error, threshold in zip(errors, thresholds, strict=True):
        certified = certified & (error <= threshold)
    evidence = InterfaceTransferEvidence(
        forward_power,
        reverse_power,
        power_error,
        forward_conservation_error,
        reverse_conservation_error,
        forward_error,
        reverse_error,
        forward_derivative_error,
        reverse_derivative_error,
        *thresholds,
        finite,
        certified,
    )
    return InterfaceTransferCertificate(
        transfer,
        evidence,
        tolerance,
        source_plan_id=source_plan_id,
        target_plan_id=target_plan_id,
        source_numeric_revision_id=source_numeric_revision_id,
        target_numeric_revision_id=target_numeric_revision_id,
        interface_id=interface_id,
        orientation_id=orientation_id,
        orientation_sign=sign,
        frame_id=frame_id,
        measure_id=measure_id,
        common_quadrature_id=common_quadrature_id,
        probe_id=probe.probe_id,
    )


__all__ = [
    "InterfaceTransferCertificate",
    "InterfaceTransferEvidence",
    "InterfaceTransferProbe",
    "InterfaceTransferTolerance",
    "certify_interface_transfer",
]
