#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact-decimal polynomial matrix programs and finite sampled audits."""

from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


def _decimal(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} decimal value must be non-empty.")
    try:
        parsed = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{name} contains an invalid decimal value.") from error
    if not parsed.is_finite():
        raise ValueError(f"{name} decimal values must be finite.")
    return text


def _decimal_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_decimal(value, name=name) for value in values)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def decimal_coefficients(
    values: ArrayLike, /, *, significant_digits: int
) -> tuple[str, ...]:
    """Convert one finite host vector to declared decimal strings."""
    digits = int(significant_digits)
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("values must be one nonempty finite vector.")
    if digits < 2:
        raise ValueError("significant_digits must be at least two.")
    return tuple(format(float(value), f".{digits}g") for value in raw)


class DampedRationalPrefactor(StrictModule):
    """SDPB damped-rational prefactor with exact decimal strings."""

    base: str = eqx.field(static=True)
    constant: str = eqx.field(static=True)
    poles: tuple[str, ...] = eqx.field(static=True)
    prefactor_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: str,
        constant: str,
        poles: Sequence[str] = (),
        /,
    ):
        base_value = _decimal(base, name="prefactor base")
        constant_value = _decimal(constant, name="prefactor constant")
        pole_values = tuple(_decimal(value, name="prefactor pole") for value in poles)
        if Decimal(base_value) <= 0 or Decimal(base_value) > 1:
            raise ValueError("A damped-rational base must lie in (0, 1].")
        if Decimal(constant_value) <= 0:
            raise ValueError("A damped-rational constant must be positive.")
        self.base = base_value
        self.constant = constant_value
        self.poles = pole_values
        self.prefactor_id = canonical_fingerprint(
            {
                "kind": "damped-rational-prefactor",
                "base": base_value,
                "constant": constant_value,
                "poles": pole_values,
            }
        )

    def to_record(self) -> dict[str, object]:
        return {
            "base": self.base,
            "constant": self.constant,
            "poles": list(self.poles),
        }


class PolynomialMatrixBlock(StrictModule):
    """One symmetric matrix whose entries are functional-indexed polynomials."""

    prefactor: DampedRationalPrefactor
    polynomials: tuple[tuple[tuple[tuple[str, ...], ...], ...], ...] = eqx.field(
        static=True
    )
    sample_points: tuple[str, ...] = eqx.field(static=True)
    sample_scalings: tuple[str, ...] = eqx.field(static=True)
    reduced_sample_scalings: tuple[str, ...] = eqx.field(static=True)
    bilinear_basis: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    matrix_dimension: int = eqx.field(static=True)
    functional_count: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        prefactor: DampedRationalPrefactor,
        polynomials: Sequence[Sequence[Sequence[Sequence[str]]]],
        /,
        *,
        sample_points: Sequence[str] = (),
        sample_scalings: Sequence[str] = (),
        reduced_sample_scalings: Sequence[str] = (),
        bilinear_basis: Sequence[Sequence[str]] = (),
    ):
        if not isinstance(prefactor, DampedRationalPrefactor):
            raise TypeError("prefactor must be DampedRationalPrefactor.")
        raw_rows = tuple(tuple(row) for row in polynomials)
        if not raw_rows or any(len(row) != len(raw_rows) for row in raw_rows):
            raise ValueError("Polynomial matrix blocks must be nonempty and square.")
        converted = tuple(
            tuple(
                tuple(
                    _decimal_tuple(coefficients, name="polynomial coefficients")
                    for coefficients in entry
                )
                for entry in row
            )
            for row in raw_rows
        )
        functional_count = len(converted[0][0])
        if functional_count < 1 or any(
            len(entry) != functional_count for row in converted for entry in row
        ):
            raise ValueError("Every matrix entry must use the same functional count.")
        for row in range(len(converted)):
            for column in range(row):
                left = tuple(
                    tuple(Decimal(value) for value in polynomial)
                    for polynomial in converted[row][column]
                )
                right = tuple(
                    tuple(Decimal(value) for value in polynomial)
                    for polynomial in converted[column][row]
                )
                if left != right:
                    raise ValueError("Polynomial matrix blocks must be symmetric.")
        points = tuple(_decimal(value, name="sample point") for value in sample_points)
        scalings = tuple(
            _decimal(value, name="sample scaling") for value in sample_scalings
        )
        reduced = tuple(
            _decimal(value, name="reduced sample scaling")
            for value in reduced_sample_scalings
        )
        if points and (len(scalings) != len(points) or len(reduced) != len(points)):
            raise ValueError(
                "Sample points and both scaling arrays must have equal length."
            )
        if not points and (scalings or reduced):
            raise ValueError("Sample scalings require sample points.")
        basis = tuple(
            tuple(_decimal(value, name="bilinear basis") for value in row)
            for row in bilinear_basis
        )
        if basis and points and any(len(row) != len(points) for row in basis):
            raise ValueError("Bilinear basis rows must match the sample-point count.")
        maximum_degree = max(
            len(polynomial) - 1
            for row in converted
            for entry in row
            for polynomial in entry
        )
        self.prefactor = prefactor
        self.polynomials = converted
        self.sample_points = points
        self.sample_scalings = scalings
        self.reduced_sample_scalings = reduced
        self.bilinear_basis = basis
        self.matrix_dimension = len(converted)
        self.functional_count = functional_count
        self.maximum_degree = maximum_degree
        self.block_id = canonical_fingerprint(
            {
                "kind": "polynomial-matrix-block",
                "prefactor": prefactor.prefactor_id,
                "polynomials": converted,
                "sample_points": points,
                "sample_scalings": scalings,
                "reduced_sample_scalings": reduced,
                "bilinear_basis": basis,
            }
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "DampedRational": self.prefactor.to_record(),
            "polynomials": [
                [[list(polynomial) for polynomial in entry] for entry in row]
                for row in self.polynomials
            ],
        }
        if self.sample_points:
            record["samplePoints"] = list(self.sample_points)
            record["sampleScalings"] = list(self.sample_scalings)
            record["reducedSampleScalings"] = list(self.reduced_sample_scalings)
        if self.bilinear_basis:
            record["bilinearBasis"] = [list(row) for row in self.bilinear_basis]
        return record


class ConformalPolynomialMatrixProgram(StrictModule):
    """Canonical exact-decimal PMP prior to external preprocessing."""

    blocks: tuple[PolynomialMatrixBlock, ...]
    objective: tuple[str, ...] = eqx.field(static=True)
    normalization: tuple[str, ...] = eqx.field(static=True)
    frontend_id: str = eqx.field(static=True)
    frontend_precision_bits: int = eqx.field(static=True)
    functional_count: int = eqx.field(static=True)
    maximum_decimal_bytes: int = eqx.field(static=True)
    pmp_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: Sequence[str],
        normalization: Sequence[str],
        blocks: Sequence[PolynomialMatrixBlock],
        /,
        *,
        frontend_id: str,
        frontend_precision_bits: int,
        maximum_decimal_bytes: int = 1 << 30,
    ):
        objective_values = _decimal_tuple(objective, name="objective")
        normalization_values = _decimal_tuple(normalization, name="normalization")
        block_values = tuple(blocks)
        frontend = str(frontend_id)
        precision = int(frontend_precision_bits)
        maximum = int(maximum_decimal_bytes)
        if len(objective_values) != len(normalization_values):
            raise ValueError("Objective and normalization dimensions must match.")
        if not block_values or any(
            not isinstance(value, PolynomialMatrixBlock) for value in block_values
        ):
            raise TypeError("At least one PolynomialMatrixBlock is required.")
        count = len(objective_values)
        if any(value.functional_count != count for value in block_values):
            raise ValueError("Every PMP block must use the objective functional count.")
        if not frontend or precision < 2 or maximum < 1:
            raise ValueError("Frontend identity, precision, and byte limit are required.")
        if all(Decimal(value) == 0 for value in normalization_values):
            raise ValueError("PMP normalization must be nonzero.")
        record = {
            "objective": list(objective_values),
            "normalization": list(normalization_values),
            "PositiveMatrixWithPrefactorArray": [
                value.to_record() for value in block_values
            ],
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > maximum:
            raise ValueError("Serialized PMP exceeds maximum_decimal_bytes.")
        self.blocks = block_values
        self.objective = objective_values
        self.normalization = normalization_values
        self.frontend_id = frontend
        self.frontend_precision_bits = precision
        self.functional_count = count
        self.maximum_decimal_bytes = maximum
        self.pmp_id = canonical_fingerprint(
            {
                "kind": "conformal-polynomial-matrix-program",
                "record": record,
                "frontend_id": frontend,
                "frontend_precision_bits": precision,
                "maximum_decimal_bytes": maximum,
            }
        )

    def to_record(self) -> dict[str, object]:
        return {
            "objective": list(self.objective),
            "normalization": list(self.normalization),
            "PositiveMatrixWithPrefactorArray": [
                value.to_record() for value in self.blocks
            ],
        }

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_record(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("ascii")


class PMPSampledAuditEvidence(StrictModule):
    functional: Array
    sample_points: Array
    minimum_eigenvalues: Array
    normalization_value: Array
    normalization_residual: Array
    finite: Array
    positive_semidefinite: Array
    accepted: Array
    pmp_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _evaluate_polynomial(coefficients: tuple[str, ...], point: float, /) -> float:
    value = 0.0
    for coefficient in reversed(coefficients):
        value = value * point + float(Decimal(coefficient))
    return value


def audit_pmp_samples(
    program: ConformalPolynomialMatrixProgram,
    functional: ArrayLike,
    sample_points: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
) -> PMPSampledAuditEvidence:
    """Evaluate finite sample PSD constraints; this is not continuum positivity."""
    if not isinstance(program, ConformalPolynomialMatrixProgram):
        raise TypeError("program must be ConformalPolynomialMatrixProgram.")
    alpha = np.asarray(functional, dtype=float)
    points = np.asarray(sample_points, dtype=float)
    tolerance_value = float(tolerance)
    if alpha.shape != (program.functional_count,) or not np.all(np.isfinite(alpha)):
        raise ValueError("functional must be one finite PMP functional vector.")
    if points.ndim != 1 or points.size == 0 or not np.all(np.isfinite(points)):
        raise ValueError("sample_points must be one nonempty finite vector.")
    if np.any(points < 0.0):
        raise ValueError("PMP sample points must lie in x >= 0.")
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")
    minimum = np.empty((len(program.blocks), points.size), dtype=float)
    for block_index, block in enumerate(program.blocks):
        for point_index, point in enumerate(points):
            matrix = np.empty(
                (block.matrix_dimension, block.matrix_dimension), dtype=float
            )
            for row in range(block.matrix_dimension):
                for column in range(block.matrix_dimension):
                    matrix[row, column] = sum(
                        alpha[functional_index]
                        * _evaluate_polynomial(polynomial, float(point))
                        for functional_index, polynomial in enumerate(
                            block.polynomials[row][column]
                        )
                    )
            minimum[block_index, point_index] = float(np.min(np.linalg.eigvalsh(matrix)))
    normalization = float(
        np.dot(
            alpha,
            np.asarray([float(Decimal(value)) for value in program.normalization]),
        )
    )
    residual = abs(normalization - 1.0)
    finite = bool(np.all(np.isfinite(minimum)) and np.isfinite(normalization))
    psd = finite and bool(np.all(minimum >= -tolerance_value))
    accepted = psd and residual <= tolerance_value
    return PMPSampledAuditEvidence(
        functional=jnp.asarray(alpha),
        sample_points=jnp.asarray(points),
        minimum_eigenvalues=jnp.asarray(minimum),
        normalization_value=jnp.asarray(normalization),
        normalization_residual=jnp.asarray(residual),
        finite=jnp.asarray(finite),
        positive_semidefinite=jnp.asarray(psd),
        accepted=jnp.asarray(accepted),
        pmp_id=program.pmp_id,
        claim="finite-sampled-polynomial-matrix-audit-not-continuum-positivity",
    )


__all__ = [
    "ConformalPolynomialMatrixProgram",
    "DampedRationalPrefactor",
    "PMPSampledAuditEvidence",
    "PolynomialMatrixBlock",
    "audit_pmp_samples",
    "decimal_coefficients",
]
