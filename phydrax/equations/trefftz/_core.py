#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import AbstractArrayModel, MODEL_CONSTRUCTION_CERTIFICATE_KEYS
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


TrialEquationFamily = Literal[
    "laplace",
    "polyharmonic",
    "helmholtz",
    "linear-elasticity",
    "dirac",
]
TrialExactness = Literal["algebraic"]
TrialCoverage = Literal["finite-subspace", "finite-parametric-family"]
TrialValidityRegion = Literal["all-space", "off-singular-support"]
TRIAL_SPACE_CERTIFICATE_KEY = next(iter(MODEL_CONSTRUCTION_CERTIFICATE_KEYS))
TRIAL_SPACE_REPRESENTATION_KEY = "trial_space_runtime_representation"


class SimilarityNormalization(StrictModule, NonTrainableState):
    """Fixed translation and scalar dilation preserving Euclidean PDE structure."""

    center: Array
    scale: Array
    normalization_id: str = eqx.field(static=True)

    def __init__(self, center: ArrayLike, scale: float = 1.0, /):
        center_host = np.asarray(center, dtype=float)
        scale_ = float(scale)
        if center_host.ndim != 1 or center_host.size < 2:
            raise ValueError(
                "SimilarityNormalization center must be a vector of size at least two."
            )
        if not np.all(np.isfinite(center_host)):
            raise ValueError("SimilarityNormalization center must be finite.")
        if not math.isfinite(scale_) or scale_ <= 0.0:
            raise ValueError("SimilarityNormalization scale must be finite and positive.")
        self.center = jnp.asarray(center_host, dtype=float)
        self.scale = jnp.asarray(scale_, dtype=float).reshape(())
        self.normalization_id = canonical_fingerprint(
            {
                "kind": "euclidean-similarity-normalization-v1",
                "center": center_host.tolist(),
                "scale": scale_,
            }
        )

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])

    def __call__(self, point: ArrayLike, /) -> Array:
        value = jnp.asarray(point)
        if value.shape != (self.dimension,):
            raise ValueError(
                f"SimilarityNormalization expected shape ({self.dimension},); got {value.shape}."
            )
        return (value - self.center.astype(value.dtype)) / self.scale.astype(value.dtype)


class TrefftzResourceBudget(StrictModule, NonTrainableState):
    """Hard pre-allocation limits for one exact trial-space basis."""

    maximum_rank: int = eqx.field(static=True)
    maximum_monomials: int = eqx.field(static=True)
    maximum_basis_entries: int = eqx.field(static=True)
    maximum_basis_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_rank: int = 20_000,
        maximum_monomials: int = 100_000,
        maximum_basis_entries: int = 10_000_000,
        maximum_basis_bytes: int = 256 * 1024**2,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_rank,
                maximum_monomials,
                maximum_basis_entries,
                maximum_basis_bytes,
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError("Trefftz resource limits must be positive.")
        (
            self.maximum_rank,
            self.maximum_monomials,
            self.maximum_basis_entries,
            self.maximum_basis_bytes,
        ) = values

    def check(
        self,
        *,
        rank: int,
        monomials: int,
        basis_entries: int,
        basis_bytes: int,
    ) -> "TrefftzResourceEvidence":
        requested = {
            "rank": int(rank),
            "monomials": int(monomials),
            "basis_entries": int(basis_entries),
            "basis_bytes": int(basis_bytes),
        }
        limits = {
            "rank": self.maximum_rank,
            "monomials": self.maximum_monomials,
            "basis_entries": self.maximum_basis_entries,
            "basis_bytes": self.maximum_basis_bytes,
        }
        exceeded = tuple(name for name in requested if requested[name] > limits[name])
        if exceeded:
            details = ", ".join(
                f"{name}={requested[name]} > {limits[name]}" for name in exceeded
            )
            raise ValueError(f"Trefftz basis exceeds its resource budget: {details}.")
        return TrefftzResourceEvidence(
            rank=requested["rank"],
            monomials=requested["monomials"],
            basis_entries=requested["basis_entries"],
            basis_bytes=requested["basis_bytes"],
        )


class TrefftzResourceEvidence(StrictModule, NonTrainableState):
    """Realized storage dimensions admitted by a resource budget."""

    rank: int = eqx.field(static=True)
    monomials: int = eqx.field(static=True)
    basis_entries: int = eqx.field(static=True)
    basis_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self, *, rank: int, monomials: int, basis_entries: int, basis_bytes: int
    ):
        self.rank = int(rank)
        self.monomials = int(monomials)
        self.basis_entries = int(basis_entries)
        self.basis_bytes = int(basis_bytes)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "trefftz-resource-evidence-v1",
                "rank": self.rank,
                "monomials": self.monomials,
                "basis_entries": self.basis_entries,
                "basis_bytes": self.basis_bytes,
            }
        )


class TrialSpaceCertificate(StrictModule, NonTrainableState):
    """Construction claim for a finite PDE-satisfying trial space."""

    equation_family: TrialEquationFamily = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    field_shape: tuple[int, ...] = eqx.field(static=True)
    construction: str = eqx.field(static=True)
    equation_parameters: tuple[tuple[str, int | float], ...] = eqx.field(static=True)
    exactness: TrialExactness = eqx.field(static=True)
    coverage: TrialCoverage = eqx.field(static=True)
    linear_in_coefficients: bool = eqx.field(static=True)
    homogeneous_equation: bool = eqx.field(static=True)
    validity_region: TrialValidityRegion = eqx.field(static=True)
    singular_support_id: str | None = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    representation_id: str | None = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    construction_residual: Array
    construction_tolerance: float = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        equation_family: TrialEquationFamily,
        ambient_dimension: int,
        field_shape: Sequence[int] = (),
        construction: str,
        equation_parameters: Mapping[str, int | float]
        | Sequence[tuple[str, int | float]] = (),
        normalization_id: str,
        basis_id: str,
        rank: int,
        assumptions: Sequence[str],
        construction_residual: ArrayLike,
        construction_tolerance: float,
        representation_id: str | None = None,
        coverage: TrialCoverage = "finite-subspace",
        linear_in_coefficients: bool = True,
        validity_region: TrialValidityRegion = "all-space",
        singular_support_id: str | None = None,
    ):
        if equation_family not in (
            "laplace",
            "polyharmonic",
            "helmholtz",
            "linear-elasticity",
            "dirac",
        ):
            raise ValueError("Unknown Trefftz equation family.")
        dimension = int(ambient_dimension)
        rank_ = int(rank)
        shape = tuple(int(value) for value in field_shape)
        tolerance = float(construction_tolerance)
        residual = jnp.asarray(construction_residual, dtype=float).reshape(())
        parameters = tuple(
            sorted(
                dict(equation_parameters).items(),
                key=lambda item: item[0],
            )
        )
        assumptions_ = tuple(str(value) for value in assumptions)
        if coverage not in ("finite-subspace", "finite-parametric-family"):
            raise ValueError("Unknown Trefftz coverage contract.")
        if coverage == "finite-subspace" and not linear_in_coefficients:
            raise ValueError(
                "A finite-subspace certificate must be linear in its coefficients."
            )
        if validity_region not in ("all-space", "off-singular-support"):
            raise ValueError("Unknown Trefftz validity region.")
        singular_support = (
            None if singular_support_id is None else str(singular_support_id)
        )
        if validity_region == "all-space" and singular_support is not None:
            raise ValueError("All-space trial fields cannot declare singular support.")
        if validity_region == "off-singular-support" and not singular_support:
            raise ValueError(
                "Off-singular-support trial fields require a singular support ID."
            )
        if dimension < 2:
            raise ValueError("Trefftz certificates require ambient_dimension >= 2.")
        if rank_ <= 0:
            raise ValueError("Trefftz certificates require positive rank.")
        if any(value <= 0 for value in shape):
            raise ValueError("Trial-space field dimensions must be positive.")
        if not construction or not normalization_id or not basis_id:
            raise ValueError(
                "Trefftz construction, normalization, and basis IDs are required."
            )
        if any(not name for name, _ in parameters):
            raise ValueError("Trefftz equation-parameter names must be nonempty.")
        if not assumptions_ or any(not value for value in assumptions_):
            raise ValueError("Trefftz certificates require nonempty assumptions.")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "Trefftz construction tolerance must be finite and nonnegative."
            )
        residual_host = float(residual)
        if not math.isfinite(residual_host) or residual_host < 0.0:
            raise ValueError(
                "Trefftz construction residual must be finite and nonnegative."
            )
        if residual_host > tolerance:
            raise ValueError(
                "Trefftz construction residual exceeds tolerance: "
                f"{residual_host:.3e} > {tolerance:.3e}."
            )
        representation = None if representation_id is None else str(representation_id)
        if representation_id is not None and not representation:
            raise ValueError("Trefftz representation_id must be non-empty when supplied.")
        self.equation_family = equation_family
        self.ambient_dimension = dimension
        self.field_shape = shape
        self.construction = str(construction)
        self.equation_parameters = parameters
        self.exactness = "algebraic"
        self.coverage = coverage
        self.linear_in_coefficients = bool(linear_in_coefficients)
        self.homogeneous_equation = True
        self.validity_region = validity_region
        self.singular_support_id = singular_support
        self.normalization_id = str(normalization_id)
        self.basis_id = str(basis_id)
        self.representation_id = representation
        self.rank = rank_
        self.assumptions = assumptions_
        self.construction_residual = residual
        self.construction_tolerance = tolerance
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "trefftz-trial-space-certificate-v2",
                "equation_family": equation_family,
                "ambient_dimension": dimension,
                "field_shape": list(shape),
                "construction": construction,
                "equation_parameters": list(parameters),
                "normalization_id": normalization_id,
                "basis_id": basis_id,
                "representation_id": self.representation_id,
                "rank": rank_,
                "assumptions": list(assumptions_),
                "exactness": self.exactness,
                "coverage": self.coverage,
                "linear_in_coefficients": self.linear_in_coefficients,
                "validity_region": validity_region,
                "singular_support_id": singular_support,
            }
        )

    def for_field_shape(self, field_shape: Sequence[int], /) -> "TrialSpaceCertificate":
        return TrialSpaceCertificate(
            equation_family=self.equation_family,
            ambient_dimension=self.ambient_dimension,
            field_shape=field_shape,
            construction=self.construction,
            equation_parameters=self.equation_parameters,
            normalization_id=self.normalization_id,
            basis_id=self.basis_id,
            representation_id=self.representation_id,
            rank=self.rank,
            assumptions=self.assumptions,
            construction_residual=self.construction_residual,
            construction_tolerance=self.construction_tolerance,
            validity_region=self.validity_region,
            coverage=self.coverage,
            linear_in_coefficients=self.linear_in_coefficients,
            singular_support_id=self.singular_support_id,
        )


def trial_target_fingerprint(
    points: ArrayLike,
    ambient_dimension: int,
    /,
) -> str:
    values = jnp.asarray(points, dtype=float)
    dimension = int(ambient_dimension)
    if dimension <= 0 or values.ndim < 1 or values.shape[-1] != dimension:
        raise ValueError("Trial target points must end in the ambient dimension.")
    flattened = values.reshape((-1, dimension))
    if flattened.shape[0] == 0:
        raise ValueError("Trial target points must be nonempty.")
    return canonical_fingerprint(
        {
            "kind": "trial-space-target-points-v1",
            "points": array_tree_fingerprint(flattened),
        }
    )


class AbstractTrialSpaceAdmissibility(StrictModule, NonTrainableState):
    """Abstract target-domain evidence for a restricted-validity trial field."""

    pde_membership_valid: AbstractAttribute[Array]
    accuracy_supported: AbstractAttribute[Array]
    target_count: AbstractAttribute[int]
    target_fingerprint: AbstractAttribute[str]
    singular_support_id: AbstractAttribute[str]
    report_id: AbstractAttribute[str]


class TrialSpaceAuditReport(StrictModule, NonTrainableState):
    """Sampled residual evidence for one certified trial field."""

    finite: Array
    maximum_residual: Array
    root_mean_square_residual: Array
    reference_scale: Array
    tolerance: Array
    valid: Array
    pde_membership_valid: Array
    evaluation_accuracy_supported: Array
    admissibility_report_id: str | None = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    audit_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        finite: ArrayLike,
        maximum_residual: ArrayLike,
        root_mean_square_residual: ArrayLike,
        reference_scale: ArrayLike,
        tolerance: ArrayLike,
        point_count: int,
        certificate_id: str,
        point_fingerprint: str,
        pde_membership_valid: ArrayLike = True,
        evaluation_accuracy_supported: ArrayLike = True,
        admissibility_report_id: str | None = None,
    ):
        finite_ = jnp.asarray(finite, dtype=bool).reshape(())
        maximum = jnp.asarray(maximum_residual, dtype=float).reshape(())
        rms = jnp.asarray(root_mean_square_residual, dtype=float).reshape(())
        scale = jnp.asarray(reference_scale, dtype=float).reshape(())
        tolerance_ = jnp.asarray(tolerance, dtype=float).reshape(())
        membership = jnp.asarray(pde_membership_valid, dtype=bool).reshape(())
        accuracy = jnp.asarray(
            evaluation_accuracy_supported,
            dtype=bool,
        ).reshape(())
        count = int(point_count)
        if count <= 0:
            raise ValueError("Trial-space audits require at least one point.")
        if not certificate_id or not point_fingerprint:
            raise ValueError(
                "Trial-space audits require certificate and point fingerprints."
            )
        self.finite = finite_
        self.maximum_residual = maximum
        self.root_mean_square_residual = rms
        self.reference_scale = scale
        self.tolerance = tolerance_
        self.pde_membership_valid = membership
        self.evaluation_accuracy_supported = accuracy
        self.admissibility_report_id = (
            None if admissibility_report_id is None else str(admissibility_report_id)
        )
        self.valid = finite_ & membership & (maximum <= tolerance_)
        self.point_count = count
        self.certificate_id = str(certificate_id)
        self.audit_id = canonical_fingerprint(
            {
                "kind": "trefftz-trial-space-audit-v2",
                "certificate_id": certificate_id,
                "point_fingerprint": point_fingerprint,
                "point_count": count,
                "admissibility_report_id": self.admissibility_report_id,
            }
        )


class AbstractTrefftzBasis(StrictModule, NonTrainableState):
    """Abstract fixed feature basis whose span satisfies one homogeneous PDE."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def rank(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def dtype(self) -> jnp.dtype:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def basis_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def certificate(self) -> TrialSpaceCertificate:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def resource_evidence(self) -> TrefftzResourceEvidence:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, point: ArrayLike, /) -> Array:
        raise NotImplementedError


class LinearTrefftzField(AbstractArrayModel):
    """Trainable real linear combination of one fixed Trefftz basis."""

    basis: AbstractTrefftzBasis
    coefficients: Array
    in_size: int
    out_size: int | Literal["scalar"]

    def __init__(
        self,
        basis: AbstractTrefftzBasis,
        /,
        *,
        out_size: int | Literal["scalar"] = "scalar",
        initial_scale: float = 0.0,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(basis, AbstractTrefftzBasis):
            raise TypeError("LinearTrefftzField requires an AbstractTrefftzBasis.")
        output_count = 1 if out_size == "scalar" else int(out_size)
        scale = float(initial_scale)
        if output_count <= 0:
            raise ValueError("LinearTrefftzField out_size must be positive.")
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "LinearTrefftzField initial_scale must be finite and nonnegative."
            )
        shape = (output_count, basis.rank)
        if scale == 0.0:
            coefficients = jnp.zeros(shape, dtype=basis.dtype)
        else:
            coefficients = (
                scale
                * jr.normal(key, shape, dtype=basis.dtype)
                / math.sqrt(float(basis.rank))
            )
        self.basis = basis
        self.coefficients = coefficients
        self.in_size = basis.dimension
        self.out_size = out_size

    def __call__(self, point: Array, /, *, key: Any = None) -> Array:
        del key
        features = self.basis.evaluate(point)
        values = self.coefficients @ features
        if self.out_size == "scalar":
            return values.reshape(())
        return values

    def model_metadata(self) -> Mapping[str, Any]:
        field_shape = () if self.out_size == "scalar" else (int(self.out_size),)
        return {
            TRIAL_SPACE_CERTIFICATE_KEY: self.basis.certificate.for_field_shape(
                field_shape
            )
        }


__all__ = [
    "AbstractTrialSpaceAdmissibility",
    "AbstractTrefftzBasis",
    "LinearTrefftzField",
    "SimilarityNormalization",
    "TrefftzResourceBudget",
    "TrefftzResourceEvidence",
    "TRIAL_SPACE_CERTIFICATE_KEY",
    "TRIAL_SPACE_REPRESENTATION_KEY",
    "TrialCoverage",
    "trial_target_fingerprint",
    "TrialEquationFamily",
    "TrialExactness",
    "TrialSpaceAuditReport",
    "TrialSpaceCertificate",
    "TrialValidityRegion",
]
