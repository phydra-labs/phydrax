#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule


RandomFieldRole: TypeAlias = Literal[
    "input",
    "initial_condition",
    "coefficient",
    "boundary_data",
    "forcing",
    "observation",
]


def _digest(namespace: str, *parts: Any) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_role(role: str, /) -> RandomFieldRole:
    if role not in (
        "input",
        "initial_condition",
        "coefficient",
        "boundary_data",
        "forcing",
        "observation",
    ):
        raise ValueError(
            "role must be 'input', 'initial_condition', 'coefficient', "
            "'boundary_data', 'forcing', or 'observation'."
        )
    return role


def _validate_mode_ids(mode_ids: Sequence[str], /) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in mode_ids)
    if not resolved or any(not value for value in resolved):
        raise ValueError("mode_ids must contain at least one non-empty identifier.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("mode_ids must be unique.")
    return resolved


class GaussianCoefficientRealization(StrictModule):
    """Reusable standard-normal latent coefficients with explicit mode identity."""

    coefficients: Array
    root_key: Array
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    parent_realization_id: str | None = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        root_key: Key[Array, ""],
        /,
        *,
        mode_ids: Sequence[str],
        coupling_id: str,
        realization_id: str | None = None,
        parent_realization_id: str | None = None,
        label: str | None = None,
    ):
        identifiers = _validate_mode_ids(mode_ids)
        values = jnp.asarray(coefficients, dtype=float)
        if values.ndim < 1 or values.shape[-1] != len(identifiers):
            raise ValueError(
                "coefficients must have shape sample_shape + (len(mode_ids),)."
            )
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("coefficients must be finite.")
        if not isinstance(coupling_id, str) or not coupling_id:
            raise ValueError("coupling_id must be a non-empty string.")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("label must be a non-empty string or None.")
        key_data = tuple(
            int(value) for value in np.asarray(jr.key_data(root_key)).ravel()
        )
        sample_shape = tuple(int(size) for size in values.shape[:-1])
        identifier = (
            _digest(
                "gaussian-coefficient-realization-v1",
                key_data,
                identifiers,
                sample_shape,
                coupling_id,
                parent_realization_id,
                label,
            )
            if realization_id is None
            else str(realization_id)
        )
        if not identifier:
            raise ValueError("realization_id must be non-empty.")
        self.coefficients = values
        self.root_key = root_key
        self.mode_ids = identifiers
        self.sample_shape = sample_shape
        self.realization_id = identifier
        self.coupling_id = coupling_id
        self.parent_realization_id = parent_realization_id
        self.label = label

    @classmethod
    def sample(
        cls,
        key: Key[Array, ""],
        mode_ids: Sequence[str],
        /,
        *,
        sample_shape: Sequence[int] = (),
        coupling_id: str | None = None,
        label: str | None = None,
    ) -> "GaussianCoefficientRealization":
        identifiers = _validate_mode_ids(mode_ids)
        shape = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("sample_shape dimensions must be positive.")
        resolved_coupling = (
            _digest("gaussian-coefficient-coupling-v1", identifiers, label)
            if coupling_id is None
            else coupling_id
        )
        values = jr.normal(key, shape + (len(identifiers),), dtype=float)
        return cls(
            values,
            key,
            mode_ids=identifiers,
            coupling_id=resolved_coupling,
            label=label,
        )

    def select(self, mode_ids: Sequence[str], /) -> "GaussianCoefficientRealization":
        identifiers = _validate_mode_ids(mode_ids)
        source_index = {mode_id: index for index, mode_id in enumerate(self.mode_ids)}
        missing = tuple(mode_id for mode_id in identifiers if mode_id not in source_index)
        if missing:
            raise ValueError(
                "Coefficient realization does not contain requested modes "
                f"{missing!r}; construct an explicit coupling over their union."
            )
        indices = tuple(source_index[mode_id] for mode_id in identifiers)
        identifier = _digest(
            "gaussian-coefficient-selection-v1",
            self.realization_id,
            identifiers,
        )
        return GaussianCoefficientRealization(
            self.coefficients[..., indices],
            self.root_key,
            mode_ids=identifiers,
            coupling_id=self.coupling_id,
            realization_id=identifier,
            parent_realization_id=self.realization_id,
            label=self.label,
        )


class SpatialBasisSynthesis(StrictModule):
    """Basis-independent synthesis adapter for one discretized spatial field."""

    modes: Array
    eigenvalues: Array
    quadrature_weights: Array
    mean: Array
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    discretization_id: str | None = eqx.field(static=True)
    synthesis_id: str = eqx.field(static=True)

    def __init__(
        self,
        modes: ArrayLike,
        eigenvalues: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
        *,
        mode_ids: Sequence[str],
        basis_id: str,
        discretization_id: str | None = None,
        mean: ArrayLike = 0.0,
    ):
        mode_array = jnp.asarray(modes, dtype=float)
        if mode_array.ndim < 2:
            raise ValueError("modes must have shape spatial_shape + (rank,).")
        spatial_shape = tuple(int(size) for size in mode_array.shape[:-1])
        identifiers = _validate_mode_ids(mode_ids)
        if mode_array.shape[-1] != len(identifiers):
            raise ValueError("modes and mode_ids must have the same rank.")
        eigenvalue_array = jnp.asarray(eigenvalues, dtype=float).reshape((-1,))
        if eigenvalue_array.shape != (len(identifiers),):
            raise ValueError("eigenvalues must contain one value per mode.")
        if not bool(jnp.all(jnp.isfinite(eigenvalue_array))) or bool(
            jnp.any(eigenvalue_array < 0.0)
        ):
            raise ValueError("eigenvalues must be finite and non-negative.")
        weights = jnp.asarray(quadrature_weights, dtype=float)
        if weights.shape != spatial_shape:
            raise ValueError("quadrature_weights must have exact spatial_shape.")
        if not bool(jnp.all(jnp.isfinite(weights))) or bool(jnp.any(weights <= 0.0)):
            raise ValueError("quadrature_weights must be finite and positive.")
        mean_array = jnp.asarray(mean, dtype=float)
        if mean_array.shape == ():
            mean_array = jnp.broadcast_to(mean_array, spatial_shape)
        if mean_array.shape != spatial_shape:
            raise ValueError("mean must be scalar or have exact spatial_shape.")
        if not bool(jnp.all(jnp.isfinite(mode_array))) or not bool(
            jnp.all(jnp.isfinite(mean_array))
        ):
            raise ValueError("modes and mean must be finite.")
        if not isinstance(basis_id, str) or not basis_id:
            raise ValueError("basis_id must be a non-empty string.")
        if discretization_id is not None and (
            not isinstance(discretization_id, str) or not discretization_id
        ):
            raise ValueError("discretization_id must be non-empty or None.")
        self.modes = mode_array
        self.eigenvalues = eigenvalue_array
        self.quadrature_weights = weights
        self.mean = mean_array
        self.spatial_shape = spatial_shape
        self.mode_ids = identifiers
        self.basis_id = basis_id
        self.discretization_id = discretization_id
        self.synthesis_id = _digest(
            "spatial-basis-synthesis-v1",
            basis_id,
            discretization_id,
            identifiers,
            tuple(np.asarray(mean_array).ravel()),
        )

    @classmethod
    def from_spatial_noise_basis(
        cls,
        basis: Any,
        /,
        *,
        mean: ArrayLike = 0.0,
    ) -> "SpatialBasisSynthesis":
        from ._spatial_noise import SpatialNoiseBasis

        if not isinstance(basis, SpatialNoiseBasis):
            raise TypeError("basis must be a SpatialNoiseBasis.")
        if jnp.iscomplexobj(basis.modes):
            raise ValueError(
                "Static random-field synthesis requires a real point-value basis; "
                "reconstruct modal spectral modes before creating the synthesis."
            )
        return cls(
            basis.modes,
            basis.eigenvalues,
            basis.quadrature_weights,
            mode_ids=basis.mode_ids,
            basis_id=basis.basis_id,
            discretization_id=basis.field_space_id,
            mean=mean,
        )

    @property
    def rank(self) -> int:
        return len(self.mode_ids)

    @property
    def covariance_factor(self) -> Array:
        scale = jnp.sqrt(self.eigenvalues).reshape(
            (1,) * len(self.spatial_shape) + (self.rank,)
        )
        return self.modes * scale

    def synthesize(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients, dtype=float)
        if values.ndim < 1 or values.shape[-1] != self.rank:
            raise ValueError(
                f"coefficients must end in rank {self.rank}; got {values.shape}."
            )
        centered = jnp.tensordot(
            values,
            self.covariance_factor,
            axes=([-1], [-1]),
        )
        return centered + self.mean

    def modal_coefficients(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values, dtype=float)
        if (
            field.ndim < len(self.spatial_shape)
            or tuple(field.shape[-len(self.spatial_shape) :]) != self.spatial_shape
        ):
            raise ValueError("values must end in spatial_shape.")
        centered = field - self.mean
        weighted_modes = self.modes * self.quadrature_weights[..., None]
        field_axes = tuple(range(field.ndim - len(self.spatial_shape), field.ndim))
        mode_axes = tuple(range(len(self.spatial_shape)))
        return jnp.tensordot(centered, weighted_modes, axes=(field_axes, mode_axes))


class RandomFieldSample(StrictModule):
    """Static field values plus latent, role, grid, and transform provenance."""

    values: Array
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    role: RandomFieldRole = eqx.field(static=True)
    source: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)
    discretization_id: str | None = eqx.field(static=True)
    coefficient_realization_id: str = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    transform_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        sample_shape: Sequence[int],
        spatial_shape: Sequence[int],
        role: RandomFieldRole,
        source: str,
        field_id: str,
        basis_id: str,
        discretization_id: str | None,
        coefficient_realization_id: str,
        coupling_id: str,
        transform_id: str | None = None,
    ):
        array = jnp.asarray(values)
        sample = tuple(int(size) for size in sample_shape)
        spatial = tuple(int(size) for size in spatial_shape)
        if array.shape != sample + spatial:
            raise ValueError("values must have shape sample_shape + spatial_shape.")
        self.values = array
        self.sample_shape = sample
        self.spatial_shape = spatial
        self.role = _validate_role(role)
        self.source = source
        self.field_id = field_id
        self.basis_id = basis_id
        self.discretization_id = discretization_id
        self.coefficient_realization_id = coefficient_realization_id
        self.coupling_id = coupling_id
        self.transform_id = transform_id

    @property
    def transformed(self) -> bool:
        return self.transform_id is not None

    @property
    def num_samples(self) -> int:
        return int(np.prod(self.sample_shape, dtype=int)) if self.sample_shape else 1

    @property
    def case_values(self) -> Array:
        """Return a case-first view accepted by operator-dataset array adapters."""
        return self.values.reshape((self.num_samples, *self.spatial_shape))

    def operator_case_provenance(self):
        """Return one leakage-safe operator provenance record per latent draw."""
        from ..nn.operator import OperatorCaseProvenance

        return tuple(
            OperatorCaseProvenance(
                case_id=f"{self.field_id}:{index}",
                identities={
                    "random_field_draw": _digest(
                        "random-field-draw-v1",
                        self.coefficient_realization_id,
                        index,
                    ),
                    "latent_coupling": _digest(
                        "random-field-coupled-draw-v1",
                        self.coupling_id,
                        index,
                    ),
                },
            )
            for index in range(self.num_samples)
        )


class StaticGaussianRandomField(StrictModule):
    """A static Gaussian field with an explicit semantic role."""

    synthesis: SpatialBasisSynthesis
    role: RandomFieldRole = eqx.field(static=True)
    source: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        synthesis: SpatialBasisSynthesis,
        /,
        *,
        role: RandomFieldRole = "input",
        source: str = "latent",
    ):
        if not isinstance(synthesis, SpatialBasisSynthesis):
            raise TypeError("synthesis must be a SpatialBasisSynthesis.")
        resolved_role = _validate_role(role)
        if not isinstance(source, str) or not source:
            raise ValueError("source must be a non-empty string.")
        self.synthesis = synthesis
        self.role = resolved_role
        self.source = source
        self.field_id = _digest(
            "static-gaussian-random-field-v1",
            synthesis.synthesis_id,
            resolved_role,
            source,
        )

    @property
    def mode_ids(self) -> tuple[str, ...]:
        return self.synthesis.mode_ids

    def realize(
        self,
        key: Key[Array, ""],
        /,
        *,
        sample_shape: Sequence[int] = (),
        coupling_id: str | None = None,
        label: str | None = None,
    ) -> GaussianCoefficientRealization:
        return GaussianCoefficientRealization.sample(
            key,
            self.mode_ids,
            sample_shape=sample_shape,
            coupling_id=coupling_id,
            label=label,
        )

    def sample(
        self,
        realization: GaussianCoefficientRealization,
        /,
    ) -> RandomFieldSample:
        if not isinstance(realization, GaussianCoefficientRealization):
            raise TypeError("realization must be a GaussianCoefficientRealization.")
        selected = realization.select(self.mode_ids)
        values = self.synthesis.synthesize(selected.coefficients)
        return RandomFieldSample(
            values,
            sample_shape=selected.sample_shape,
            spatial_shape=self.synthesis.spatial_shape,
            role=self.role,
            source=self.source,
            field_id=self.field_id,
            basis_id=self.synthesis.basis_id,
            discretization_id=self.synthesis.discretization_id,
            coefficient_realization_id=selected.realization_id,
            coupling_id=selected.coupling_id,
        )

    def transform(
        self,
        transform: Callable[[Array], ArrayLike],
        /,
        *,
        transform_id: str,
    ) -> "TransformedRandomField":
        return TransformedRandomField(self, transform, transform_id=transform_id)


class TransformedRandomField(StrictModule):
    """An explicit deterministic transform of a static Gaussian field."""

    base: StaticGaussianRandomField
    transform_function: Callable[[Array], ArrayLike]
    transform_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: StaticGaussianRandomField,
        transform_function: Callable[[Array], ArrayLike],
        /,
        *,
        transform_id: str,
    ):
        if not isinstance(base, StaticGaussianRandomField):
            raise TypeError("base must be a StaticGaussianRandomField.")
        if not callable(transform_function):
            raise TypeError("transform_function must be callable.")
        if not isinstance(transform_id, str) or not transform_id:
            raise ValueError("transform_id must be a non-empty string.")
        self.base = base
        self.transform_function = transform_function
        self.transform_id = transform_id
        self.field_id = _digest(
            "transformed-random-field-v1",
            base.field_id,
            transform_id,
        )

    @property
    def mode_ids(self) -> tuple[str, ...]:
        return self.base.mode_ids

    @property
    def role(self) -> RandomFieldRole:
        return self.base.role

    def realize(
        self,
        key: Key[Array, ""],
        /,
        *,
        sample_shape: Sequence[int] = (),
        coupling_id: str | None = None,
        label: str | None = None,
    ) -> GaussianCoefficientRealization:
        return self.base.realize(
            key,
            sample_shape=sample_shape,
            coupling_id=coupling_id,
            label=label,
        )

    def sample(
        self,
        realization: GaussianCoefficientRealization,
        /,
    ) -> RandomFieldSample:
        gaussian = self.base.sample(realization)
        transformed = jnp.asarray(self.transform_function(gaussian.values))
        if transformed.shape != gaussian.values.shape:
            raise ValueError("A random-field transform must preserve the field shape.")
        if not bool(jnp.all(jnp.isfinite(transformed))):
            raise ValueError("A random-field transform must return finite values.")
        return RandomFieldSample(
            transformed,
            sample_shape=gaussian.sample_shape,
            spatial_shape=gaussian.spatial_shape,
            role=gaussian.role,
            source=gaussian.source,
            field_id=self.field_id,
            basis_id=gaussian.basis_id,
            discretization_id=gaussian.discretization_id,
            coefficient_realization_id=gaussian.coefficient_realization_id,
            coupling_id=gaussian.coupling_id,
            transform_id=self.transform_id,
        )


RandomFieldModel: TypeAlias = StaticGaussianRandomField | TransformedRandomField


class GaussianFieldCoupling(StrictModule):
    """One union of latent mode IDs shared deliberately across field resolutions."""

    fields: tuple[RandomFieldModel, ...]
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    common_mode_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[RandomFieldModel],
        /,
        *,
        label: str | None = None,
    ):
        resolved = tuple(fields)
        if len(resolved) < 2:
            raise ValueError("A GaussianFieldCoupling requires at least two fields.")
        if any(
            not isinstance(field, (StaticGaussianRandomField, TransformedRandomField))
            for field in resolved
        ):
            raise TypeError("fields must contain static Gaussian or transformed fields.")
        if label is not None and (not isinstance(label, str) or not label):
            raise ValueError("label must be a non-empty string or None.")
        union = tuple(
            dict.fromkeys(mode for field in resolved for mode in field.mode_ids)
        )
        shared = set(resolved[0].mode_ids)
        for field in resolved[1:]:
            shared.intersection_update(field.mode_ids)
        common = tuple(mode for mode in union if mode in shared)
        if not common:
            raise ValueError(
                "Coupled fields must share at least one explicit mode_id; matching PRNG "
                "keys alone does not define cross-resolution coupling."
            )
        field_ids = tuple(field.field_id for field in resolved)
        self.fields = resolved
        self.mode_ids = union
        self.common_mode_ids = common
        self.coupling_id = _digest(
            "gaussian-field-cross-resolution-coupling-v1",
            field_ids,
            union,
            common,
            label,
        )
        self.label = label

    def realize(
        self,
        key: Key[Array, ""],
        /,
        *,
        sample_shape: Sequence[int] = (),
    ) -> GaussianCoefficientRealization:
        return GaussianCoefficientRealization.sample(
            key,
            self.mode_ids,
            sample_shape=sample_shape,
            coupling_id=self.coupling_id,
            label=self.label,
        )

    def sample(
        self,
        realization: GaussianCoefficientRealization,
        /,
    ) -> tuple[RandomFieldSample, ...]:
        if realization.coupling_id != self.coupling_id:
            raise ValueError("realization does not belong to this GaussianFieldCoupling.")
        return tuple(field.sample(realization) for field in self.fields)


class GaussianFieldDiagnostics(StrictModule):
    """Modal and pointwise covariance diagnostics for a Gaussian field realization."""

    coefficient_mean_norm: float = eqx.field(static=True)
    coefficient_covariance_relative_error: float = eqx.field(static=True)
    pointwise_variance_relative_error: float = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    replay_exact: bool = eqx.field(static=True)


def gaussian_field_diagnostics(
    field: StaticGaussianRandomField,
    realization: GaussianCoefficientRealization,
    /,
) -> GaussianFieldDiagnostics:
    """Check latent standard normal moments, nodal variance, and exact replay."""
    if not isinstance(field, StaticGaussianRandomField):
        raise TypeError("field must be an untransformed StaticGaussianRandomField.")
    selected = realization.select(field.mode_ids)
    coefficients = selected.coefficients.reshape((-1, len(field.mode_ids)))
    count = int(coefficients.shape[0])
    if count < 2:
        raise ValueError("Diagnostics require at least two field samples.")
    centered = coefficients - jnp.mean(coefficients, axis=0)
    covariance = centered.T @ centered / float(count - 1)
    covariance_error = jnp.linalg.norm(covariance - jnp.eye(field.synthesis.rank))
    covariance_error = covariance_error / jnp.sqrt(float(field.synthesis.rank))
    first = field.sample(realization)
    replay = field.sample(realization)
    flattened = first.values.reshape((count, -1))
    empirical_variance = jnp.var(flattened, axis=0, ddof=1)
    expected_variance = jnp.sum(
        field.synthesis.covariance_factor.reshape((-1, field.synthesis.rank)) ** 2,
        axis=-1,
    )
    variance_error = jnp.linalg.norm(
        empirical_variance - expected_variance
    ) / jnp.maximum(
        jnp.linalg.norm(expected_variance),
        1e-14,
    )
    return GaussianFieldDiagnostics(
        coefficient_mean_norm=float(jnp.linalg.norm(jnp.mean(coefficients, axis=0))),
        coefficient_covariance_relative_error=float(covariance_error),
        pointwise_variance_relative_error=float(variance_error),
        sample_count=count,
        replay_exact=bool(jnp.array_equal(first.values, replay.values)),
    )


__all__ = [
    "GaussianCoefficientRealization",
    "GaussianFieldCoupling",
    "GaussianFieldDiagnostics",
    "RandomFieldModel",
    "RandomFieldRole",
    "RandomFieldSample",
    "SpatialBasisSynthesis",
    "StaticGaussianRandomField",
    "TransformedRandomField",
    "gaussian_field_diagnostics",
]
