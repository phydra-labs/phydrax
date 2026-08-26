#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import hashlib
import heapq
from collections.abc import Sequence
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ..linalg import ArraySpace, DiagonalPairing
from ._axis import AxisDiscretization
from ._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from ._lifecycle import (
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from ._measure import DiscreteMeasure
from ._spaces import DiscreteFieldSpace, TensorDofLayout
from ._spectral import SpectralDecomposition
from ._support import DiscreteSupport
from ._topology import EntitySet, PointTopology


_TensorBasis = Literal["fourier", "sine", "cosine"]


def _hash_parts(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            digest.update(part.encode("utf-8"))
            digest.update(b"\0")
            continue
        array = np.ascontiguousarray(np.asarray(part))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _canonicalize_mode_signs(modes: np.ndarray, /) -> np.ndarray:
    out = np.array(modes, dtype=float, copy=True)
    for column in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, column])))
        if out[pivot, column] < 0.0:
            out[:, column] *= -1.0
    return out


def _axis_data(
    axis: AxisDiscretization,
    basis: _TensorBasis,
    /,
) -> tuple[np.ndarray, np.ndarray, float]:
    nodes = np.asarray(axis.nodes, dtype=float)
    weights = np.asarray(axis.quad_weights, dtype=float)
    spacing = np.diff(nodes)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
        raise ValueError("Exact tensor eigensystems require uniformly spaced axes.")

    if basis in ("fourier", "sine"):
        expected = np.full_like(weights, weights[0])
    else:
        pattern = np.ones_like(weights)
        pattern[[0, -1]] = 0.5
        expected = pattern * (np.sum(weights) / np.sum(pattern))
    if not np.allclose(weights, expected, rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"{basis} axis quadrature weights are incompatible with its exact "
            "weighted eigensystem."
        )
    return nodes, weights, float(spacing[0])


def _axis_eigenvalues(
    axis: AxisDiscretization,
    basis: _TensorBasis,
    /,
) -> np.ndarray:
    """Return one exact axis spectrum in deterministic real-mode order."""
    nodes, _, spacing = _axis_data(axis, basis)
    count = int(nodes.size)
    if basis == "fourier":
        mode_indices = np.arange(count, dtype=int)
        frequencies = ((mode_indices + 1) // 2).astype(float)
        frequencies[0] = 0.0
        return (2.0 * np.pi * frequencies / (float(count) * spacing)) ** 2
    if basis == "sine":
        frequencies = np.arange(1, count + 1, dtype=float)
        return (np.pi * frequencies / (float(count) * spacing)) ** 2
    frequencies = np.arange(count, dtype=float)
    return (np.pi * frequencies / (float(count - 1) * spacing)) ** 2


def _axis_modes(
    axis: AxisDiscretization,
    basis: _TensorBasis,
    mode_indices: np.ndarray,
    /,
) -> np.ndarray:
    """Evaluate only selected real axis modes, using O(axis_size * rank) memory."""
    nodes, weights, _ = _axis_data(axis, basis)
    count = int(nodes.size)
    requested = np.asarray(mode_indices, dtype=int).reshape((-1,))
    if np.any(requested < 0) or np.any(requested >= count):
        raise ValueError("Axis mode index lies outside the eigensystem.")
    unique, inverse = np.unique(requested, return_inverse=True)
    node_indices = np.arange(count, dtype=float)
    modes = np.empty((count, unique.size), dtype=float)
    for column, mode_index in enumerate(unique):
        if basis == "fourier":
            if mode_index == 0:
                modes[:, column] = 1.0
                continue
            frequency = (int(mode_index) + 1) // 2
            angle = 2.0 * np.pi * float(frequency) * node_indices / float(count)
            modes[:, column] = np.cos(angle) if mode_index % 2 == 1 else np.sin(angle)
        elif basis == "sine":
            frequency = int(mode_index) + 1
            modes[:, column] = np.sin(
                np.pi * (node_indices + 0.5) * float(frequency) / float(count)
            )
        else:
            modes[:, column] = np.cos(
                np.pi * node_indices * float(mode_index) / float(count - 1)
            )

    norms = np.sqrt(np.sum(weights[:, None] * modes**2, axis=0))
    modes = _canonicalize_mode_signs(modes / norms[None, :])
    gram = modes.T @ (weights[:, None] * modes)
    if not np.allclose(gram, np.eye(unique.size), rtol=1e-9, atol=1e-10):
        raise ValueError(
            f"{basis} axis modes failed their weighted orthonormality contract."
        )
    return modes[:, inverse]


def _smallest_tensor_indices(
    axis_eigenvalues: tuple[np.ndarray, ...],
    retained: int,
    /,
) -> tuple[tuple[int, ...], ...]:
    """Select the smallest tensor sums without materializing their full product."""
    start = (0,) * len(axis_eigenvalues)
    queue: list[tuple[float, tuple[int, ...]]] = [(0.0, start)]
    seen = {start}
    selected: list[tuple[int, ...]] = []
    while len(selected) < retained:
        _, indices = heapq.heappop(queue)
        selected.append(indices)
        for axis_index, values in enumerate(axis_eigenvalues):
            if indices[axis_index] + 1 >= values.size:
                continue
            neighbor = list(indices)
            neighbor[axis_index] += 1
            neighbor_tuple = tuple(neighbor)
            if neighbor_tuple in seen:
                continue
            seen.add(neighbor_tuple)
            total = sum(
                float(axis_eigenvalues[index][mode])
                for index, mode in enumerate(neighbor_tuple)
            )
            heapq.heappush(queue, (total, neighbor_tuple))
    return tuple(selected)


def _dual_basis_first_derivative(
    state: Array,
    nodes: Array,
    /,
    *,
    axis: int,
    basis: Literal["sine", "cosine"],
) -> Array:
    """Differentiate the parity-dual values produced by a primal gradient."""
    coordinates = jnp.asarray(nodes, dtype=float).reshape((-1,))
    count = int(coordinates.size)
    if count < 2:
        return jnp.zeros_like(state)
    spacing = coordinates[1] - coordinates[0]
    values = jnp.moveaxis(state, axis, 0)
    if basis == "sine":
        extended = jnp.concatenate((values, values[::-1]), axis=0)
        extended_size = 2 * count
    else:
        extended = jnp.concatenate((values, -values[-2:0:-1]), axis=0)
        extended_size = 2 * (count - 1)
    frequencies = 2.0 * jnp.pi * jnp.fft.fftfreq(extended_size, d=spacing)
    frequency_shape = (extended_size,) + (1,) * (extended.ndim - 1)
    coefficients = jnp.fft.fft(extended, axis=0)
    derivative = jnp.fft.ifft(
        1j * frequencies.reshape(frequency_shape) * coefficients,
        axis=0,
    )[:count]
    if not jnp.iscomplexobj(state):
        derivative = jnp.real(derivative)
    return jnp.moveaxis(derivative, 0, axis)


def _normalize_spatial_axes(
    axes: int | Sequence[int] | None,
    rank: int,
    /,
) -> tuple[int, ...]:
    if axes is None:
        return tuple(range(rank))
    values = (int(axes),) if isinstance(axes, int) else tuple(int(axis) for axis in axes)
    if not values:
        raise ValueError("At least one spatial axis is required.")
    if len(set(values)) != len(values) or any(
        axis < 0 or axis >= rank for axis in values
    ):
        raise ValueError(f"Spatial axes must be unique and lie in [0, {rank}).")
    return values


class AbstractStrongFormDiscretization(AbstractPreparedDiscretization):
    """Matrix-free spatial state contract for method-of-lines problems.

    ``state_shape`` is the leading spatial shape. Methods preserve any trailing
    channel axes. ``laplacian_matrix`` is intentionally explicit and should be used
    only for analysis or small systems.
    """

    @property
    @abc.abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def num_points(self) -> int:
        return int(prod(self.state_shape))

    @property
    def spatial_dimension(self) -> int:
        """Physical derivative dimension, independent of state storage rank."""
        return len(self.state_shape)

    @property
    @abc.abstractmethod
    def quadrature_weights(self) -> Array:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def discretization_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def points(self) -> Array | None:
        """Flattened spatial coordinates, when the discretization has them."""
        raise NotImplementedError

    @abc.abstractmethod
    def partial_derivative(
        self,
        state: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def divergence(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
        dual: bool = False,
    ) -> Array:
        raise NotImplementedError

    def curl(
        self,
        state: ArrayLike,
        /,
        *,
        axes: Sequence[int] | None = None,
    ) -> Array:
        selected = _normalize_spatial_axes(axes, self.spatial_dimension)
        value = jnp.asarray(state)
        if (
            len(selected) != 3
            or value.ndim <= len(self.state_shape)
            or value.shape[-1] != 3
        ):
            raise ValueError("Curl requires three spatial axes and three components.")
        first, second, third = selected
        return jnp.stack(
            (
                self.partial_derivative(value[..., 2], axis=second)
                - self.partial_derivative(value[..., 1], axis=third),
                self.partial_derivative(value[..., 0], axis=third)
                - self.partial_derivative(value[..., 2], axis=first),
                self.partial_derivative(value[..., 1], axis=first)
                - self.partial_derivative(value[..., 0], axis=second),
            ),
            axis=-1,
        )

    @abc.abstractmethod
    def laplacian(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def integral(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def flatten(self, state: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def unflatten(self, state: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def laplacian_matrix(self) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def eigenpairs(self, *, rank: int | None = None) -> tuple[Array, Array]:
        """Return eigenvalues of ``-laplacian`` and weighted-orthonormal modes."""
        raise NotImplementedError


class EigenbasisDiscretization(AbstractStrongFormDiscretization):
    """Method-of-lines wrapper around an existing manifold spectral plan."""

    plan: SpectralDecomposition
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    _state_shape: tuple[int, ...] = eqx.field(static=True)
    _discretization_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralDecomposition,
        /,
        *,
        field_names: Sequence[str] = ("state",),
        key: DiscretizationKey | None = None,
        numeric_version: str = "0",
        dtype: Any = float,
    ):
        if not isinstance(plan, SpectralDecomposition):
            raise TypeError("plan must be a SpectralDecomposition.")
        fields = tuple(str(name) for name in field_names)
        if (
            not fields
            or any(not name for name in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ValueError("field_names must contain unique non-empty names.")
        key_ = (
            DiscretizationKey("spectral", DiscretizationRole.PHYSICAL)
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        points = EntitySet("spectral_points", 0, np.arange(plan.num_points))
        topology = PointTopology(points)
        support = DiscreteSupport(topology, 1, plan.decomposition_id)
        measure = DiscreteMeasure(
            "spectral",
            support.support_id,
            points.entity_set_id,
            plan.quadrature_weights,
            normalization="physical",
        )
        pairing = DiagonalPairing(plan.quadrature_weights)
        layout = TensorDofLayout(("point",), (plan.num_points,))
        spaces = tuple(
            DiscreteFieldSpace(
                field_name,
                support.support_id,
                layout,
                ArraySpace(
                    (plan.num_points,),
                    dtype=dtype,
                    pairing=pairing,
                    space_id=canonical_fingerprint(
                        {
                            "kind": "spectral-field-coordinates",
                            "field": field_name,
                            "basis": plan.decomposition_id,
                        }
                    ),
                ),
                representation="point_value",
                reconstruction_id=plan.decomposition_id,
            )
            for field_name in fields
        )
        capabilities = (
            DiscretizationCapability.STRONG_DERIVATIVE,
            DiscretizationCapability.SPECTRAL_TRANSFORM,
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.MATRIX_FREE,
        )
        preparation = PreparationReport(
            capabilities=capabilities,
            resource_counts={
                "points": plan.num_points,
                "modes": plan.num_modes,
                "fields": len(spaces),
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=key_,
            support=support,
            field_spaces=spaces,
            measures=(measure,),
            capabilities=capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        prepared_identifier = _hash_parts(
            "spectral-spatial-v2",
            plan.decomposition_id,
            plan.eigenvalues,
            plan.quadrature_weights,
        )
        self.plan = plan
        self.key = key_
        self.support = support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.decomposition_id
        self.prepared_id = prepared_identifier
        self.numeric_version = version
        self.preparation = preparation
        self._state_shape = (int(plan.num_points),)
        self._discretization_id = prepared_identifier

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def quadrature_weights(self) -> Array:
        return self.plan.quadrature_weights

    @property
    def discretization_id(self) -> str:
        return self._discretization_id

    @property
    def points(self) -> None:
        return None

    def _validate_state(self, state: ArrayLike, /) -> Array:
        array = jnp.asarray(state)
        if array.ndim < 1 or int(array.shape[0]) != self.plan.num_points:
            raise ValueError(
                "State must begin with spectral point count "
                f"({self.plan.num_points},); got {array.shape}."
            )
        return array

    def partial_derivative(
        self,
        state: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        raise NotImplementedError(
            "EigenbasisDiscretization has no coordinate derivative frame."
        )

    def gradient(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        raise NotImplementedError(
            "EigenbasisDiscretization has no coordinate gradient frame."
        )

    def divergence(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
        dual: bool = False,
    ) -> Array:
        raise NotImplementedError(
            "EigenbasisDiscretization has no coordinate divergence frame."
        )

    def integral(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        selected = _normalize_spatial_axes(axes, 1)
        if selected != (0,):
            raise ValueError("Spectral spatial integrals expose one point axis.")
        array = self._validate_state(state)
        return jnp.tensordot(
            self.quadrature_weights,
            array,
            axes=((0,), (0,)),
        )

    def laplacian(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        selected = _normalize_spatial_axes(axes, 1)
        if selected != (0,):
            raise ValueError("Spectral spatial Laplacians expose one point axis.")
        array = self._validate_state(state)
        coefficients = oe.contract("mp,p...->m...", self.plan.analysis, array)
        scale = self.plan.eigenvalues.reshape(
            (self.plan.num_modes,) + (1,) * (coefficients.ndim - 1)
        )
        return -oe.contract("pm,m...->p...", self.plan.synthesis, scale * coefficients)

    def flatten(self, state: ArrayLike, /) -> Array:
        return self._validate_state(state)

    def unflatten(self, state: ArrayLike, /) -> Array:
        return self._validate_state(state)

    def laplacian_matrix(self) -> Array:
        return -(
            self.plan.synthesis @ (self.plan.eigenvalues[:, None] * self.plan.analysis)
        )

    def eigenpairs(self, *, rank: int | None = None) -> tuple[Array, Array]:
        count = self.plan.num_modes
        retained = count if rank is None else int(rank)
        if retained <= 0 or retained > count:
            raise ValueError(f"rank must lie in [1, {count}].")
        return (
            self.plan.eigenvalues[:retained],
            self.plan.synthesis[:, :retained],
        )


__all__ = [
    "AbstractStrongFormDiscretization",
    "EigenbasisDiscretization",
]
