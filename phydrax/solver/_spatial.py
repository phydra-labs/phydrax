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
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from phydrax.domain import AxisDiscretization, broadcasted_grid, GridSpec

from .._model import SpectralDiscretizationProvider
from .._strict import StrictModule
from ..operators.differential._array_ops import (
    _basis_nth_derivative,
    _fd_nth_derivative,
)


_TensorBasis = Literal["uniform", "fourier", "sine", "cosine"]


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

    if basis in ("uniform", "fourier", "sine"):
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
    if basis in ("uniform", "fourier"):
        mode_indices = np.arange(count, dtype=int)
        frequencies = ((mode_indices + 1) // 2).astype(float)
        frequencies[0] = 0.0
        if basis == "uniform":
            return 4.0 * np.sin(np.pi * frequencies / float(count)) ** 2 / spacing**2
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
        if basis in ("uniform", "fourier"):
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


class AbstractSpatialDiscretization(StrictModule):
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
        selected = _normalize_spatial_axes(axes, len(self.state_shape))
        value = jnp.asarray(state)
        if len(selected) != 3 or value.ndim <= len(self.state_shape) or value.shape[-1] != 3:
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


class TensorGridDiscretization(AbstractSpatialDiscretization):
    """Tensor-grid Laplacian using existing materialized axis discretizations.

    Supported axes are periodic ``uniform`` finite differences, ``fourier``
    periodic spectral derivatives, ``sine`` odd extensions (homogeneous
    Dirichlet), and ``cosine`` even extensions (homogeneous Neumann).
    """

    axes: tuple[AxisDiscretization, ...]
    _state_shape: tuple[int, ...] = eqx.field(static=True)
    _quadrature_weights: Array
    _points: Array
    basis: tuple[_TensorBasis, ...] = eqx.field(static=True)
    boundary_conditions: tuple[str, ...] = eqx.field(static=True)
    _discretization_id: str = eqx.field(static=True)

    def __init__(self, axes: Sequence[AxisDiscretization], /):
        axes_value = tuple(axes)
        if not axes_value:
            raise ValueError("TensorGridDiscretization requires at least one axis.")
        basis: list[_TensorBasis] = []
        boundary: list[str] = []
        for index, axis in enumerate(axes_value):
            if not isinstance(axis, AxisDiscretization):
                raise TypeError("axes must contain AxisDiscretization objects.")
            nodes = np.asarray(axis.nodes, dtype=float)
            if nodes.size < 2 or np.any(~np.isfinite(nodes)):
                raise ValueError(f"Axis {index} requires at least two finite nodes.")
            if np.any(np.diff(nodes) <= 0.0):
                raise ValueError(f"Axis {index} nodes must be strictly increasing.")
            weights = axis.quad_weights
            if weights is None:
                raise ValueError(f"Axis {index} requires quadrature weights.")
            weights_host = np.asarray(weights, dtype=float)
            if np.any(~np.isfinite(weights_host)) or np.any(weights_host <= 0.0):
                raise ValueError("Axis quadrature weights must be finite and positive.")
            if axis.basis == "uniform":
                if not axis.periodic:
                    raise ValueError(
                        "Uniform tensor-grid axes currently require periodic=True; "
                        "use sine or cosine axes for homogeneous boundaries."
                    )
                spacing = np.diff(nodes)
                if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
                    raise ValueError(
                        "Periodic finite differences require uniform spacing."
                    )
                basis.append("uniform")
                boundary.append("periodic")
            elif axis.basis == "fourier":
                if not axis.periodic:
                    raise ValueError("Fourier axes must be periodic.")
                basis.append("fourier")
                boundary.append("periodic")
            elif axis.basis == "sine":
                if axis.periodic:
                    raise ValueError("Sine axes must be non-periodic.")
                basis.append("sine")
                boundary.append("homogeneous_dirichlet")
            elif axis.basis == "cosine":
                if axis.periodic:
                    raise ValueError("Cosine axes must be non-periodic.")
                basis.append("cosine")
                boundary.append("homogeneous_neumann")
            else:
                raise ValueError(
                    "TensorGridDiscretization supports uniform, fourier, sine, "
                    "and cosine axes."
                )
        shape = tuple(int(axis.nodes.size) for axis in axes_value)
        tensor_weights = jnp.asarray(1.0, dtype=float)
        for index, axis in enumerate(axes_value):
            reshape = [1] * len(axes_value)
            reshape[index] = shape[index]
            assert axis.quad_weights is not None
            tensor_weights = tensor_weights * jnp.asarray(axis.quad_weights).reshape(
                tuple(reshape)
            )
        identifier_parts: list[Any] = ["tensor-grid-v2"]
        for axis in axes_value:
            identifier_parts.extend(
                (
                    axis.basis,
                    str(bool(axis.periodic)),
                    axis.nodes,
                    axis.quad_weights,
                )
            )
        self.axes = axes_value
        self._state_shape = shape
        self._quadrature_weights = jnp.broadcast_to(tensor_weights, shape)
        self._points = broadcasted_grid(tuple(axis.nodes for axis in axes_value)).reshape(
            (-1, len(axes_value))
        )
        self.basis = tuple(basis)
        self.boundary_conditions = tuple(boundary)
        self._discretization_id = _hash_parts(*identifier_parts)

    @classmethod
    def from_grid_spec(
        cls,
        spec: GridSpec,
        bounds: ArrayLike,
        /,
    ) -> "TensorGridDiscretization":
        """Materialize a ``GridSpec`` over bounds shaped ``(2, num_axes)``."""
        if not isinstance(spec, GridSpec):
            raise TypeError("spec must be a GridSpec.")
        limits = jnp.asarray(bounds, dtype=float)
        if limits.shape != (2, len(spec.axes)):
            raise ValueError(
                f"bounds must have shape {(2, len(spec.axes))}; got {limits.shape}."
            )
        return cls(
            tuple(
                axis_spec.materialize(limits[0, index], limits[1, index])
                for index, axis_spec in enumerate(spec.axes)
            )
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def quadrature_weights(self) -> Array:
        return self._quadrature_weights

    @property
    def discretization_id(self) -> str:
        return self._discretization_id

    @property
    def points(self) -> Array:
        return self._points

    def _validate_state(self, state: ArrayLike, /) -> Array:
        array = jnp.asarray(state)
        spatial_rank = len(self.state_shape)
        if (
            array.ndim < spatial_rank
            or tuple(array.shape[:spatial_rank]) != self.state_shape
        ):
            raise ValueError(
                "State must begin with tensor-grid shape "
                f"{self.state_shape}; got {array.shape}."
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
        array = self._validate_state(state)
        axis_index = int(axis)
        derivative_order = int(order)
        if axis_index < 0 or axis_index >= len(self.state_shape):
            raise ValueError(
                f"axis must lie in [0, {len(self.state_shape)}); got {axis_index}."
            )
        if derivative_order <= 0:
            raise ValueError("order must be positive.")
        axis_data = self.axes[axis_index]
        basis = self.basis[axis_index]
        if basis == "uniform":
            spacing = axis_data.nodes[1] - axis_data.nodes[0]
            if derivative_order == 2:
                return (
                    jnp.roll(array, -1, axis=axis_index)
                    - 2.0 * array
                    + jnp.roll(array, 1, axis=axis_index)
                ) / spacing**2
            return _fd_nth_derivative(
                array,
                dx=spacing,
                axis=axis_index,
                order=derivative_order,
                periodic=True,
            )
        return _basis_nth_derivative(
            array,
            axis_data.nodes,
            axis=axis_index,
            order=derivative_order,
            basis=basis,
        )

    def gradient(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        array = self._validate_state(state)
        selected = _normalize_spatial_axes(axes, len(self.state_shape))
        components = []
        for axis_index in selected:
            if self.basis[axis_index] == "uniform":
                spacing = self.axes[axis_index].nodes[1] - self.axes[axis_index].nodes[0]
                component = (
                    jnp.roll(array, -1, axis=axis_index) - array
                ) / spacing
            else:
                component = self.partial_derivative(array, axis=axis_index)
            components.append(component)
        return jnp.stack(tuple(components), axis=-1)

    def divergence(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
        dual: bool = False,
    ) -> Array:
        array = self._validate_state(state)
        selected = _normalize_spatial_axes(axes, len(self.state_shape))
        if array.ndim <= len(self.state_shape) or int(array.shape[-1]) != len(selected):
            raise ValueError(
                "Divergence requires a trailing component axis matching the "
                f"{len(selected)} selected spatial axes; got {array.shape}."
            )
        out = jnp.zeros_like(array[..., 0])
        for component_index, axis_index in enumerate(selected):
            component = array[..., component_index]
            basis = self.basis[axis_index]
            if dual and basis == "uniform":
                spacing = self.axes[axis_index].nodes[1] - self.axes[axis_index].nodes[0]
                derivative = (
                    component - jnp.roll(component, 1, axis=axis_index)
                ) / spacing
            elif dual and basis in ("sine", "cosine"):
                derivative = _dual_basis_first_derivative(
                    component,
                    self.axes[axis_index].nodes,
                    axis=axis_index,
                    basis=basis,
                )
            else:
                derivative = self.partial_derivative(component, axis=axis_index)
            out = out + derivative
        return out

    def laplacian(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        array = self._validate_state(state)
        selected = _normalize_spatial_axes(axes, len(self.state_shape))
        out = jnp.zeros_like(array)
        for axis_index in selected:
            axis = self.axes[axis_index]
            basis = self.basis[axis_index]
            if basis == "uniform":
                spacing = axis.nodes[1] - axis.nodes[0]
                second = (
                    jnp.roll(array, -1, axis=axis_index)
                    - 2.0 * array
                    + jnp.roll(array, 1, axis=axis_index)
                ) / spacing**2
            else:
                second = self.partial_derivative(array, axis=axis_index, order=2)
            out = out + second
        return out

    def integral(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        out = self._validate_state(state)
        selected = _normalize_spatial_axes(axes, len(self.state_shape))
        for axis_index in sorted(selected, reverse=True):
            weights = self.axes[axis_index].quad_weights
            assert weights is not None
            out = jnp.tensordot(weights, out, axes=((0,), (axis_index,)))
        return out

    def flatten(self, state: ArrayLike, /) -> Array:
        array = self._validate_state(state)
        return array.reshape((self.num_points,) + array.shape[len(self.state_shape) :])

    def unflatten(self, state: ArrayLike, /) -> Array:
        array = jnp.asarray(state)
        if array.ndim < 1 or int(array.shape[0]) != self.num_points:
            raise ValueError(
                f"Flattened state must begin with ({self.num_points},); got {array.shape}."
            )
        return array.reshape(self.state_shape + array.shape[1:])

    def laplacian_matrix(self) -> Array:
        identity = jnp.eye(self.num_points, dtype=float)
        columns = jax.vmap(
            lambda vector: self.flatten(self.laplacian(self.unflatten(vector)))
        )(identity)
        return columns.T

    def eigenpairs(self, *, rank: int | None = None) -> tuple[Array, Array]:
        """Return the smallest separable tensor modes without a dense Laplacian."""
        count = self.num_points
        retained = count if rank is None else int(rank)
        if retained <= 0 or retained > count:
            raise ValueError(f"rank must lie in [1, {count}].")

        axis_eigenvalues = tuple(
            _axis_eigenvalues(axis, basis)
            for axis, basis in zip(self.axes, self.basis, strict=True)
        )
        selected = _smallest_tensor_indices(axis_eigenvalues, retained)
        selected_array = np.asarray(selected, dtype=int)
        eigenvalues = np.asarray(
            [
                sum(
                    float(axis_eigenvalues[axis_index][mode_index])
                    for axis_index, mode_index in enumerate(indices)
                )
                for indices in selected
            ],
            dtype=float,
        )

        modes = np.ones(self.state_shape + (retained,), dtype=float)
        for axis_index, (axis, basis) in enumerate(
            zip(self.axes, self.basis, strict=True)
        ):
            selected_modes = _axis_modes(
                axis,
                basis,
                selected_array[:, axis_index],
            )
            reshape = [1] * len(self.state_shape) + [retained]
            reshape[axis_index] = self.state_shape[axis_index]
            modes *= selected_modes.reshape(tuple(reshape))
        modes = _canonicalize_mode_signs(modes.reshape((count, retained))).reshape(
            self.state_shape + (retained,)
        )
        return jnp.asarray(eigenvalues), jnp.asarray(modes)


class SpectralSpatialDiscretization(AbstractSpatialDiscretization):
    """Method-of-lines wrapper around an existing manifold spectral plan."""

    plan: Any
    _state_shape: tuple[int, ...] = eqx.field(static=True)
    _discretization_id: str = eqx.field(static=True)

    def __init__(self, plan: Any, /):
        if not isinstance(plan, SpectralDiscretizationProvider):
            raise TypeError("plan must implement SpectralDiscretizationProvider.")
        self.plan = plan
        self._state_shape = (int(plan.num_points),)
        self._discretization_id = _hash_parts(
            "spectral-spatial-v1",
            plan.basis_id,
            plan.eigenvalues,
            plan.quadrature_weights,
        )

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
            "SpectralSpatialDiscretization has no coordinate derivative frame."
        )

    def gradient(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        raise NotImplementedError(
            "SpectralSpatialDiscretization has no coordinate gradient frame."
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
            "SpectralSpatialDiscretization has no coordinate divergence frame."
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
    "AbstractSpatialDiscretization",
    "SpectralSpatialDiscretization",
    "TensorGridDiscretization",
]
