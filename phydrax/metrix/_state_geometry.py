#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.sparse.linalg as jsparse
from jaxtyping import Array, ArrayLike

from .._strict import AbstractAttribute, StrictModule


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _same_shape(value: Array, reference: Array, name: str, /) -> None:
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must preserve state shape {reference.shape}; got {value.shape}."
        )


class AbstractStateGeometry(StrictModule):
    """Retraction geometry for an array-valued differential-equation state.

    Vector fields keep the ordinary ``(time, state, args) -> state-shaped array``
    contract. A geometry projects those ambient arrays onto the tangent space and
    expresses them in local, state-shaped coordinates used by geometric solvers.
    """

    geometry_id: AbstractAttribute[str]
    retraction_method: AbstractAttribute[str]
    trivial: AbstractAttribute[bool]
    supports_exact_pullback: AbstractAttribute[bool]
    supports_commutator_free: AbstractAttribute[bool]

    @abstractmethod
    def contains(self, state: ArrayLike, /) -> Array:
        """Return one scalar boolean indicating membership in the state space."""
        raise NotImplementedError

    @abstractmethod
    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        """Project a state-shaped ambient vector onto the tangent space at state."""
        raise NotImplementedError

    @abstractmethod
    def to_local(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        """Express a tangent vector in state-shaped local coordinates."""
        raise NotImplementedError

    @abstractmethod
    def from_local(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        """Convert state-shaped local coordinates to an ambient tangent vector."""
        raise NotImplementedError

    @abstractmethod
    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        """Map local coordinates at state back onto the state space."""
        raise NotImplementedError

    @abstractmethod
    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        """Return local coordinates at state for a nearby point."""
        raise NotImplementedError

    @abstractmethod
    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        """Pull a tangent at ``retract(state, local_tangent)`` into local velocity."""
        raise NotImplementedError

    def local_retraction(self, state: ArrayLike, /) -> LocalRetraction:
        """Bind this geometry's retraction and pullback to one base point."""
        return LocalRetraction(self, state)

    def interpolate(
        self,
        left: ArrayLike,
        right: ArrayLike,
        weight: ArrayLike,
        /,
    ) -> Array:
        """Interpolate on the state space along the local retraction from left."""
        left_array = jnp.asarray(left)
        right_array = jnp.asarray(right)
        _same_shape(right_array, left_array, "Interpolation endpoint")
        local = self.inverse_retract(left_array, right_array)
        return self.retract(left_array, jnp.asarray(weight) * local)


class LocalRetraction(StrictModule):
    """A geometry retraction bound to a validated base point.

    ``evaluate`` maps state-shaped local coordinates to the state space;
    ``pullback`` converts an ambient tangent at that point into the derivative of
    those local coordinates. The identifiers record the resolved geometry method.
    """

    geometry: AbstractStateGeometry
    base_point: Array
    retraction_id: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(self, geometry: AbstractStateGeometry, base_point: ArrayLike, /):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("LocalRetraction geometry must be an AbstractStateGeometry.")
        base = jnp.asarray(base_point)
        membership = jnp.asarray(geometry.contains(base), dtype=bool)
        if membership.shape != ():
            raise ValueError("State geometry contains() must return a scalar boolean.")
        base = eqx.error_if(
            base,
            ~membership,
            "LocalRetraction base point is outside the state geometry.",
        )
        self.geometry = geometry
        self.base_point = base
        self.retraction_id = f"{geometry.geometry_id}:local-retraction"
        self.resolved_method = geometry.retraction_method

    def evaluate(self, local_tangent: ArrayLike, /) -> Array:
        local = jnp.asarray(local_tangent)
        _same_shape(local, self.base_point, "Local retraction coordinates")
        point = jnp.asarray(self.geometry.retract(self.base_point, local))
        _same_shape(point, self.base_point, "Local retraction")
        return point

    def __call__(self, local_tangent: ArrayLike, /) -> Array:
        return self.evaluate(local_tangent)

    def pullback(
        self,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        if not self.geometry.supports_exact_pullback:
            raise ValueError(
                "Local retraction pullback requires explicit inverse-differential "
                "capability."
            )
        local = jnp.asarray(local_tangent)
        vector = jnp.asarray(tangent)
        _same_shape(local, self.base_point, "Local retraction coordinates")
        _same_shape(vector, self.base_point, "Retraction tangent")
        velocity = jnp.asarray(
            self.geometry.pullback(self.base_point, local, vector)
        )
        _same_shape(velocity, self.base_point, "Local retraction pullback")
        return velocity


class EuclideanStateGeometry(AbstractStateGeometry):
    """Identity geometry for unconstrained array-valued states."""

    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)

    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)
    def __init__(
        self,
        *,
        geometry_id: str = "state-geometry:euclidean",
    ):
        self.geometry_id = _identifier(geometry_id, "geometry_id")
        self.retraction_method = "addition"
        self.trivial = True

        self.supports_exact_pullback = True
        self.supports_commutator_free = True

    def contains(self, state: ArrayLike, /) -> Array:
        return jnp.all(jnp.isfinite(jnp.asarray(state)))

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        vector_array = jnp.asarray(vector)
        _same_shape(vector_array, state_array, "Euclidean tangent")
        return vector_array

    def to_local(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(state, tangent)

    def from_local(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(state, local_tangent)

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = self.from_local(state_array, local_tangent)
        return state_array + local

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Euclidean retraction point")
        return point_array - state_array

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        _same_shape(local, state_array, "Euclidean local tangent")
        return self.to_local(state_array + local, tangent)


class EmbeddedStateGeometry(AbstractStateGeometry):
    """Adapter for an embedded state space defined by explicit array callables."""

    membership: Callable[[Array], Array]
    tangent_projection: Callable[[Array, Array], Array]
    retraction: Callable[[Array, Array], Array]
    inverse_retraction: Callable[[Array, Array], Array] | None
    retraction_pullback: Callable[[Array, Array, Array], Array] | None
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        membership: Callable[[Array], Array],
        tangent_projection: Callable[[Array, Array], Array],
        retraction: Callable[[Array, Array], Array],
        geometry_id: str,
        retraction_method: str,
        inverse_retraction: Callable[[Array, Array], Array] | None = None,
        retraction_pullback: Callable[[Array, Array, Array], Array] | None = None,
    ):
        for function, name in (
            (membership, "membership"),
            (tangent_projection, "tangent_projection"),
            (retraction, "retraction"),
        ):
            if not callable(function):
                raise TypeError(f"{name} must be callable.")
        if inverse_retraction is not None and not callable(inverse_retraction):
            raise TypeError("inverse_retraction must be callable or None.")
        if retraction_pullback is not None and not callable(retraction_pullback):
            raise TypeError("retraction_pullback must be callable or None.")
        self.membership = membership
        self.tangent_projection = tangent_projection
        self.retraction = retraction
        self.inverse_retraction = inverse_retraction
        self.retraction_pullback = retraction_pullback
        self.geometry_id = _identifier(geometry_id, "geometry_id")
        self.retraction_method = _identifier(
            retraction_method,
            "retraction_method",
        )
        self.trivial = False
        self.supports_exact_pullback = (
            inverse_retraction is not None and retraction_pullback is not None
        )
        self.supports_commutator_free = False

    def contains(self, state: ArrayLike, /) -> Array:
        return jnp.asarray(self.membership(jnp.asarray(state)), dtype=bool)

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        projected = jnp.asarray(
            self.tangent_projection(state_array, jnp.asarray(vector))
        )
        _same_shape(projected, state_array, "Embedded tangent projection")
        return projected

    def to_local(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(state, tangent)

    def from_local(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        return self.project_tangent(state, local_tangent)

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point = jnp.asarray(self.retraction(state_array, jnp.asarray(local_tangent)))
        _same_shape(point, state_array, "Embedded retraction")
        return point

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        point_array = jnp.asarray(point)
        _same_shape(point_array, state_array, "Embedded retraction point")
        if self.inverse_retraction is None:
            raise ValueError(
                "Embedded inverse_retract requires an explicit "
                "inverse_retraction callable."
            )
        local = jnp.asarray(self.inverse_retraction(state_array, point_array))
        _same_shape(local, state_array, "Embedded inverse retraction")
        return local

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = jnp.asarray(state)
        local = jnp.asarray(local_tangent)
        vector = jnp.asarray(tangent)
        _same_shape(local, state_array, "Embedded local tangent")
        _same_shape(vector, state_array, "Embedded retraction tangent")
        if self.retraction_pullback is None:
            raise ValueError(
                "Embedded pullback requires an explicit retraction_pullback callable."
            )
        velocity = jnp.asarray(
            self.retraction_pullback(state_array, local, vector)
        )
        _same_shape(velocity, state_array, "Embedded retraction pullback")
        return velocity


def _point_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("point_shape must contain positive dimensions.")
    return shape


class PointwiseStateGeometry(AbstractStateGeometry):
    """Apply one state geometry independently over leading point axes."""

    geometry: AbstractStateGeometry
    point_shape: tuple[int, ...] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        geometry: AbstractStateGeometry,
        point_shape: Sequence[int],
        /,
        *,
        geometry_id: str | None = None,
    ):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("Pointwise geometry must wrap an AbstractStateGeometry.")
        shape = _point_shape(point_shape)
        self.geometry = geometry
        self.point_shape = shape
        self.geometry_id = (
            f"{geometry.geometry_id}:pointwise:{'x'.join(map(str, shape))}"
            if geometry_id is None
            else _identifier(geometry_id, "geometry_id")
        )
        self.retraction_method = f"pointwise:{geometry.retraction_method}"
        self.trivial = geometry.trivial
        self.supports_exact_pullback = geometry.supports_exact_pullback
        self.supports_commutator_free = geometry.supports_commutator_free

    def _validate(self, value: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape[-len(self.point_shape) :] != self.point_shape:
            raise ValueError(
                f"{name} must have trailing point shape {self.point_shape}; "
                f"got {array.shape}."
            )
        return array

    def _unary(self, function: Callable[[Array], Array], value: Array, /) -> Array:
        if value.ndim == len(self.point_shape):
            return function(value)
        leading = value.shape[: -len(self.point_shape)]
        flat = value.reshape((-1,) + self.point_shape)
        mapped = jax.vmap(function)(flat)
        return mapped.reshape(leading + mapped.shape[1:])

    def _binary(
        self,
        function: Callable[[Array, Array], Array],
        left: Array,
        right: Array,
        /,
    ) -> Array:
        if left.ndim == len(self.point_shape):
            return function(left, right)
        leading = left.shape[: -len(self.point_shape)]
        left_flat = left.reshape((-1,) + self.point_shape)
        right_flat = right.reshape((-1,) + self.point_shape)
        mapped = jax.vmap(function)(left_flat, right_flat)
        return mapped.reshape(leading + mapped.shape[1:])

    def contains(self, state: ArrayLike, /) -> Array:
        state_array = self._validate(state, "Pointwise state")
        return jnp.all(self._unary(self.geometry.contains, state_array))

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, "Pointwise state")
        vector_array = self._validate(vector, "Pointwise tangent")
        _same_shape(vector_array, state_array, "Pointwise tangent")
        return self._binary(
            self.geometry.project_tangent,
            state_array,
            vector_array,
        )

    def to_local(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, "Pointwise state")
        tangent_array = self._validate(tangent, "Pointwise tangent")
        _same_shape(tangent_array, state_array, "Pointwise tangent")
        return self._binary(self.geometry.to_local, state_array, tangent_array)

    def from_local(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, "Pointwise state")
        local = self._validate(local_tangent, "Pointwise local tangent")
        _same_shape(local, state_array, "Pointwise local tangent")
        return self._binary(self.geometry.from_local, state_array, local)

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, "Pointwise state")
        local = self._validate(local_tangent, "Pointwise local tangent")
        _same_shape(local, state_array, "Pointwise local tangent")
        return self._binary(self.geometry.retract, state_array, local)

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, "Pointwise state")
        point_array = self._validate(point, "Pointwise retraction point")
        _same_shape(point_array, state_array, "Pointwise retraction point")
        return self._binary(
            self.geometry.inverse_retract,
            state_array,
            point_array,
        )

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        state_array = self._validate(state, "Pointwise state")
        local = self._validate(local_tangent, "Pointwise local tangent")
        vector = self._validate(tangent, "Pointwise retraction tangent")
        _same_shape(local, state_array, "Pointwise local tangent")
        _same_shape(vector, state_array, "Pointwise retraction tangent")
        leading = state_array.shape[: -len(self.point_shape)]
        state_flat = state_array.reshape((-1,) + self.point_shape)
        local_flat = local.reshape((-1,) + self.point_shape)
        vector_flat = vector.reshape((-1,) + self.point_shape)
        mapped = jax.vmap(self.geometry.pullback)(
            state_flat,
            local_flat,
            vector_flat,
        )
        return mapped.reshape(leading + mapped.shape[1:])


MatrixRetraction: TypeAlias = Literal["exponential", "cayley"]


def _dimension(value: int, /) -> int:
    dimension = int(value)
    if dimension < 2:
        raise ValueError("Matrix state geometry dimension must be at least two.")
    return dimension


def _matrix_shape(value: ArrayLike, dimension: int, name: str, /) -> Array:
    array = jnp.asarray(value)
    expected = (dimension, dimension)
    if array.shape[-2:] != expected:
        raise ValueError(
            f"{name} must have trailing matrix shape {expected}; got {array.shape}."
        )
    return array


def _transpose(value: Array, /) -> Array:
    return jnp.swapaxes(value, -1, -2)


def _symmetric(value: Array, /) -> Array:
    return 0.5 * (value + _transpose(value))


def _skew(value: Array, /) -> Array:
    return 0.5 * (value - _transpose(value))


def _matrix_map(function: Callable[[Array], Array], value: Array, /) -> Array:
    if value.ndim == 2:
        return function(value)
    leading = value.shape[:-2]
    flat = value.reshape((-1,) + value.shape[-2:])
    mapped = jax.vmap(function)(flat)
    return mapped.reshape(leading + mapped.shape[1:])


def _matrix_exponential(value: Array, /) -> Array:
    return _matrix_map(jsp.linalg.expm, value)

def _symmetric_matrix_logarithm_primal(value: Array, /) -> Array:
    eigenvalues, eigenvectors = jnp.linalg.eigh(_symmetric(value))
    safe = jnp.maximum(eigenvalues, jnp.finfo(value.dtype).tiny)
    return _symmetric(
        (eigenvectors * jnp.expand_dims(jnp.log(safe), axis=-2))
        @ _transpose(eigenvectors)
    )


@jax.custom_jvp
def _symmetric_matrix_logarithm(value: Array, /) -> Array:
    return _symmetric_matrix_logarithm_primal(value)


@_symmetric_matrix_logarithm.defjvp
def _symmetric_matrix_logarithm_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    symmetric_value = _symmetric(value)
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetric_value)
    safe = jnp.maximum(eigenvalues, jnp.finfo(value.dtype).tiny)
    left = jnp.expand_dims(safe, axis=-1)
    right = jnp.expand_dims(safe, axis=-2)
    difference = left - right
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(left), jnp.abs(right)),
        jnp.finfo(value.dtype).tiny,
    )
    close = jnp.abs(difference) <= jnp.sqrt(jnp.finfo(value.dtype).eps) * scale
    safe_difference = jnp.where(close, 1.0, difference)
    relative_difference = jnp.where(close, 0.0, difference / right)
    divided_difference = jnp.log1p(relative_difference) / safe_difference
    coefficient = jnp.where(
        close,
        2.0 / (left + right),
        divided_difference,
    )
    transformed = _transpose(eigenvectors) @ _symmetric(tangent) @ eigenvectors
    derivative = eigenvectors @ (coefficient * transformed) @ _transpose(
        eigenvectors
    )
    return _symmetric_matrix_logarithm_primal(value), _symmetric(derivative)



def _principal_local_so_logarithm(value: Array, /) -> Array:
    identity = jnp.eye(value.shape[-1], dtype=value.dtype)
    cayley = jnp.linalg.solve(value + identity, value - identity)
    radius = jnp.linalg.norm(cayley, ord=2, axis=(-2, -1))
    cayley = eqx.error_if(
        cayley,
        jnp.any(radius >= 0.5),
        "SO exponential inverse_retract requires a principal local rotation "
        "with Cayley radius below 0.5.",
    )
    square = cayley @ cayley
    term = cayley
    series = cayley
    for denominator in range(3, 64, 2):
        term = term @ square
        series = series + term / denominator
    return _skew(2.0 * series)



class SpecialOrthogonalStateGeometry(AbstractStateGeometry):
    """Left-trivialized state geometry for one or batches of SO(n) matrices."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: MatrixRetraction = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        retraction: MatrixRetraction = "exponential",
        tolerance: float = 1e-6,
        geometry_id: str | None = None,
    ):
        n = _dimension(dimension)
        if retraction not in ("exponential", "cayley"):
            raise ValueError("SO(n) retraction must be 'exponential' or 'cayley'.")
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.dimension = n
        self.tolerance = float(tolerance)
        self.geometry_id = (
            f"state-geometry:so:{n}:{retraction}"
            if geometry_id is None
            else _identifier(geometry_id, "geometry_id")
        )
        self.retraction_method = retraction
        self.trivial = False
        self.supports_exact_pullback = True
        self.supports_commutator_free = True

    def contains(self, state: ArrayLike, /) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        identity = jnp.eye(self.dimension, dtype=matrix.dtype)
        orthogonality = jnp.max(
            jnp.abs(_transpose(matrix) @ matrix - identity),
            axis=(-2, -1),
        )
        determinant = jnp.linalg.det(matrix)
        finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
        return jnp.all(
            finite
            & (orthogonality <= self.tolerance)
            & (determinant > 0.0)
        )

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        ambient = _matrix_shape(vector, self.dimension, "SO(n) tangent")
        _same_shape(ambient, matrix, "SO(n) tangent")
        return matrix @ _skew(_transpose(matrix) @ ambient)

    def to_local(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        projected = self.project_tangent(matrix, tangent)
        return _skew(_transpose(matrix) @ projected)

    def from_local(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SO(n) local tangent",
        )
        _same_shape(local, matrix, "SO(n) local tangent")
        return matrix @ _skew(local)

    def _increment(self, local_tangent: Array, /) -> Array:
        algebra = _skew(local_tangent)
        if self.retraction_method == "exponential":
            return _matrix_exponential(algebra)
        identity = jnp.eye(self.dimension, dtype=algebra.dtype)
        return jnp.linalg.solve(
            identity - 0.5 * algebra,
            identity + 0.5 * algebra,
        )

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SO(n) local tangent",
        )
        _same_shape(local, matrix, "SO(n) local tangent")
        return matrix @ self._increment(local)

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        target = _matrix_shape(point, self.dimension, "SO(n) retraction point")
        _same_shape(target, matrix, "SO(n) retraction point")
        relative = _transpose(matrix) @ target
        if self.retraction_method == "cayley":
            identity = jnp.eye(self.dimension, dtype=relative.dtype)
            cayley = jnp.linalg.solve(relative + identity, relative - identity)
            return 2.0 * _skew(cayley)
        return _principal_local_so_logarithm(relative)

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SO(n) state")
        local = _skew(
            _matrix_shape(
                local_tangent,
                self.dimension,
                "SO(n) local tangent",
            )
        )
        point = self.retract(matrix, local)
        vector = self.project_tangent(point, tangent)
        body_velocity = self.to_local(point, vector)
        if self.retraction_method == "cayley":
            identity = jnp.eye(self.dimension, dtype=matrix.dtype)
            right_factor = self._increment(local) + identity
            relative_velocity = _transpose(matrix) @ vector
            left_factor = (
                2.0
                * (identity - 0.5 * local)
                @ relative_velocity
            )
            velocity = jnp.linalg.solve(
                _transpose(right_factor),
                _transpose(left_factor),
            )
            return _skew(_transpose(velocity))

        def differential(local_velocity):
            _, ambient_velocity = jax.jvp(
                lambda value: self.retract(matrix, value),
                (local,),
                (_skew(local_velocity),),
            )
            return (
                self.to_local(point, ambient_velocity)
                + _symmetric(local_velocity)
            )

        tolerance = 1e-10 if matrix.dtype == jnp.dtype(jnp.float64) else 1e-5
        restart = 8
        krylov_cycles = max(
            4,
            2 * (self.dimension * self.dimension + restart - 1) // restart,
        )
        right_hand_side_norm = jnp.linalg.norm(
            body_velocity,
            axis=(-2, -1),
            keepdims=True,
        )
        scale = jnp.maximum(
            right_hand_side_norm,
            jnp.finfo(matrix.dtype).tiny,
        )
        normalized_body_velocity = body_velocity / scale
        normalized_velocity, _ = jsparse.gmres(
            differential,
            normalized_body_velocity,
            x0=normalized_body_velocity,
            tol=tolerance,
            atol=0.0,
            restart=restart,
            maxiter=krylov_cycles,
        )
        velocity = scale * normalized_velocity
        residual = differential(velocity) - body_velocity
        residual_norm = jnp.linalg.norm(
            residual,
            axis=(-2, -1),
            keepdims=True,
        )
        relative_residual = residual_norm / scale
        failed = jnp.where(
            right_hand_side_norm == 0.0,
            residual_norm != 0.0,
            relative_residual > 2.0 * tolerance,
        )
        velocity = eqx.error_if(
            velocity,
            failed,
            "SO exponential pullback matrix-free solve did not converge.",
        )
        return _skew(velocity)


class SymmetricPositiveDefiniteStateGeometry(AbstractStateGeometry):
    """Congruence/exponential state geometry for SPD(n) matrices."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        tolerance: float = 1e-8,
        geometry_id: str | None = None,
    ):
        n = _dimension(dimension)
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        self.dimension = n
        self.tolerance = float(tolerance)
        self.geometry_id = (
            f"state-geometry:spd:{n}:congruence-exponential"
            if geometry_id is None
            else _identifier(geometry_id, "geometry_id")
        )
        self.retraction_method = "congruence-exponential"
        self.trivial = False
        self.supports_exact_pullback = True
        self.supports_commutator_free = False

    def _congruence_factor(self, state: Array, /) -> Array:
        return jnp.linalg.cholesky(_symmetric(state))

    def _inverse_congruence(
        self,
        factor: Array,
        value: Array,
        /,
    ) -> Array:
        left_solved = jnp.linalg.solve(factor, value)
        return _transpose(
            jnp.linalg.solve(factor, _transpose(left_solved))
        )

    def contains(self, state: ArrayLike, /) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        symmetry_error = jnp.max(
            jnp.abs(matrix - _transpose(matrix)),
            axis=(-2, -1),
        )
        minimum = jnp.min(jnp.linalg.eigvalsh(_symmetric(matrix)), axis=-1)
        finite = jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
        return jnp.all(
            finite
            & (symmetry_error <= self.tolerance)
            & (minimum > self.tolerance)
        )

    def project_tangent(
        self,
        state: ArrayLike,
        vector: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        ambient = _matrix_shape(vector, self.dimension, "SPD(n) tangent")
        _same_shape(ambient, matrix, "SPD(n) tangent")
        return _symmetric(ambient)

    def to_local(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        projected = self.project_tangent(matrix, tangent)
        factor = self._congruence_factor(matrix)
        return _symmetric(self._inverse_congruence(factor, projected))

    def from_local(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SPD(n) local tangent",
        )
        _same_shape(local, matrix, "SPD(n) local tangent")
        factor = self._congruence_factor(matrix)
        return _symmetric(factor @ _symmetric(local) @ _transpose(factor))

    def retract(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SPD(n) local tangent",
        )
        _same_shape(local, matrix, "SPD(n) local tangent")
        factor = self._congruence_factor(matrix)
        return _symmetric(
            factor @ _matrix_exponential(_symmetric(local)) @ _transpose(factor)
        )

    def inverse_retract(
        self,
        state: ArrayLike,
        point: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        target = _matrix_shape(point, self.dimension, "SPD(n) retraction point")
        _same_shape(target, matrix, "SPD(n) retraction point")
        factor = self._congruence_factor(matrix)
        relative = _symmetric(self._inverse_congruence(factor, target))
        return _symmetric_matrix_logarithm(relative)

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        matrix = _matrix_shape(state, self.dimension, "SPD(n) state")
        local = _matrix_shape(
            local_tangent,
            self.dimension,
            "SPD(n) local tangent",
        )
        point = self.retract(matrix, local)
        vector = self.project_tangent(point, tangent)
        _, velocity = jax.jvp(
            lambda target: self.inverse_retract(matrix, target),
            (point,),
            (vector,),
        )
        return _symmetric(velocity)


__all__ = [
    "AbstractStateGeometry",
    "EmbeddedStateGeometry",
    "EuclideanStateGeometry",
    "LocalRetraction",
    "PointwiseStateGeometry",
    "SpecialOrthogonalStateGeometry",
    "SymmetricPositiveDefiniteStateGeometry",
]
