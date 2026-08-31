#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.discrete_velocity._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
)
from ._hyperbolic_systems import AbstractAdmissibleSystem


DVMEquilibrium = Callable[[Array, Any], ArrayLike]


def _reflection_routes(
    quadrature: CertifiedDiscreteVelocityQuadrature, /, *, tolerance: float = 1e-11
) -> tuple[tuple[int, ...], ...]:
    velocities = np.asarray(quadrature.velocities)
    routes: list[tuple[int, ...]] = []
    for axis in range(quadrature.dimension):
        reflected = velocities.copy()
        reflected[:, axis] *= -1.0
        axis_routes = []
        for value in reflected:
            distances = np.max(np.abs(velocities - value[None, :]), axis=1)
            matches = np.flatnonzero(distances <= tolerance)
            if matches.size != 1:
                raise ValueError(
                    "Discrete-velocity reflection requires one unique mirrored velocity "
                    f"for every population on axis {axis}."
                )
            axis_routes.append(int(matches[0]))
        routes.append(tuple(axis_routes))
    return tuple(routes)


class DiscreteVelocityAdvectionSystem(AbstractAdmissibleSystem):
    """Diagonal conservative transport for one certified velocity quadrature."""

    quadrature: CertifiedDiscreteVelocityQuadrature
    reflection_routes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    population_floor: float = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        /,
        *,
        population_floor: float = 0.0,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        floor = float(population_floor)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("population_floor must be finite and non-negative.")
        routes = _reflection_routes(quadrature)
        self.quadrature = quadrature
        self.reflection_routes = routes
        self.population_floor = floor
        self.dimension = quadrature.dimension
        self.component_names = tuple(
            f"population_{index}" for index in range(quadrature.population_count)
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "discrete-velocity-advection-system-v1",
                "quadrature": quadrature.quadrature_id,
                "population_floor": floor,
                "reflection_routes": [list(route) for route in routes],
            }
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Discrete-velocity flux axis is out of range.")
        values = self.quadrature.validate_populations(state)
        return values * self.quadrature.velocities[:, axis_]

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> Array:
        del args
        axis_ = int(axis)
        left_ = self.quadrature.validate_populations(left)
        right_ = self.quadrature.validate_populations(right)
        if left_.shape != right_.shape or not 0 <= axis_ < self.dimension:
            raise ValueError("DVM face states or axis are incompatible.")
        speed = jnp.max(jnp.abs(self.quadrature.velocities[:, axis_]))
        return jnp.broadcast_to(speed, left_.shape[:-1])

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        axis_ = int(axis)
        left_ = self.quadrature.validate_populations(left)
        right_ = self.quadrature.validate_populations(right)
        if left_.shape != right_.shape or not 0 <= axis_ < self.dimension:
            raise ValueError("DVM face states or axis are incompatible.")
        velocities = self.quadrature.velocities[:, axis_]
        shape = left_.shape[:-1]
        return (
            jnp.broadcast_to(jnp.min(velocities), shape),
            jnp.broadcast_to(jnp.max(velocities), shape),
        )

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        normal: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        del args
        left_ = self.quadrature.validate_populations(left)
        right_ = self.quadrature.validate_populations(right)
        normal_ = jnp.asarray(normal)
        if (
            left_.shape != right_.shape
            or normal_.ndim == 0
            or normal_.shape[-1] != self.dimension
            or normal_.shape[:-1] != left_.shape[:-1]
        ):
            raise ValueError("DVM face states and normals have incompatible shapes.")
        normal_velocities = oe.contract(
            "...d,qd->...q", normal_, self.quadrature.velocities
        )
        return jnp.min(normal_velocities, axis=-1), jnp.max(normal_velocities, axis=-1)

    def conserved_to_primitive(self, state: Array, /) -> Array:
        return self.quadrature.validate_populations(state)

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        return self.quadrature.validate_populations(primitive)

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        values = self.quadrature.validate_populations(state)
        axis_ = int(axis)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Discrete-velocity reflection axis is out of range.")
        return jnp.take(values, jnp.asarray(self.reflection_routes[axis_]), axis=-1)

    def admissible(self, state: Array, /) -> Array:
        values = self.quadrature.validate_populations(state)
        return jnp.all(jnp.isfinite(values) & (values >= self.population_floor), axis=-1)


class ConservativeDVMSourceEvidence(StrictModule):
    """Declared-moment residual for one evaluated DVM source."""

    source: Array
    moment_residual: Array
    maximum_absolute_residual: Array


class AbstractConservativeDVMSource(StrictModule, NonTrainableState):
    """Finite-volume-compatible source with explicit invariant declarations."""

    quadrature: CertifiedDiscreteVelocityQuadrature
    moment_matrix: Array
    moment_names: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def __call__(
        self,
        time: Array,
        state: Array,
        coordinates: Array,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError

    def evidence(
        self,
        time: Array,
        state: Array,
        coordinates: Array,
        args: Any = None,
        /,
    ) -> ConservativeDVMSourceEvidence:
        source = self(time, state, coordinates, args)
        residual = oe.contract("mq,...q->...m", self.moment_matrix, source)
        return ConservativeDVMSourceEvidence(
            source=source,
            moment_residual=residual,
            maximum_absolute_residual=jnp.max(jnp.abs(residual)),
        )


class ConservativeRelaxationDVMSource(AbstractConservativeDVMSource):
    """Projected BGK-like relaxation that conserves every declared moment."""

    nullspace_projector: Array
    equilibrium: DVMEquilibrium = eqx.field(static=True)
    equilibrium_id: str = eqx.field(static=True)
    relaxation_rate: float = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        moment_matrix: ArrayLike,
        equilibrium: DVMEquilibrium,
        /,
        *,
        moment_names: Sequence[str],
        equilibrium_id: str,
        relaxation_rate: float,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        matrix = np.asarray(moment_matrix)
        names = tuple(str(value) for value in moment_names)
        identifier = str(equilibrium_id)
        rate = float(relaxation_rate)
        if matrix.ndim != 2 or matrix.shape[1] != quadrature.population_count:
            raise ValueError("moment_matrix must have shape (M, Q).")
        if (
            matrix.shape[0] == 0
            or len(names) != matrix.shape[0]
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("moment_names must uniquely label every moment row.")
        if not callable(equilibrium) or not identifier:
            raise ValueError(
                "A callable equilibrium and stable non-empty equilibrium_id are required."
            )
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError("relaxation_rate must be finite and non-negative.")
        if (
            np.any(~np.isfinite(matrix))
            or np.linalg.matrix_rank(matrix) != matrix.shape[0]
        ):
            raise ValueError(
                "Declared DVM moments must be finite and linearly independent."
            )
        gram = matrix @ matrix.T
        projector = np.eye(
            quadrature.population_count, dtype=matrix.dtype
        ) - matrix.T @ np.linalg.solve(gram, matrix)
        self.quadrature = quadrature
        self.moment_matrix = jnp.asarray(matrix)
        self.moment_names = names
        self.nullspace_projector = jnp.asarray(projector)
        self.equilibrium = equilibrium
        self.equilibrium_id = identifier
        self.relaxation_rate = rate
        self.source_id = canonical_fingerprint(
            {
                "kind": "conservative-relaxation-dvm-source-v1",
                "quadrature": quadrature.quadrature_id,
                "moment_matrix": array_tree_fingerprint(matrix),
                "moment_names": list(names),
                "equilibrium": identifier,
                "relaxation_rate": rate,
            }
        )

    def __call__(
        self,
        time: Array,
        state: Array,
        coordinates: Array,
        args: Any = None,
        /,
    ) -> Array:
        del time, coordinates
        values = self.quadrature.validate_populations(state)
        equilibrium = self.quadrature.validate_populations(self.equilibrium(values, args))
        if equilibrium.shape != values.shape:
            raise ValueError("DVM equilibrium must match the population field shape.")
        raw = self.relaxation_rate * (equilibrium - values)
        return oe.contract("pq,...q->...p", self.nullspace_projector, raw)


class DiscreteVelocitySourceComposition(AbstractConservativeDVMSource):
    """One conservative sum of sources sharing an exact moment contract."""

    sources: tuple[AbstractConservativeDVMSource, ...]

    def __init__(self, sources: Sequence[AbstractConservativeDVMSource], /):
        sources_ = tuple(sources)
        if not sources_ or any(
            not isinstance(source, AbstractConservativeDVMSource) for source in sources_
        ):
            raise TypeError("sources must contain conservative DVM source modules.")
        reference = sources_[0]
        reference_matrix = np.asarray(reference.moment_matrix)
        for source in sources_[1:]:
            if source.quadrature.quadrature_id != reference.quadrature.quadrature_id:
                raise ValueError("Composed DVM sources must use the same quadrature.")
            if source.moment_names != reference.moment_names or not np.array_equal(
                np.asarray(source.moment_matrix), reference_matrix
            ):
                raise ValueError(
                    "Composed DVM sources must declare identical conserved moments."
                )
        self.sources = sources_
        self.quadrature = reference.quadrature
        self.moment_matrix = reference.moment_matrix
        self.moment_names = reference.moment_names
        self.source_id = canonical_fingerprint(
            {
                "kind": "discrete-velocity-source-composition-v1",
                "sources": [source.source_id for source in sources_],
                "quadrature": reference.quadrature.quadrature_id,
                "moments": list(reference.moment_names),
            }
        )

    def __call__(
        self,
        time: Array,
        state: Array,
        coordinates: Array,
        args: Any = None,
        /,
    ) -> Array:
        values = self.quadrature.validate_populations(state)
        result = jnp.zeros_like(values)
        for source in self.sources:
            result = result + source(time, values, coordinates, args)
        return result


__all__ = [
    "AbstractConservativeDVMSource",
    "ConservativeDVMSourceEvidence",
    "ConservativeRelaxationDVMSource",
    "DiscreteVelocityAdvectionSystem",
    "DiscreteVelocitySourceComposition",
]
