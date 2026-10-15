#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Eager CPU Grid execution with explicit host lattice interchange."""

from __future__ import annotations

from typing import Any, ClassVar, final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..metrix._gauge_representation import FundamentalGaugeRepresentation
from ..operators.path_integral._lattice_fermion import (
    CloverWilsonDiracOperator,
    WilsonDiracOperator,
)
from ._types import BackendAvailability
from .lattice import (
    LatticeKernelCapabilities,
    LatticeKernelCapabilityError,
    LatticeProviderStatus,
)


CAPABILITIES = LatticeKernelCapabilities(
    "gpt-grid",
    (
        "gauge.wilson_action",
        "gauge.wilson_force",
        "fermion.wilson_dslash",
        "fermion.clover_dslash",
        "fermion.block_dslash",
    ),
    ("complex64", "complex128"),
    ("single_device",),
    (),
    native_complex=True,
)


def adapter_availability(discovery: BackendAvailability) -> BackendAvailability:
    if not discovery.available:
        return discovery
    # Optional native-extension load/allocation failures belong to availability,
    # not a successful provider declaration.
    try:
        import gpt as g

        grid = g.grid([4, 4, 4, 4], g.single)
        field = g.mcolor(grid)
        field[:] = 0
        g.norm2(field)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return BackendAvailability(
            capabilities=discovery.capabilities,
            available=False,
            requirement=discovery.requirement,
            reason=f"GPT native Grid runtime is unavailable: {type(error).__name__}: {error}",
            versions=discovery.versions,
            environment_issues=discovery.environment_issues,
        )
    return BackendAvailability(
        capabilities=discovery.capabilities,
        available=True,
        requirement=discovery.requirement,
        reason="GPT native Grid extension allocated and reduced a CPU lattice; eager host interchange",
        versions=discovery.versions,
        environment_issues=discovery.environment_issues,
    )


def _host(value: ArrayLike, operation: str) -> np.ndarray:
    if isinstance(value, jax_core.Tracer):
        raise LatticeKernelCapabilityError(
            "gpt-grid",
            operation,
            "eager CPU host interchange does not support JAX transformations",
        )
    return np.asarray(jax.device_get(value))


def _geometry(
    shape: tuple[int, ...], periodic: tuple[bool, ...], operation: str
) -> np.ndarray:
    if (
        len(shape) != 4
        or any(size < 2 or size % 2 for size in shape)
        or periodic != (True,) * 4
    ):
        raise LatticeKernelCapabilityError(
            "gpt-grid",
            operation,
            "requires an even, periodic four-dimensional SU(3) lattice",
        )
    return np.stack(np.unravel_index(np.arange(np.prod(shape)), shape), axis=-1).astype(
        np.int32
    )


def _gauge(
    links: np.ndarray, shape: tuple[int, ...], coordinates: np.ndarray
) -> tuple[Any, list[Any]]:
    import gpt as g

    if links.shape != (np.prod(shape), 4, 3, 3):
        raise ValueError("Grid gauge links must have shape (sites, 4, 3, 3).")
    grid = g.grid(
        list(shape), g.single if links.dtype == np.dtype(np.complex64) else g.double
    )
    fields = [g.mcolor(grid) for _ in range(4)]
    for direction, field in enumerate(fields):
        field[coordinates] = np.ascontiguousarray(links[:, direction])
    return grid, fields


@final
class GridLatticeProvider(StrictModule):
    provider_id: ClassVar[str] = "gpt-grid"
    capabilities: ClassVar[LatticeKernelCapabilities] = CAPABILITIES

    def status(self) -> LatticeProviderStatus:
        from .lattice import grid_availability

        discovery = grid_availability()
        return LatticeProviderStatus(
            self.capabilities,
            available=discovery.available,
            reason=discovery.reason,
            versions=discovery.versions,
        )

    def gauge_action(
        self,
        links: ArrayLike,
        beta: ArrayLike,
        owned_mask: ArrayLike,
        /,
        *,
        lattice_shape: tuple[int, ...],
        periodic: tuple[bool, ...],
    ) -> Array:
        import gpt as g

        operation = "gauge.wilson_action"
        values = _host(links, operation)
        self.capabilities.require(operation, values.dtype)
        coordinates = _geometry(lattice_shape, periodic, operation)
        mask = _host(owned_mask, operation)
        if mask.shape != (values.shape[0], 6):
            raise ValueError("owned_mask must have shape (sites, 6).")
        if not np.all(mask):
            raise LatticeKernelCapabilityError(
                self.provider_id, operation, "action requires fully owned plaquettes"
            )
        _, fields = _gauge(values, lattice_shape, coordinates)
        result = g.qcd.gauge.action.wilson(float(_host(beta, operation)))(fields)
        return jnp.asarray(result, dtype=values.real.dtype)

    def gauge_force(
        self,
        links: ArrayLike,
        beta: ArrayLike,
        owned_mask: ArrayLike,
        /,
        *,
        lattice_shape: tuple[int, ...],
        periodic: tuple[bool, ...],
    ) -> Array:
        import gpt as g

        operation = "gauge.wilson_force"
        values = _host(links, operation)
        self.capabilities.require(operation, values.dtype)
        coordinates = _geometry(lattice_shape, periodic, operation)
        mask = _host(owned_mask, operation)
        if mask.shape != values.shape[:2]:
            raise ValueError("owned_mask must have shape (sites, 4).")
        _, fields = _gauge(values, lattice_shape, coordinates)
        staples = g.qcd.gauge.action.wilson(float(_host(beta, operation))).staples(fields)
        result = np.stack(
            [
                g(-g.qcd.gauge.project.traceless_anti_hermitian(g(u * g.adj(s))))[
                    coordinates
                ]
                for u, s in zip(fields, staples, strict=True)
            ],
            axis=1,
        ).astype(values.dtype)
        return jnp.asarray(np.where(mask[..., None, None], result, 0))

    def wilson_dslash(
        self,
        operator: WilsonDiracOperator,
        spinor: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> Array:
        import gpt as g

        operation = "fermion.wilson_dslash"
        values = _host(spinor, operation)
        self.capabilities.require(operation, values.dtype)
        shape = operator.lattice_shape
        coordinates = _geometry(shape, operator.boundary.topology.periodic, operation)
        if values.shape != (coordinates.shape[0], 4, 3):
            raise ValueError("Grid spinors must have shape (sites, 4, 3).")
        if not isinstance(operator.representation, FundamentalGaugeRepresentation):
            raise LatticeKernelCapabilityError(
                self.provider_id,
                operation,
                "requires the fundamental color representation",
            )
        sites = np.arange(coordinates.shape[0]).reshape(shape)
        expected_routes = np.stack(
            [np.roll(sites, -1, axis=axis).reshape(-1) for axis in range(4)], axis=1
        )
        if not np.array_equal(
            _host(operator.transport.forward_sites, operation), expected_routes
        ):
            raise LatticeKernelCapabilityError(
                self.provider_id,
                operation,
                "transport must follow canonical periodic nearest-neighbor routes",
            )
        links = _host(operator.links, operation)[
            _host(operator.transport.forward_edges, operation)
        ]
        orientations = _host(operator.transport.forward_orientations, operation)
        links = np.where(
            orientations[..., None, None] > 0, links, links.conj().swapaxes(-1, -2)
        )
        phases = _host(operator.boundary.phases, operation)
        links = links.astype(values.dtype)
        for axis, extent in enumerate(shape):
            links[coordinates[:, axis] == extent - 1, axis] *= phases[axis]
        grid, fields = _gauge(links, shape, coordinates)
        source = g.vspincolor(grid)
        source[coordinates] = np.ascontiguousarray(values)
        diagonal = (
            operator.mass + 4 * operator.wilson_parameter / operator.lattice_spacing
        )
        result = g(diagonal * source)
        gamma = _host(operator.gamma_matrices, operation)
        identity = np.eye(4, dtype=values.dtype)
        for axis in range(4):
            forward = g(fields[axis] * g.cshift(source, axis, 1))
            backward = g(g.cshift(g.adj(fields[axis]) * source, axis, -1))
            sign = -1 if adjoint else 1
            minus = g.mspin(operator.wilson_parameter * identity - sign * gamma[axis])
            plus = g.mspin(operator.wilson_parameter * identity + sign * gamma[axis])
            result -= (0.5 / operator.lattice_spacing) * (
                minus * forward + plus * backward
            )
        return jnp.asarray(result[coordinates], dtype=values.dtype)

    def clover_dslash(
        self,
        operator: CloverWilsonDiracOperator,
        spinor: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> Array:
        import gpt as g

        operation = "fermion.clover_dslash"
        values = _host(spinor, operation)
        self.capabilities.require(operation, values.dtype)
        wilson = self.wilson_dslash(operator.wilson, values, adjoint=adjoint)
        coordinates = _geometry(
            operator.lattice_shape, operator.boundary.topology.periodic, operation
        )
        grid = g.grid(
            list(operator.lattice_shape),
            g.single if values.dtype == np.dtype(np.complex64) else g.double,
        )
        source = g.vspincolor(grid)
        source[coordinates] = np.ascontiguousarray(values)
        block = g.mspincolor(grid)
        # GPT stores matrix spin-color axes (spin, spin, color, color).
        clover = _host(operator.clover_term, operation).transpose(0, 1, 3, 2, 4)
        block[coordinates] = np.ascontiguousarray(clover)
        correction = g((g.adj(block) if adjoint else block) * source)
        return wilson + jnp.asarray(correction[coordinates], dtype=values.dtype)

    def block_dslash(
        self,
        operator: WilsonDiracOperator | CloverWilsonDiracOperator,
        spinor: ArrayLike,
        /,
        *,
        adjoint: bool = False,
    ) -> Array:
        values = _host(spinor, "fermion.block_dslash")
        self.capabilities.require("fermion.block_dslash", values.dtype)
        if values.ndim != 4 or values.shape[-1] < 1:
            raise ValueError("Block spinors require a non-empty final RHS axis.")
        if isinstance(operator, CloverWilsonDiracOperator):
            outputs = [
                self.clover_dslash(operator, values[..., rhs], adjoint=adjoint)
                for rhs in range(values.shape[-1])
            ]
        else:
            outputs = [
                self.wilson_dslash(operator, values[..., rhs], adjoint=adjoint)
                for rhs in range(values.shape[-1])
            ]
        return jnp.stack(outputs, axis=-1)
