"""Eager CPU Qlat Wilson gauge kernels with explicit host interchange.

Qlat's public Hamiltonian includes the Wilson plaquette constant.
Qlat uses first-coordinate-fastest sites, unlike the canonical C-order lattice.
The Python package does not expose a Wilson Dirac multiplication primitive.
"""

from __future__ import annotations

from typing import Any, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from ._types import BackendAvailability
from .lattice import (
    LatticeKernelCapabilities,
    LatticeKernelCapabilityError,
    LatticeProviderStatus,
)


CAPABILITIES = LatticeKernelCapabilities(
    "qlat",
    ("gauge.wilson_action", "gauge.wilson_force"),
    ("complex128",),
    ("single_device",),
    (),
    native_complex=True,
)


def adapter_availability(discovery: BackendAvailability) -> BackendAvailability:
    if not discovery.available:
        return discovery
    import qlat as q
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() != 1:
        return BackendAvailability(
            capabilities=discovery.capabilities,
            available=False,
            requirement=discovery.requirement,
            reason="Qlat adapter implements single-process periodic gauge fields only",
            versions=discovery.versions,
            environment_issues=discovery.environment_issues,
        )
    if not callable(q.gf_hamilton_node) or not callable(q.set_gm_force):
        return BackendAvailability(
            capabilities=discovery.capabilities,
            available=False,
            requirement=discovery.requirement,
            reason="Qlat native gauge action/force entry points are not callable",
            versions=discovery.versions,
            environment_issues=discovery.environment_issues,
        )
    return BackendAvailability(
        capabilities=discovery.capabilities,
        available=True,
        requirement=discovery.requirement,
        reason="Qlat native CPU Wilson gauge action and force; eager complex128, periodic 4D SU(3), single process",
        versions=discovery.versions,
        environment_issues=discovery.environment_issues,
    )


def _gauge_field(
    links: ArrayLike,
    lattice_shape: tuple[int, ...],
    periodic: tuple[bool, ...],
    operation: str,
) -> Any:
    import qlat as q

    if isinstance(links, jax_core.Tracer):
        raise LatticeKernelCapabilityError("qlat", operation, "eager host execution only")
    values = np.asarray(jax.device_get(links))
    CAPABILITIES.require(operation, values.dtype)
    if (
        len(lattice_shape) != 4
        or any(extent < 2 for extent in lattice_shape)
        or periodic != (True, True, True, True)
        or values.shape != (np.prod(lattice_shape), 4, 3, 3)
    ):
        raise LatticeKernelCapabilityError(
            "qlat",
            operation,
            "requires periodic 4D SU(3) links with extents at least two",
        )
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() != 1:
        raise LatticeKernelCapabilityError("qlat", operation, "requires one MPI process")
    if not q.is_initialized():
        q.begin_with_mpi()
    if q.get_num_node() != 1:
        raise LatticeKernelCapabilityError("qlat", operation, "requires one Qlat node")
    field = q.GaugeField(q.Geometry(q.Coordinate(list(lattice_shape))))
    np.asarray(field)[:] = (
        values.reshape((*lattice_shape, 4, 3, 3))
        .transpose(3, 2, 1, 0, 4, 5, 6)
        .reshape(values.shape)
    )
    return field


class QlatLatticeProvider:
    """Native Qlat gauge kernels; no emulated fermion or transformation support."""

    __slots__ = ()
    provider_id = "qlat"
    capabilities = CAPABILITIES

    def status(self) -> LatticeProviderStatus:
        from .lattice import qlat_availability

        discovery = qlat_availability()
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
        import qlat as q

        field = _gauge_field(links, lattice_shape, periodic, "gauge.wilson_action")
        mask = np.asarray(jax.device_get(owned_mask), dtype=np.bool_)
        if mask.shape != (np.prod(lattice_shape), 6) or not np.all(mask):
            raise LatticeKernelCapabilityError(
                "qlat", "gauge.wilson_action", "requires a fully owned plaquette mask"
            )
        beta_ = np.asarray(jax.device_get(beta), dtype=np.float64).item()
        action = q.gf_hamilton_node(field, q.GaugeAction(beta_, 0.0))
        return jnp.asarray(action, dtype=jnp.float64)

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
        import qlat as q

        field = _gauge_field(links, lattice_shape, periodic, "gauge.wilson_force")
        mask = np.asarray(jax.device_get(owned_mask), dtype=np.bool_)
        if mask.shape != (np.prod(lattice_shape), 4):
            raise ValueError("owned_mask must match the site/direction link axes")
        beta_ = np.asarray(jax.device_get(beta), dtype=np.float64).item()
        momentum = q.GaugeMomentum(field.geo)
        q.set_gm_force(momentum, field, q.GaugeAction(beta_, 0.0))
        force = (
            np.asarray(momentum)
            .reshape((*reversed(lattice_shape), 4, 3, 3))
            .transpose(3, 2, 1, 0, 4, 5, 6)
            .reshape((*mask.shape, 3, 3))
        )
        return jnp.asarray(
            np.where(mask[..., None, None], force, 0), dtype=jnp.complex128
        )

    def wilson_dslash(
        self, operator: Any, spinor: ArrayLike, /, *, adjoint: bool = False
    ) -> NoReturn:
        raise LatticeKernelCapabilityError(
            "qlat",
            "fermion.wilson_dslash",
            "Qlat exposes inversion, not Wilson Dirac multiplication",
        )

    def clover_dslash(
        self, operator: Any, spinor: ArrayLike, /, *, adjoint: bool = False
    ) -> NoReturn:
        raise LatticeKernelCapabilityError(
            "qlat", "fermion.clover_dslash", "Qlat exposes no clover Dirac multiplication"
        )

    def block_dslash(
        self, operator: Any, spinor: ArrayLike, /, *, adjoint: bool = False
    ) -> NoReturn:
        raise LatticeKernelCapabilityError(
            "qlat", "fermion.block_dslash", "Qlat exposes no block Dirac multiplication"
        )
