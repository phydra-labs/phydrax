"""Device-resident QUDA kernels over canonical Phydrax lattice fields.

Each pure FFI call reloads its gauge. Native code serializes QUDA's process-wide
resident state and refuses a second device. Autodiff is not supported.
"""

from math import prod
from typing import Any, ClassVar, final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._cuda_native import load_cuda_component
from .._strict import StrictModule
from ..ein import contract
from ..operators.path_integral._lattice_fermion import (
    CloverWilsonDiracOperator,
    WilsonDiracOperator,
)
from ._lattice_geometry import canonical_links
from .lattice import (
    LatticeKernelCapabilities,
    LatticeKernelCapabilityError,
    LatticeProviderStatus,
)


CAPABILITIES = LatticeKernelCapabilities(
    "quda",
    (
        "gauge.wilson_action",
        "gauge.wilson_force",
        "fermion.wilson_dslash",
        "fermion.clover_dslash",
        "fermion.block_dslash",
    ),
    ("complex64", "complex128"),
    ("single_device",),
    ("jit",),
    native_complex=True,
)


def _geometry(shape: tuple[int, ...], periodic: tuple[bool, ...]) -> None:
    if jax.process_count() != 1:
        raise LatticeKernelCapabilityError(
            "quda", "lattice.geometry", "packaged QUDA requires a single JAX process"
        )
    if (
        len(shape) != 4
        or any(size < 2 or size % 2 for size in shape)
        or periodic != (True,) * 4
    ):
        raise LatticeKernelCapabilityError(
            "quda", "lattice.geometry", "requires periodic four-dimensional even extents"
        )


def _site_indices(shape: tuple[int, ...], *, inverse: bool = False) -> Array:
    """Map x-fastest lexicographic sites to/from parity-major QUDA order."""
    volume = prod(shape)
    sites = jnp.arange(volume if inverse else volume // 2, dtype=jnp.int32)
    remainder = sites if inverse else 2 * sites
    parity = jnp.zeros_like(sites)
    for extent in reversed(shape):
        parity = parity + remainder % extent
        remainder = remainder // extent
    if inverse:
        return (parity % 2) * (volume // 2) + sites // 2
    return jnp.stack((2 * sites + parity % 2, 2 * sites + (1 - parity % 2)))


def _pack_gauge(values: Array, shape: tuple[int, ...]) -> Array:
    directions = jnp.moveaxis(values, 1, 0)[::-1]
    return jnp.take(directions, _site_indices(shape), axis=1)


def _unpack_gauge(values: Array, shape: tuple[int, ...]) -> Array:
    lex = jnp.take(
        values.reshape((4, prod(shape), 3, 3)),
        _site_indices(shape, inverse=True),
        axis=1,
    )
    return jnp.moveaxis(lex[::-1], 0, 1)


def _gamma_basis(source_gamma: Array) -> Array:
    """Unitary intertwiner from any validated Clifford basis to QUDA's basis.

    Clifford twirling projects each matrix unit onto the one-dimensional
    intertwiner space. At least one projection is nonzero; choose its largest
    norm and normalize it. Schur's lemma makes the result unitary. All numerical
    work remains on device, including when operator gamma leaves are traced.
    """
    # Canonical axis order is t,z,y,x, opposite QUDA's x,y,z,t directions.
    target_gamma = jnp.asarray(
        (
            ((0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0)),
            ((0, 0, 1j, 0), (0, 0, 0, -1j), (-1j, 0, 0, 0), (0, 1j, 0, 0)),
            ((0, 0, 0, -1), (0, 0, 1, 0), (0, 1, 0, 0), (-1, 0, 0, 0)),
            ((0, 0, 0, 1j), (0, 0, 1j, 0), (0, -1j, 0, 0), (-1j, 0, 0, 0)),
        ),
        dtype=source_gamma.dtype,
    )
    source = jnp.eye(4, dtype=source_gamma.dtype)[None]
    target = jnp.eye(4, dtype=source_gamma.dtype)[None]
    for axis in range(4):
        source = jnp.concatenate((source, source @ source_gamma[axis]), axis=0)
        target = jnp.concatenate((target, target @ target_gamma[axis]), axis=0)
    # P(E_pq)_ad = sum_k T_k[a,p] conj(S_k[d,q]) / 16.
    candidates = contract(
        "kap,kdq->pqad", target, jnp.conj(source), backend="jax"
    ).reshape((16, 4, 4))
    norms = jnp.sum(jnp.abs(candidates) ** 2, axis=(1, 2))
    index = jnp.argmax(norms)
    return candidates[index] * (2 / jnp.sqrt(norms[index]))


def _ffi(
    operation: str,
    values: Array,
    result: jax.ShapeDtypeStruct,
    *operands: Array,
    shape: tuple[int, ...],
    adjoint: bool | None = None,
) -> Array:
    load_cuda_component("quda")
    suffix = "c128" if values.dtype == np.dtype(np.complex128) else "c64"
    attributes = dict(
        zip(
            ("x", "y", "z", "t"),
            (np.int64(size) for size in reversed(shape)),
            strict=True,
        )
    )
    if adjoint is not None:
        attributes["dagger"] = np.int64(adjoint)
    return jax.ffi.ffi_call(
        f"phydrax_quda_{operation}_{suffix}",
        result,
        vmap_method="sequential",
        has_side_effect=False,
    )(values, *operands, **attributes)


@final
class QudaLatticeProvider(StrictModule):
    provider_id: ClassVar[str] = "quda"
    capabilities: ClassVar[LatticeKernelCapabilities] = CAPABILITIES

    def __init__(self) -> None:
        from .lattice import quda_availability

        quda_availability().require("gauge.wilson_action")
        if jax.process_count() != 1:
            raise LatticeKernelCapabilityError(
                self.provider_id,
                "lattice.geometry",
                "packaged QUDA requires a single JAX process",
            )

    def status(self) -> LatticeProviderStatus:
        from .lattice import quda_availability

        evidence = quda_availability()
        return LatticeProviderStatus(
            self.capabilities,
            available=evidence.available,
            reason=evidence.reason,
            versions=evidence.versions,
        )

    def _gauge_operation(
        self,
        links: ArrayLike,
        beta: ArrayLike,
        owned_mask: ArrayLike,
        *,
        lattice_shape: tuple[int, ...],
        periodic: tuple[bool, ...],
        force: bool,
    ) -> Array:
        _geometry(lattice_shape, periodic)
        values = canonical_links(links, lattice_shape, periodic)
        operation = "gauge.wilson_force" if force else "gauge.wilson_action"
        self.capabilities.require(operation, values.dtype)
        if values.shape[-2:] != (3, 3):
            raise LatticeKernelCapabilityError(
                self.provider_id, operation, "requires SU(3) links"
            )
        mask = jnp.asarray(owned_mask, dtype=jnp.bool_)
        expected = (prod(lattice_shape), 4 if force else 6)
        if mask.shape != expected:
            raise ValueError(f"owned_mask must have shape {expected}.")
        coefficient = jnp.asarray(beta, dtype=values.real.dtype)
        gauge = _pack_gauge(values, lattice_shape)
        if force:
            result = _ffi(
                "gauge_force",
                gauge,
                jax.ShapeDtypeStruct(gauge.shape, gauge.dtype),
                shape=lattice_shape,
            )
            canonical = _unpack_gauge(result, lattice_shape)
            return jnp.where(mask[..., None, None], coefficient * canonical, 0)
        # Native validation refuses partial ownership, including a dynamic mask
        # inside jit. QUDA's scalar plaquette reduction cannot represent it.
        result = _ffi(
            "gauge_action",
            gauge,
            jax.ShapeDtypeStruct((), values.real.dtype),
            mask,
            shape=lattice_shape,
        )
        return coefficient * result

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
        return self._gauge_operation(
            links,
            beta,
            owned_mask,
            lattice_shape=lattice_shape,
            periodic=periodic,
            force=False,
        )

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
        return self._gauge_operation(
            links,
            beta,
            owned_mask,
            lattice_shape=lattice_shape,
            periodic=periodic,
            force=True,
        )

    def _dslash(
        self,
        operator: Any,
        spinor: ArrayLike,
        *,
        adjoint: bool,
        operation: str,
    ) -> Array:
        if isinstance(operator, CloverWilsonDiracOperator):
            wilson = operator.wilson
        elif isinstance(operator, WilsonDiracOperator):
            wilson = operator
        else:
            raise TypeError("QUDA requires a canonical Wilson or clover-Wilson operator.")
        _geometry(wilson.lattice_shape, wilson.boundary.topology.periodic)
        if (
            wilson.wilson_parameter != 1.0
            or wilson.color_components != 3
            or wilson.spin_components != 4
        ):
            raise LatticeKernelCapabilityError(
                self.provider_id,
                operation,
                "requires Wilson r=1 and SU(3) fundamental color with four spin components",
            )
        values = jnp.asarray(spinor)
        self.capabilities.require(operation, values.dtype)
        if (
            values.shape[:3] != (wilson.site_count, 4, 3)
            or values.ndim not in (3, 4)
            or (values.ndim == 4 and values.shape[-1] < 1)
        ):
            raise ValueError("Spinor must have shape (site,4,3[,rhs]) with nonempty RHS.")
        if values.dtype != wilson.links.dtype:
            raise ValueError("Spinor and operator links must have the same dtype.")
        links = wilson.links[wilson.transport.forward_edges]
        coordinates = jnp.indices(wilson.lattice_shape).reshape((4, wilson.site_count)).T
        wrap = coordinates == jnp.asarray(wilson.lattice_shape, dtype=jnp.int32) - 1
        phases = jnp.where(wrap, wilson.boundary.phases[None], 1)
        gauge = _pack_gauge(links * phases[..., None, None], wilson.lattice_shape)
        basis = _gamma_basis(wilson.gamma_matrices)
        block = values if values.ndim == 4 else values[..., None]
        transformed = contract("ab,sbcr->rsac", basis, block, backend="jax")
        packed = jnp.take(transformed, _site_indices(wilson.lattice_shape), axis=1)
        hopping = _ffi(
            "wilson_dslash",
            gauge,
            jax.ShapeDtypeStruct(packed.shape, packed.dtype),
            packed,
            shape=wilson.lattice_shape,
            adjoint=adjoint,
        )
        hopping = jnp.take(
            hopping.reshape((block.shape[-1], wilson.site_count, 4, 3)),
            _site_indices(wilson.lattice_shape, inverse=True),
            axis=1,
        )
        hopping = contract("ba,rsbc->sacr", jnp.conj(basis), hopping, backend="jax")
        hopping = hopping if values.ndim == 4 else hopping[..., 0]

        def diagonal(vector: Array) -> Array:
            if adjoint:
                return operator.gamma5_mv(
                    operator.diagonal_mv(operator.gamma5_mv(vector))
                )
            return operator.diagonal_mv(vector)

        local = (
            jax.vmap(diagonal, in_axes=-1, out_axes=-1)(values)
            if values.ndim == 4
            else diagonal(values)
        )
        scale = jnp.asarray(0.5 / wilson.lattice_spacing, dtype=values.real.dtype)
        return local - scale * hopping

    def wilson_dslash(
        self, operator: Any, spinor: ArrayLike, /, *, adjoint: bool = False
    ) -> Array:
        return self._dslash(
            operator, spinor, adjoint=adjoint, operation="fermion.wilson_dslash"
        )

    def clover_dslash(
        self, operator: Any, spinor: ArrayLike, /, *, adjoint: bool = False
    ) -> Array:
        return self._dslash(
            operator, spinor, adjoint=adjoint, operation="fermion.clover_dslash"
        )

    def block_dslash(
        self, operator: Any, right_hand_sides: ArrayLike, /, *, adjoint: bool = False
    ) -> Array:
        return self._dslash(
            operator,
            right_hand_sides,
            adjoint=adjoint,
            operation="fermion.block_dslash",
        )
