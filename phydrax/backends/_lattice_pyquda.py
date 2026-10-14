"""PyQUDA execution over canonical Phydrax fields.

Eager CUDA fields cross via DLPack; jit uses an explicit host-staged callback.
QUDA's resident gauge is process-global, so calls are serialized.
"""

from collections.abc import Callable
from math import prod
from threading import RLock
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._strict import StrictModule
from ._types import BackendAvailability
from .lattice import (
    LatticeKernelCapabilities,
    LatticeKernelCapabilityError,
    LatticeProviderStatus,
)


CAPABILITIES = LatticeKernelCapabilities(
    "pyquda",
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
_LOCK = RLock()
_INITIALIZED = False


def _initialize() -> None:
    global _INITIALIZED
    import pyquda

    if not _INITIALIZED:
        pyquda.init(
            [1, 1, 1, 1],
            backend="cupy",
            enable_p2p=0,
            enable_gdr=False,
            enable_nvshmem=False,
        )
        if pyquda.getMPISize() != 1 or pyquda.getArrayBackend() != "cupy":
            raise RuntimeError(
                "Phydrax PyQUDA requires a single MPI rank with the CuPy backend."
            )
        _INITIALIZED = True


def adapter_availability(discovery: BackendAvailability) -> BackendAvailability:
    if not discovery.available:
        return discovery
    try:
        import cupy as cp
        from pyquda import quda
        from pyquda_utils import core

        if cp.cuda.runtime.getDeviceCount() < 1 or not any(
            device.platform == "gpu" for device in jax.devices()
        ):
            raise RuntimeError("PyQUDA requires a CUDA device shared with JAX.")
        # Loading these public native entry points establishes ABI availability.
        if not callable(quda.dslashQuda) or not callable(core.getWilson):
            raise RuntimeError("PyQUDA Wilson runtime entry points are absent.")
        with _LOCK:
            _initialize()
    except (ImportError, OSError, RuntimeError) as error:
        return BackendAvailability(
            capabilities=discovery.capabilities,
            available=False,
            requirement=discovery.requirement,
            reason=f"PyQUDA adapter runtime unavailable: {error}",
            versions=discovery.versions,
        )
    return BackendAvailability(
        capabilities=discovery.capabilities,
        available=True,
        requirement=discovery.requirement,
        reason="PyQUDA CUDA adapter initialized; single-rank periodic 4D SU(3), r=1; jit uses host staging, autodiff is not supported",
        versions=discovery.versions,
    )


def _device(value: Any) -> Any:
    import cupy as cp

    if isinstance(value, Array) and value.device.platform == "gpu":
        return cp.from_dlpack(value)
    return cp.asarray(value)


def _execute(
    callback: Callable[..., Any], spec: jax.ShapeDtypeStruct, *values: Array
) -> Array:
    import cupy as cp

    if any(isinstance(value, jax_core.Tracer) for value in values):

        def host(*args: np.ndarray) -> np.ndarray:
            with _LOCK:
                result = callback(*args)
                return cp.asnumpy(result).astype(spec.dtype, copy=False)

        return jax.pure_callback(host, spec, *values, vmap_method="sequential")
    with _LOCK:
        result = callback(*values)
        result = cp.asarray(result, dtype=spec.dtype)
        return jax.dlpack.from_dlpack(result)


def _geometry(shape: tuple[int, ...], periodic: tuple[bool, ...]) -> None:
    if (
        len(shape) != 4
        or any(size < 2 or size % 2 for size in shape)
        or periodic != (True,) * 4
    ):
        raise LatticeKernelCapabilityError(
            "pyquda",
            "lattice.geometry",
            "requires periodic four-dimensional even extents",
        )


def _minimal(dirac: Any, dtype: np.dtype) -> None:
    from pyquda.enum_quda import QudaPrecision, QudaReconstructType

    precision = (
        QudaPrecision.QUDA_DOUBLE_PRECISION
        if dtype == np.dtype(np.complex128)
        else QudaPrecision.QUDA_SINGLE_PRECISION
    )
    keys = ("cuda", "sloppy", "refinement_sloppy", "precondition", "eigensolver")
    dirac.setPrecision(**dict.fromkeys(keys, precision))
    dirac.setReconstruct(**dict.fromkeys(keys, QudaReconstructType.QUDA_RECONSTRUCT_NO))


def _gauge(links: Any, shape: tuple[int, ...]) -> tuple[Any, Any]:
    import cupy as cp
    from pyquda.field import LatticeGauge, LatticeInfo

    info = LatticeInfo(list(reversed(shape)), 1, 1.0)
    # Canonical axes become QUDA t,z,y,x; direction mu is therefore 3-mu.
    values = _device(links).reshape(shape + (4, 3, 3))
    values = cp.moveaxis(values, 4, 0)[::-1].astype(cp.complex128)
    return info, LatticeGauge(info, info.evenodd(values, True, "cupy"))


class PyqudaLatticeProvider(StrictModule):
    provider_id: ClassVar[str] = "pyquda"
    capabilities: ClassVar[LatticeKernelCapabilities] = CAPABILITIES

    def __init__(self) -> None:
        from .lattice import pyquda_availability

        pyquda_availability().require("gauge.wilson_action")

    def status(self) -> LatticeProviderStatus:
        from .lattice import pyquda_availability

        evidence = pyquda_availability()
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
        import cupy as cp
        from pyquda.action import GaugeAction
        from pyquda.action.abstract import LoopParam
        from pyquda.field import LatticeMom

        from ._lattice_geometry import canonical_links

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

        def run(u: Any, b: Any, m: Any) -> Any:
            info, gauge = _gauge(u, lattice_shape)
            paths = [
                [mu, nu, mu + 4, nu + 4] for mu in range(4) for nu in range(mu + 1, 4)
            ]
            action = GaugeAction(info, LoopParam(paths, [1.0] * 6), float(np.asarray(b)))
            _minimal(action.dirac, np.dtype(values.dtype))
            with action.dirac.useGauge(gauge):
                if not force:
                    if not bool(cp.all(_device(m))):
                        raise LatticeKernelCapabilityError(
                            self.provider_id,
                            operation,
                            "action requires a fully owned periodic lattice",
                        )
                    # Initializes extendedGaugeResident before loop-trace on QUDA ba501e4.
                    action.dirac.plaquette()
                    return cp.asarray(
                        action.action() + float(np.asarray(b)) * 6 * info.volume
                    )
                momentum = LatticeMom(info)
                action.force(1.0, momentum)
            raw = momentum.lexico(False)
            decoded = cp.zeros(raw.shape[:-1] + (3, 3), dtype=values.dtype)
            for index, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
                decoded[..., i, j] = raw[..., 2 * index] + 1j * raw[..., 2 * index + 1]
                decoded[..., j, i] = -decoded[..., i, j].conj()
            for color in range(3):
                decoded[..., color, color] = 1j * raw[..., 6 + color]
            canonical = cp.moveaxis(decoded[::-1], 0, 4).reshape(values.shape)
            return cp.where(_device(m)[..., None, None], canonical, 0)

        spec = jax.ShapeDtypeStruct(
            values.shape if force else (), values.dtype if force else values.real.dtype
        )
        return _execute(run, spec, values, coefficient, mask)

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
        self, operator: Any, spinor: ArrayLike, *, adjoint: bool, operation: str
    ) -> Array:
        import cupy as cp
        from pyquda.enum_quda import QudaDagType
        from pyquda.field import LatticeFermion, MultiLatticeFermion
        from pyquda.gamma import DeGrandRossiMatrix
        from pyquda_utils import core

        from ..operators.path_integral._lattice_fermion import (
            CloverWilsonDiracOperator,
            WilsonDiracOperator,
        )

        if isinstance(operator, CloverWilsonDiracOperator):
            wilson = operator.wilson
            clover = operator.clover_term
        elif isinstance(operator, WilsonDiracOperator):
            wilson = operator
            clover = jnp.zeros((0,), dtype=operator.links.dtype)
        else:
            raise TypeError(
                "PyQUDA requires a canonical Wilson or clover-Wilson operator."
            )
        _geometry(wilson.lattice_shape, wilson.boundary.topology.periodic)
        if wilson.wilson_parameter != 1.0 or wilson.color_components != 3:
            raise LatticeKernelCapabilityError(
                self.provider_id,
                operation,
                "requires Wilson r=1 and SU(3) fundamental color",
            )
        values = jnp.asarray(spinor)
        self.capabilities.require(operation, values.dtype)
        if values.shape[:3] != (wilson.site_count, 4, 3) or values.ndim not in (3, 4):
            raise ValueError("Spinor must have shape (site,4,3[,rhs]).")
        links = wilson.links[wilson.transport.forward_edges]

        def run(u: Any, x: Any, gamma: Any, phases: Any, local: Any) -> Any:
            info, gauge = _gauge(u, wilson.lattice_shape)
            phase = cp.asarray(phases)
            coordinates = cp.indices(wilson.lattice_shape)
            lex = cp.moveaxis(gauge.lexico(False)[::-1], 0, 4)
            for axis in range(4):
                lex[..., axis, :, :] *= cp.where(
                    coordinates[axis] == wilson.lattice_shape[axis] - 1, phase[axis], 1
                )[..., None, None]
            gauge.data = info.evenodd(cp.moveaxis(lex, 4, 0)[::-1].copy(), True, "cupy")
            target = np.stack(
                (
                    DeGrandRossiMatrix.gamma_4,
                    DeGrandRossiMatrix.gamma_3,
                    DeGrandRossiMatrix.gamma_2,
                    DeGrandRossiMatrix.gamma_1,
                )
            )
            source_gamma = cp.asnumpy(_device(gamma))
            # Solve G_quda B = B G_phydrax once at this explicit host boundary.
            equations = np.concatenate(
                [
                    np.kron(np.eye(4), target[axis])
                    - np.kron(source_gamma[axis].T, np.eye(4))
                    for axis in range(4)
                ]
            )
            _, _, vh = np.linalg.svd(equations)
            basis = vh[-1].conj().reshape((4, 4), order="F") * 2
            if np.max(np.abs(target @ basis - basis @ source_gamma)) > 1e-5:
                raise LatticeKernelCapabilityError(
                    self.provider_id,
                    operation,
                    "gamma basis is not unitarily equivalent to QUDA",
                )
            basis_device = cp.asarray(basis, dtype=values.dtype)
            original = _device(x)
            block = original.ndim == 4
            transformed = cp.einsum("ab,sbc...->sac...", basis_device, original).astype(
                cp.complex128
            )
            if block:
                lex_source = cp.moveaxis(transformed, -1, 0).reshape(
                    (values.shape[-1],) + wilson.lattice_shape + (4, 3)
                )
                source = MultiLatticeFermion(
                    info, values.shape[-1], info.evenodd(lex_source, True, "cupy")
                )
            else:
                source = LatticeFermion(
                    info,
                    info.evenodd(
                        transformed.reshape(wilson.lattice_shape + (4, 3)), False, "cupy"
                    ),
                )
            dirac = core.getWilson(info, wilson.mass * wilson.lattice_spacing, 1e-10, 10)
            _minimal(dirac, np.dtype(values.dtype))
            dirac.invert_param.dagger = (
                QudaDagType.QUDA_DAG_YES if adjoint else QudaDagType.QUDA_DAG_NO
            )
            with dirac.useGauge(gauge):
                hopping = (
                    dirac.dslashMultiSrc(source) if block else dirac.dslash(source)
                ).lexico(False)
            if block:
                hopping = cp.moveaxis(
                    hopping.reshape((values.shape[-1], wilson.site_count, 4, 3)), 0, -1
                )
            else:
                hopping = hopping.reshape(values.shape)
            hopping = cp.einsum("ab,sbc...->sac...", basis_device.conj().T, hopping)
            result = (wilson.mass + 4 / wilson.lattice_spacing) * original - hopping * (
                0.5 / wilson.lattice_spacing
            )
            if local.size:
                matrices = _device(local).reshape((wilson.site_count, 12, 12))
                if adjoint:
                    matrices = matrices.conj().swapaxes(-1, -2)
                vector = original.reshape((wilson.site_count, 12, -1))
                result += (matrices @ vector).reshape(values.shape)
            return result.astype(values.dtype)

        return _execute(
            run,
            jax.ShapeDtypeStruct(values.shape, values.dtype),
            links,
            values,
            wilson.gamma_matrices,
            wilson.boundary.phases,
            clover,
        )

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
            operator, right_hand_sides, adjoint=adjoint, operation="fermion.block_dslash"
        )
