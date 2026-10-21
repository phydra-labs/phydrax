#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bound repair of JAX point-index narrowing, installed before tracing.

The audited JAX source chooses gather/scatter and multi-axis dynamic-slice
index width from the operand shape, not the user index. Preserve the three
normalized address classes (below, within, above bounds) before conversion.
The original compiler still owns layout, slices, boolean expansion,
broadcasting, sharding, and operation modes. Compatibility is guarded by exact
source and code identity, not release labels. Two private compiler identities
intentionally change; public callables are never replaced.

Installation requires a fresh process with no initialized JAX backend. Functions
traced before Phydrax import are unsupported even if no backend was initialized;
callers must import Phydrax before tracing, not clear caches after installation.
Symbolic dimensions retain upstream width selection; bounds are staged through
dimension_as_value when needed. No device values are read on the host.
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import threading
from pathlib import Path
from types import CodeType, FunctionType
from typing import Any


_UPSTREAM_SOURCE_SHA256 = (
    "85ed8552ad648cdae8d0e70b280381adad87a7ee5424e0f6287e076091caa6bc"
)
_UPSTREAM_FUNCTION_SHA256 = (
    "3c3135e9a98d915edace7506c4c8fe536518c1f54f75f1e5cee2f0c45edf51d2"
)
_UPSTREAM_DYNAMIC_SLICE_SHA256 = (
    "7c04da9f492fc428b55462343482b64bbe2293a8e8fdd64ee973e8a9f6225030"
)
_ALGORITHM = "point-index-bounds-before-shape-width-narrowing"
_LOCK = threading.RLock()
_INSTALLED: FunctionType | None = None
_INSTALLED_DYNAMIC: FunctionType | None = None
_IDENTITY: dict[str, str] | None = None


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_upstream(indexing: Any) -> tuple[FunctionType, FunctionType]:
    """Reject source changes, wrappers, and code replacement, not version labels."""
    path = Path(indexing.__file__)
    source = path.read_bytes()
    if _digest(source) != _UPSTREAM_SOURCE_SHA256:
        raise RuntimeError(
            "Unrecognized JAX indexing source; refusing frontend index patch."
        )

    # Compile verified source only to compare code objects; never execute it.
    # Source text alone cannot detect a foreign replacement of function.__code__.
    compiled = compile(
        source, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize
    )
    codes = {
        code.co_name: code for code in compiled.co_consts if isinstance(code, CodeType)
    }
    indexer_code = codes["NDIndexer"]
    methods = {
        code.co_name: code
        for code in indexer_code.co_consts
        if isinstance(code, CodeType)
    }

    def require_function(function: Any, expected: CodeType) -> None:
        if (
            not isinstance(function, FunctionType)
            or hasattr(function, "__wrapped__")
            or function.__globals__ is not vars(indexing)
            or function.__code__ != expected
        ):
            raise RuntimeError(
                "Foreign JAX indexing callable; refusing frontend index patch."
            )

    original = indexing._index_to_gather
    require_function(original, codes["_index_to_gather"])
    if _digest(inspect.getsource(original).encode()) != _UPSTREAM_FUNCTION_SHA256:
        raise RuntimeError(
            "Unrecognized JAX index compiler; refusing frontend index patch."
        )
    for name in (
        "validate_slices",
        "convert_sequences_to_arrays",
        "expand_ellipses",
        "expand_scalar_bool_indices",
        "normalize_indices",
        "is_advanced_int_indexer",
        "to_dynamic_slice",
        "has_partial_slices",
    ):
        require_function(getattr(indexing.NDIndexer, name), methods[name])
    original_dynamic = indexing.NDIndexer.to_dynamic_slice
    if (
        _digest(inspect.getsource(original_dynamic).encode())
        != _UPSTREAM_DYNAMIC_SLICE_SHA256
    ):
        raise RuntimeError(
            "Unrecognized JAX dynamic slice compiler; refusing frontend index patch."
        )
    return original, original_dynamic


def _make_patch(original: FunctionType, indexing: Any) -> FunctionType:
    import numpy as np
    from jax import lax
    from jax._src import core

    point_types = (indexing.IndexType.INTEGER, indexing.IndexType.ARRAY)

    def bounded_index_to_gather(
        indexer: Any, *, x_sharding: Any, normalize_indices: bool = True
    ) -> Any:
        if not any(index.typ in point_types for index in indexer.indices):
            return original(
                indexer, x_sharding=x_sharding, normalize_indices=normalize_indices
            )
        indexer.validate_slices()
        indexer = indexer.convert_sequences_to_arrays()
        target = np.iinfo(
            indexing.lax_utils.int_dtype_for_shape(indexer.shape, signed=True)
        )
        target_bits = target.bits

        def needs_bounds(index: Any) -> bool:
            if index.typ == indexing.IndexType.INTEGER:
                return not target.min <= int(index.index) <= target.max
            if index.typ != indexing.IndexType.ARRAY:
                return False
            dtype = np.dtype(index.index.dtype)
            return dtype.itemsize * 8 > target_bits or (
                dtype.kind == "u" and dtype.itemsize * 8 == target_bits
            )

        risky = tuple(needs_bounds(index) for index in indexer.indices)
        if not any(risky):
            return original(
                indexer, x_sharding=x_sharding, normalize_indices=normalize_indices
            )
        # Original normalization uses consumed_axes, so it does not require
        # ellipsis/scalar-bool expansion first. Keep those for the delegate:
        # expanding here would change advanced-axis contiguity bookkeeping.
        # A symbolic dimension with a narrow default integer dtype cannot
        # represent a wider negative host scalar during normalization.
        for index, risk in zip(indexer.indices, risky, strict=True):
            if (
                risk
                and index.typ == indexing.IndexType.INTEGER
                and int(index.index) < 0
                and not core.is_constant_dim(indexer.shape[index.consumed_axes[0]])
            ):
                raise NotImplementedError(
                    "A wide negative host index with a symbolic dimension requires "
                    "JAX x64 mode or an explicitly typed array index."
                )
        if normalize_indices:
            indexer = indexer.normalize_indices()
        indices = []
        for index, risk in zip(indexer.indices, risky, strict=True):
            if not risk:
                indices.append(index)
                continue
            (axis,) = index.consumed_axes
            size = indexer.shape[axis]
            if index.typ == indexing.IndexType.INTEGER and core.is_constant_dim(size):
                bounded = min(max(int(index.index), -1), int(size))
            else:
                value = indexing.lax_numpy.asarray(index.index)
                # Both comparisons happen in the original signed/unsigned
                # dtype. -1 and size remain out of bounds after narrowing:
                # clip chooses the correct edge, while fill/drop reject them.
                upper = (
                    np.asarray(size, dtype=value.dtype)
                    if core.is_constant_dim(size)
                    else lax.convert_element_type(
                        core.dimension_as_value(size), value.dtype
                    )
                )
                bounded = lax.min(value, lax.full_like(value, upper))
                if np.dtype(value.dtype).kind == "i":
                    bounded = lax.max(bounded, lax.full_like(value, -1))
            indices.append(
                indexing.ParsedIndex(
                    index=bounded, typ=index.typ, consumed_axes=index.consumed_axes
                )
            )
        bounded_indexer = indexing.NDIndexer(shape=indexer.shape, indices=indices)
        return original(bounded_indexer, x_sharding=x_sharding, normalize_indices=False)

    return bounded_index_to_gather


def _make_dynamic_patch(original: FunctionType, indexing: Any) -> FunctionType:
    import numpy as np
    from jax import lax
    from jax._src import core

    def bounded_to_dynamic_slice(
        self: Any,
        *,
        arr_is_sharded: bool = False,
        normalize_indices: bool = True,
        mode: Any,
    ) -> Any:
        # One-axis dynamic slices do not narrow indices. Fill/drop and
        # nonscalar arrays keep the original rejection route so AUTO indexing
        # falls back to the gather compiler.
        parsed_mode = (
            None if mode is None else indexing.slicing.GatherScatterMode.from_any(mode)
        )
        if (
            len(self.shape) <= 1
            or parsed_mode
            not in (
                None,
                indexing.slicing.GatherScatterMode.CLIP,
                indexing.slicing.GatherScatterMode.PROMISE_IN_BOUNDS,
            )
            or any(
                index.typ == indexing.IndexType.ARRAY and np.ndim(index.index) != 0
                for index in self.indices
            )
        ):
            return original(
                self,
                arr_is_sharded=arr_is_sharded,
                normalize_indices=normalize_indices,
                mode=mode,
            )
        target = np.iinfo(indexing.lax_utils.int_dtype_for_shape(self.shape, signed=True))
        indices = []
        changed = False
        for index in self.indices:
            if index.typ == indexing.IndexType.INTEGER:
                risky = not target.min <= int(index.index) <= target.max
            elif index.typ == indexing.IndexType.ARRAY:
                dtype = np.dtype(index.index.dtype)
                risky = dtype.itemsize * 8 > target.bits or (
                    dtype.kind == "u" and dtype.itemsize * 8 == target.bits
                )
            else:
                risky = False
            if not risky:
                indices.append(index)
                continue
            changed = True
            (axis,) = index.consumed_axes
            size = self.shape[axis]
            # Dynamic slice normalizes later, after its dtype conversion.
            # Encode below-bounds as -size-1 (or -1 with wrapping disabled),
            # so normalization occurs exactly once and still clips to zero.
            if index.typ == indexing.IndexType.INTEGER and core.is_constant_dim(size):
                lower = -int(size) - 1 if normalize_indices else -1
                bounded = min(max(int(index.index), lower), int(size))
            else:
                value = indexing.lax_numpy.asarray(index.index)
                upper = (
                    np.asarray(size, dtype=value.dtype)
                    if core.is_constant_dim(size)
                    else lax.convert_element_type(
                        core.dimension_as_value(size), value.dtype
                    )
                )
                bounded = lax.min(value, lax.full_like(value, upper))
                if np.dtype(value.dtype).kind == "i":
                    lower = -upper - 1 if normalize_indices else -1
                    bounded = lax.max(bounded, lax.full_like(value, lower))
            indices.append(
                indexing.ParsedIndex(
                    index=bounded, typ=index.typ, consumed_axes=index.consumed_axes
                )
            )
        indexer = (
            indexing.NDIndexer(shape=self.shape, indices=indices) if changed else self
        )
        return original(
            indexer,
            arr_is_sharded=arr_is_sharded,
            normalize_indices=normalize_indices,
            mode=mode,
        )

    return bounded_to_dynamic_slice


def install_frontend_index_patch() -> None:
    """Install once before backend initialization; reject all foreign identities."""
    global _INSTALLED, _INSTALLED_DYNAMIC, _IDENTITY
    with _LOCK:
        import jax
        from jax._src import xla_bridge
        from jax._src.numpy import indexing

        if _INSTALLED is not None:
            if (
                indexing._index_to_gather is not _INSTALLED
                or indexing.NDIndexer.to_dynamic_slice is not _INSTALLED_DYNAMIC
            ):
                raise RuntimeError("The installed frontend index patch was replaced.")
            return
        original, original_dynamic = _validate_upstream(indexing)
        if xla_bridge.backends_are_initialized():
            raise RuntimeError(
                "Frontend index patch installation requires a fresh process: "
                "import Phydrax before initializing a JAX backend or tracing functions."
            )
        patch = _make_patch(original, indexing)
        dynamic_patch = _make_dynamic_patch(original_dynamic, indexing)
        identity = {
            "algorithm": _ALGORITHM,
            "jax_version": jax.__version__,
            "upstream_source_sha256": _UPSTREAM_SOURCE_SHA256,
            "upstream_function_sha256": _UPSTREAM_FUNCTION_SHA256,
            "upstream_dynamic_slice_sha256": _UPSTREAM_DYNAMIC_SLICE_SHA256,
            "patch_source_sha256": _digest(Path(__file__).read_bytes()),
        }
        # Dynamic FunctionType hooks retain the guarded upstream signatures.
        indexing._index_to_gather = patch  # ty: ignore[invalid-assignment]
        indexing.NDIndexer.to_dynamic_slice = dynamic_patch  # ty: ignore[invalid-assignment]
        _INSTALLED = patch
        _INSTALLED_DYNAMIC = dynamic_patch
        _IDENTITY = identity


def frontend_index_patch_identity() -> dict[str, str] | None:
    """Return a host-only identity snapshot, without importing JAX or discovering devices."""
    with _LOCK:
        if _INSTALLED is None:
            return None
        from jax._src.numpy import indexing

        if (
            indexing._index_to_gather is not _INSTALLED
            or indexing.NDIndexer.to_dynamic_slice is not _INSTALLED_DYNAMIC
        ):
            raise RuntimeError("The installed frontend index patch was replaced.")
        assert _IDENTITY is not None
        return dict(_IDENTITY)


__all__ = ["frontend_index_patch_identity", "install_frontend_index_patch"]
