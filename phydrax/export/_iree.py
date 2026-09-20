#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.domain import DomainFunction

from .._document_resource import decode_json_resource
from .._external_resource import read_bounded_resource, ResourceLimits
from .._fingerprint import canonical_fingerprint
from .._publication import publish_resource_set, ResourceSetPublicationReceipt
from .._resource_set import (
    read_bounded_resource_set,
    ResourceSetLimits,
)
from ..backends.iree import import_iree, iree_availability
from ._inference import make_inference_export_callable


_IREE_ARTIFACT_FORMAT = "phydrax-iree-inference"


def _sha256_bytes(value: bytes, /) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class IREEExportPolicy:
    """Static IREE compilation target and runtime driver."""

    target_backend: str = "llvm-cpu"
    runtime_driver: str = "local-task"

    def __post_init__(self) -> None:
        if not self.target_backend or not self.runtime_driver:
            raise ValueError("IREE target_backend and runtime_driver must be non-empty.")


@dataclass(frozen=True, slots=True)
class IREEArtifactManifest:
    """Canonical executable identity, ABI, and validation evidence."""

    format: str
    artifact_id: str
    module_file: str
    module_sha256: str
    compiler_version: str
    runtime_version: str
    target_backend: str
    runtime_driver: str
    function_name: str
    entry_point: str
    calling_convention_version: int
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    vectorized: bool
    has_preprocess: bool
    has_postprocess: bool
    validation_ok: bool | None
    maximum_absolute_errors: tuple[float, ...] | None
    maximum_relative_errors: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if (
            not self.input_names
            or len(self.input_names) != len(self.input_shapes)
            or len(self.input_names) != len(self.input_dtypes)
            or any(not name for name in self.input_names)
            or len(set(self.input_names)) != len(self.input_names)
        ):
            raise ValueError(
                "IREE manifest input metadata must define one unique name per input."
            )
        output_count = len(self.output_names)
        if (
            output_count == 0
            or output_count != len(self.output_shapes)
            or output_count != len(self.output_dtypes)
            or any(not name for name in self.output_names)
            or len(set(self.output_names)) != output_count
        ):
            raise ValueError(
                "IREE manifest output metadata must define one unique name per output."
            )
        if self.validation_ok is None:
            if (
                self.maximum_absolute_errors is not None
                or self.maximum_relative_errors is not None
            ):
                raise ValueError(
                    "IREE manifest validation errors require validation evidence."
                )
        elif (
            self.maximum_absolute_errors is None
            or self.maximum_relative_errors is None
            or len(self.maximum_absolute_errors) != output_count
            or len(self.maximum_relative_errors) != output_count
        ):
            raise ValueError(
                "IREE manifest validation errors must describe every output."
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "IREEArtifactManifest":
        expected = set(cls.__dataclass_fields__)
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                f"IREE manifest fields are not canonical; missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        if value["format"] != _IREE_ARTIFACT_FORMAT:
            raise ValueError("Artifact is not a Phydrax IREE inference bundle.")
        return cls(
            format=str(value["format"]),
            artifact_id=str(value["artifact_id"]),
            module_file=str(value["module_file"]),
            module_sha256=str(value["module_sha256"]),
            compiler_version=str(value["compiler_version"]),
            runtime_version=str(value["runtime_version"]),
            target_backend=str(value["target_backend"]),
            runtime_driver=str(value["runtime_driver"]),
            function_name=str(value["function_name"]),
            entry_point=str(value["entry_point"]),
            calling_convention_version=int(value["calling_convention_version"]),
            input_names=tuple(str(name) for name in value["input_names"]),
            input_shapes=tuple(tuple(shape) for shape in value["input_shapes"]),
            input_dtypes=tuple(str(dtype) for dtype in value["input_dtypes"]),
            output_names=tuple(str(name) for name in value["output_names"]),
            output_shapes=tuple(tuple(shape) for shape in value["output_shapes"]),
            output_dtypes=tuple(str(dtype) for dtype in value["output_dtypes"]),
            vectorized=bool(value["vectorized"]),
            has_preprocess=bool(value["has_preprocess"]),
            has_postprocess=bool(value["has_postprocess"]),
            validation_ok=(
                None if value["validation_ok"] is None else bool(value["validation_ok"])
            ),
            maximum_absolute_errors=(
                None
                if value["maximum_absolute_errors"] is None
                else tuple(float(error) for error in value["maximum_absolute_errors"])
            ),
            maximum_relative_errors=(
                None
                if value["maximum_relative_errors"] is None
                else tuple(float(error) for error in value["maximum_relative_errors"])
            ),
        )

    def to_dict(self, /) -> dict[str, Any]:
        value = asdict(self)
        value["input_names"] = list(self.input_names)
        value["input_shapes"] = [list(shape) for shape in self.input_shapes]
        value["input_dtypes"] = list(self.input_dtypes)
        value["output_names"] = list(self.output_names)
        value["output_shapes"] = [list(shape) for shape in self.output_shapes]
        value["output_dtypes"] = list(self.output_dtypes)
        if self.maximum_absolute_errors is not None:
            value["maximum_absolute_errors"] = list(self.maximum_absolute_errors)
        if self.maximum_relative_errors is not None:
            value["maximum_relative_errors"] = list(self.maximum_relative_errors)
        return value


@dataclass(frozen=True, slots=True)
class IREEExportResult:
    """Published executable bundle and native-parity evidence."""

    path: Path
    manifest: IREEArtifactManifest
    publication: ResourceSetPublicationReceipt | None = None


class IREEExecutable:
    """Loaded IREE executable with exact positional input and output ABI checks."""

    def __init__(
        self,
        manifest: IREEArtifactManifest,
        module_bytes: bytes,
        /,
    ):
        compiler, runtime = import_iree()
        del compiler
        config = runtime.Config(manifest.runtime_driver)
        context = runtime.SystemContext(config=config)
        module = runtime.VmModule.copy_buffer(context.instance, module_bytes)
        context.add_vm_module(module)
        self.manifest = manifest
        self._context = context
        self._module = module
        self._function = context.modules[module.name][manifest.entry_point]

    def __call__(self, *args: Any) -> np.ndarray | tuple[np.ndarray, ...]:
        if len(args) != len(self.manifest.input_shapes):
            raise ValueError(
                f"IREE executable expected {len(self.manifest.input_shapes)} inputs; got {len(args)}."
            )
        prepared = []
        for index, (argument, shape, dtype) in enumerate(
            zip(
                args,
                self.manifest.input_shapes,
                self.manifest.input_dtypes,
                strict=True,
            )
        ):
            array = np.asarray(argument)
            if tuple(array.shape) != shape:
                raise ValueError(
                    f"IREE input {index} must have shape {shape}; got {array.shape}."
                )
            if array.dtype.str != dtype:
                raise TypeError(
                    f"IREE input {index} must have dtype {dtype}; got {array.dtype.str}."
                )
            prepared.append(array)

        raw_result = self._function(*prepared)
        raw_outputs = (
            tuple(raw_result) if isinstance(raw_result, (tuple, list)) else (raw_result,)
        )
        output_count = len(self.manifest.output_names)
        if len(raw_outputs) != output_count:
            raise RuntimeError(
                f"IREE output arity differs from the artifact manifest: expected "
                f"{output_count}; got {len(raw_outputs)}."
            )
        outputs: list[np.ndarray] = []
        for index, (raw_output, name, shape, dtype) in enumerate(
            zip(
                raw_outputs,
                self.manifest.output_names,
                self.manifest.output_shapes,
                self.manifest.output_dtypes,
                strict=True,
            )
        ):
            output = np.asarray(raw_output.to_host())
            label = f"IREE output {index} ({name!r})"
            if tuple(output.shape) != shape:
                raise RuntimeError(
                    f"{label} shape differs from the artifact manifest: expected {shape}; got {output.shape}."
                )
            if output.dtype.str != dtype:
                raise RuntimeError(
                    f"{label} dtype differs from the artifact manifest: expected {dtype}; got {output.dtype.str}."
                )
            if not np.all(np.isfinite(output)):
                raise RuntimeError(f"{label} contains non-finite values.")
            outputs.append(output)
        return outputs[0] if output_count == 1 else tuple(outputs)


def load_iree(path: str | Path, /) -> IREEExecutable:
    """Verify and load a pickle-free IREE inference artifact."""

    source = Path(path).expanduser().absolute()
    bundle = read_bounded_resource_set(
        source.name,
        trusted_root=source.parent,
        limits=ResourceSetLimits(
            max_total_bytes=4 * 1024 * 1024 * 1024,
            max_member_bytes=4 * 1024 * 1024 * 1024,
            max_members=2,
            max_depth=1,
        ),
    )
    member_names = {member.relative_path for member in bundle.manifest.members}
    if "manifest.json" not in member_names:
        raise ValueError("IREE artifact manifest is missing.")
    manifest_resource = read_bounded_resource(
        "manifest.json",
        trusted_root=source,
        limits=ResourceLimits(16 * 1024 * 1024, 64, 100_000, 100_000, 0),
    )
    value = decode_json_resource(manifest_resource).value
    if not isinstance(value, Mapping):
        raise TypeError("IREE manifest JSON must contain an object.")
    manifest = IREEArtifactManifest.from_dict(value)
    if member_names != {"manifest.json", manifest.module_file}:
        raise ValueError("IREE artifact bundle inventory changed.")
    availability = iree_availability()
    availability.require("compiled-inference")
    versions = dict(availability.versions)
    if (
        versions["iree-base-compiler"] != manifest.compiler_version
        or versions["iree-base-runtime"] != manifest.runtime_version
    ):
        raise ValueError(
            "IREE artifact compiler/runtime versions differ from the runtime."
        )
    module_resource = read_bounded_resource(
        manifest.module_file,
        trusted_root=source,
        limits=ResourceLimits(4 * 1024 * 1024 * 1024, 1, 1, 0, 0),
    )
    if module_resource.manifest.content_sha256 != manifest.module_sha256:
        raise ValueError("IREE module checksum mismatch.")
    return IREEExecutable(manifest, module_resource.data)


def save_iree(
    function: Callable[..., Any] | DomainFunction,
    path: str | Path,
    /,
    *,
    inputs: Sequence[Any],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    policy: IREEExportPolicy | None = None,
    key: Any = None,
    preprocess: Callable[..., Any] | None = None,
    postprocess: Callable[..., Any] | None = None,
    vectorize: bool = False,
    validate: bool = True,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
) -> IREEExportResult:
    """Compile and publish a deterministic array or ordered array-tuple boundary."""

    if key is not None:
        raise ValueError("IREE export requires key=None for deterministic inference.")
    policy_ = IREEExportPolicy() if policy is None else policy
    if not isinstance(policy_, IREEExportPolicy):
        raise TypeError("policy must be IREEExportPolicy or None.")
    input_arrays = tuple(jnp.asarray(value) for value in inputs)
    if not input_arrays:
        raise ValueError("IREE export requires at least one concrete input array.")
    names = (
        tuple(f"input_{index}" for index in range(len(input_arrays)))
        if input_names is None
        else tuple(str(name) for name in input_names)
    )
    if len(names) != len(input_arrays) or any(not name for name in names):
        raise ValueError("input_names must name every input exactly once.")
    if len(set(names)) != len(names):
        raise ValueError("input_names must be unique.")

    export_function = make_inference_export_callable(
        function,
        key=None,
        preprocess=preprocess,
        postprocess=postprocess,
        vectorize=bool(vectorize),
    )
    exported = jax.export.export(jax.jit(export_function))(*input_arrays)
    output_avals = exported.out_avals
    output_count = len(output_avals)
    single_array_tree = jax.tree.structure(0)
    tuple_array_tree = jax.tree.structure((0,) * output_count)
    if output_count == 0 or exported.out_tree not in (
        single_array_tree,
        tuple_array_tree,
    ):
        raise TypeError(
            "IREE export requires one array output or a non-empty tuple of arrays."
        )
    output_names_ = (
        tuple(f"output_{index}" for index in range(output_count))
        if output_names is None
        else tuple(str(name) for name in output_names)
    )
    if len(output_names_) != output_count or any(not name for name in output_names_):
        raise ValueError("output_names must name every output exactly once.")
    if len(set(output_names_)) != len(output_names_):
        raise ValueError("output_names must be unique.")

    compiler_args = [
        "--iree-input-demote-f64-to-f32=false",
        "--iree-input-demote-i64-to-i32=false",
        "--iree-opt-const-expr-hoisting=false",
    ]
    compiler, _ = import_iree()
    module_bytes = bytes(
        compiler.tools.compile_str(
            exported.mlir_module(),
            target_backends=[policy_.target_backend],
            extra_args=tuple(compiler_args),
        )
    )
    module_hash = _sha256_bytes(module_bytes)
    versions = dict(iree_availability().versions)
    compiler_version = versions["iree-base-compiler"]
    runtime_version = versions["iree-base-runtime"]
    metadata = {
        "format": _IREE_ARTIFACT_FORMAT,
        "module_sha256": module_hash,
        "compiler_version": compiler_version,
        "runtime_version": runtime_version,
        "target_backend": policy_.target_backend,
        "runtime_driver": policy_.runtime_driver,
        "function_name": exported.fun_name,
        "calling_convention_version": exported.calling_convention_version,
        "input_names": names,
        "input_shapes": tuple(tuple(value.shape) for value in input_arrays),
        "input_dtypes": tuple(np.dtype(value.dtype).str for value in input_arrays),
        "output_names": output_names_,
        "output_shapes": tuple(tuple(output.shape) for output in output_avals),
        "output_dtypes": tuple(np.dtype(output.dtype).str for output in output_avals),
        "vectorized": bool(vectorize),
        "has_preprocess": preprocess is not None,
        "has_postprocess": postprocess is not None,
    }
    artifact_id = canonical_fingerprint(metadata)
    provisional = IREEArtifactManifest(
        format=_IREE_ARTIFACT_FORMAT,
        artifact_id=artifact_id,
        module_file=f"module-{module_hash[:16]}.vmfb",
        module_sha256=module_hash,
        compiler_version=compiler_version,
        runtime_version=runtime_version,
        target_backend=policy_.target_backend,
        runtime_driver=policy_.runtime_driver,
        function_name=exported.fun_name,
        entry_point="main",
        calling_convention_version=exported.calling_convention_version,
        input_names=names,
        input_shapes=metadata["input_shapes"],
        input_dtypes=metadata["input_dtypes"],
        output_names=metadata["output_names"],
        output_shapes=metadata["output_shapes"],
        output_dtypes=metadata["output_dtypes"],
        vectorized=bool(vectorize),
        has_preprocess=preprocess is not None,
        has_postprocess=postprocess is not None,
        validation_ok=None,
        maximum_absolute_errors=None,
        maximum_relative_errors=None,
    )
    executable = IREEExecutable(provisional, module_bytes)
    validation_ok = None
    maximum_absolute_errors = None
    maximum_relative_errors = None
    if validate:
        native_result = export_function(*input_arrays)
        native_outputs = (
            native_result if isinstance(native_result, tuple) else (native_result,)
        )
        if len(native_outputs) != len(output_names_):
            raise RuntimeError(
                f"Native output arity changed after export: expected {len(output_names_)}; got {len(native_outputs)}."
            )
        deployed_result = executable(*(np.asarray(value) for value in input_arrays))
        deployed_outputs = (
            deployed_result if isinstance(deployed_result, tuple) else (deployed_result,)
        )
        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        for index, (name, native_value, deployed) in enumerate(
            zip(output_names_, native_outputs, deployed_outputs, strict=True)
        ):
            native = np.asarray(native_value)
            expected_shape = provisional.output_shapes[index]
            expected_dtype = provisional.output_dtypes[index]
            label = f"IREE output {index} ({name!r})"
            if tuple(native.shape) != expected_shape:
                raise RuntimeError(
                    f"Native {label.lower()} shape changed after export: expected {expected_shape}; got {native.shape}."
                )
            if native.dtype.str != expected_dtype:
                raise RuntimeError(
                    f"Native {label.lower()} dtype changed after export: expected "
                    f"{expected_dtype}; got {native.dtype.str}."
                )
            if not np.all(np.isfinite(native)):
                raise RuntimeError(f"Native {label.lower()} contains non-finite values.")

            if np.issubdtype(native.dtype, np.inexact):
                absolute = np.abs(native - deployed)
                scale = np.maximum(np.abs(native), np.finfo(native.real.dtype).tiny)
                output_ok = bool(
                    np.allclose(native, deployed, rtol=float(rtol), atol=float(atol))
                )
            else:
                native_numeric = native.astype(np.float64)
                deployed_numeric = deployed.astype(np.float64)
                absolute = np.abs(native_numeric - deployed_numeric)
                scale = np.maximum(np.abs(native_numeric), np.finfo(np.float64).tiny)
                output_ok = bool(np.array_equal(native, deployed))
            max_absolute_error = float(np.max(absolute, initial=0.0))
            max_relative_error = float(np.max(absolute / scale, initial=0.0))
            absolute_errors.append(max_absolute_error)
            relative_errors.append(max_relative_error)
            if not output_ok:
                raise RuntimeError(
                    f"{label} failed native parity: max_abs={max_absolute_error:.3e}, max_rel={max_relative_error:.3e}."
                )
        validation_ok = True
        maximum_absolute_errors = tuple(absolute_errors)
        maximum_relative_errors = tuple(relative_errors)
    manifest = replace(
        provisional,
        validation_ok=validation_ok,
        maximum_absolute_errors=maximum_absolute_errors,
        maximum_relative_errors=maximum_relative_errors,
    )
    destination = Path(path)
    publication = publish_resource_set(
        destination,
        {
            manifest.module_file: module_bytes,
            "manifest.json": (
                json.dumps(
                    manifest.to_dict(),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        },
        limits=ResourceSetLimits(
            max_total_bytes=4 * 1024 * 1024 * 1024,
            max_member_bytes=4 * 1024 * 1024 * 1024,
            max_members=2,
            max_depth=1,
        ),
        mode="atomic_replace",
    )
    return IREEExportResult(destination, manifest, publication)


__all__ = [
    "IREEArtifactManifest",
    "IREEExecutable",
    "IREEExportPolicy",
    "IREEExportResult",
    "load_iree",
    "save_iree",
]
