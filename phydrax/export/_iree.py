#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.domain import DomainFunction

from .._fingerprint import canonical_fingerprint
from ..backends.iree import import_iree, iree_availability
from ._inference import make_inference_export_callable


_IREE_ARTIFACT_FORMAT = "phydrax-iree-inference"


def _sha256_bytes(value: bytes, /) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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
    output_shape: tuple[int, ...]
    output_dtype: str
    vectorized: bool
    has_preprocess: bool
    has_postprocess: bool
    validation_ok: bool | None
    maximum_absolute_error: float | None
    maximum_relative_error: float | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "IREEArtifactManifest":
        expected = set(cls.__dataclass_fields__)
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "IREE manifest fields are not canonical; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
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
            input_shapes=tuple(
                tuple(int(size) for size in shape) for shape in value["input_shapes"]
            ),
            input_dtypes=tuple(str(dtype) for dtype in value["input_dtypes"]),
            output_shape=tuple(int(size) for size in value["output_shape"]),
            output_dtype=str(value["output_dtype"]),
            vectorized=bool(value["vectorized"]),
            has_preprocess=bool(value["has_preprocess"]),
            has_postprocess=bool(value["has_postprocess"]),
            validation_ok=(
                None if value["validation_ok"] is None else bool(value["validation_ok"])
            ),
            maximum_absolute_error=(
                None
                if value["maximum_absolute_error"] is None
                else float(value["maximum_absolute_error"])
            ),
            maximum_relative_error=(
                None
                if value["maximum_relative_error"] is None
                else float(value["maximum_relative_error"])
            ),
        )

    def to_dict(self, /) -> dict[str, Any]:
        value = asdict(self)
        value["input_names"] = list(self.input_names)
        value["input_shapes"] = [list(shape) for shape in self.input_shapes]
        value["input_dtypes"] = list(self.input_dtypes)
        value["output_shape"] = list(self.output_shape)
        return value


@dataclass(frozen=True, slots=True)
class IREEExportResult:
    """Published executable bundle and native-parity evidence."""

    path: Path
    manifest: IREEArtifactManifest


class IREEExecutable:
    """Loaded IREE executable with exact positional shape and dtype validation."""

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

    def __call__(self, *args: Any) -> np.ndarray:
        if len(args) != len(self.manifest.input_shapes):
            raise ValueError(
                f"IREE executable expected {len(self.manifest.input_shapes)} inputs; "
                f"got {len(args)}."
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
        result = np.asarray(self._function(*prepared).to_host())
        if tuple(result.shape) != self.manifest.output_shape:
            raise RuntimeError("IREE output shape differs from the artifact manifest.")
        if result.dtype.str != self.manifest.output_dtype:
            raise RuntimeError("IREE output dtype differs from the artifact manifest.")
        return result


def load_iree(path: str | Path, /) -> IREEExecutable:
    """Verify and load a pickle-free IREE inference artifact."""

    source = Path(path)
    value = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("IREE manifest JSON must contain an object.")
    manifest = IREEArtifactManifest.from_dict(value)
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
    module_path = source / manifest.module_file
    if _sha256_file(module_path) != manifest.module_sha256:
        raise ValueError("IREE module checksum mismatch.")
    return IREEExecutable(manifest, module_path.read_bytes())


def save_iree(
    function: Callable[..., Any] | DomainFunction,
    path: str | Path,
    /,
    *,
    inputs: Sequence[Any],
    input_names: Sequence[str] | None = None,
    policy: IREEExportPolicy | None = None,
    key: Any = None,
    preprocess: Callable[..., Any] | None = None,
    postprocess: Callable[..., Any] | None = None,
    vectorize: bool = False,
    validate: bool = True,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
) -> IREEExportResult:
    """Compile and atomically publish one deterministic array-valued inference boundary."""

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
    if len(exported.out_avals) != 1:
        raise ValueError("IREE export currently requires one array output.")
    output_aval = exported.out_avals[0]
    compiler, _ = import_iree()
    module_bytes = bytes(
        compiler.tools.compile_str(
            exported.mlir_module(),
            target_backends=[policy_.target_backend],
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
        "input_shapes": tuple(
            tuple(int(size) for size in value.shape) for value in input_arrays
        ),
        "input_dtypes": tuple(np.dtype(value.dtype).str for value in input_arrays),
        "output_shape": tuple(int(size) for size in output_aval.shape),
        "output_dtype": np.dtype(output_aval.dtype).str,
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
        output_shape=metadata["output_shape"],
        output_dtype=metadata["output_dtype"],
        vectorized=bool(vectorize),
        has_preprocess=preprocess is not None,
        has_postprocess=postprocess is not None,
        validation_ok=None,
        maximum_absolute_error=None,
        maximum_relative_error=None,
    )
    executable = IREEExecutable(provisional, module_bytes)
    validation_ok = None
    maximum_absolute_error = None
    maximum_relative_error = None
    if validate:
        native = np.asarray(export_function(*input_arrays))
        deployed = executable(*(np.asarray(value) for value in input_arrays))
        absolute = np.abs(native - deployed)
        scale = np.maximum(np.abs(native), np.finfo(native.real.dtype).tiny)
        maximum_absolute_error = float(np.max(absolute, initial=0.0))
        maximum_relative_error = float(np.max(absolute / scale, initial=0.0))
        validation_ok = bool(
            np.allclose(native, deployed, rtol=float(rtol), atol=float(atol))
        )
        if not validation_ok:
            raise RuntimeError(
                "IREE output failed native parity: "
                f"max_abs={maximum_absolute_error:.3e}, "
                f"max_rel={maximum_relative_error:.3e}."
            )
    manifest = IREEArtifactManifest(
        **{
            **provisional.to_dict(),
            "input_names": provisional.input_names,
            "input_shapes": provisional.input_shapes,
            "input_dtypes": provisional.input_dtypes,
            "output_shape": provisional.output_shape,
            "validation_ok": validation_ok,
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
        }
    )
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    module_temporary = destination / "module.tmp.vmfb"
    module_temporary.write_bytes(module_bytes)
    os.replace(module_temporary, destination / manifest.module_file)
    manifest_temporary = destination / "manifest.tmp.json"
    manifest_temporary.write_text(
        json.dumps(manifest.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temporary, destination / "manifest.json")
    for candidate in destination.glob("module-*.vmfb"):
        if candidate.name != manifest.module_file:
            candidate.unlink()
    return IREEExportResult(destination, manifest)


__all__ = [
    "IREEArtifactManifest",
    "IREEExecutable",
    "IREEExportPolicy",
    "IREEExportResult",
    "load_iree",
    "save_iree",
]
