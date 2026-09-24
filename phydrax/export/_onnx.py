#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from phydrax.domain import DomainFunction

from .._external_resource import read_bounded_resource, ResourceLimits
from .._external_runtime import (
    _host_only,
    _require_optional,
    ExternalTensorSpec,
    ExternalTransport,
    HostInferenceAdapter,
)
from .._identity import (
    ArtifactBindingIdentity,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
)
from .._publication import PublicationReceipt, publish_bytes
from ._inference import make_inference_export_callable


# ONNX tensor element types admitted by the host loader, mapped to exact dtypes.
_ONNX_TENSOR_DTYPES = {
    "tensor(bool)": np.dtype(np.bool_),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
}
_ONNX_PROVIDER = "CPUExecutionProvider"


@dataclass(frozen=True)
class OnnxExportResult:
    """Result metadata returned by `save_onnx`."""

    path: Path
    publication: PublicationReceipt | None = None
    validation_ok: bool | None = None
    validation_message: str | None = None


def save_onnx(
    fn: Callable[..., Any] | DomainFunction,
    path: str | Path,
    /,
    *,
    inputs: Sequence[Any],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    model_name: str | None = None,
    key: Any = None,
    preprocess: Callable[..., Any] | None = None,
    postprocess: Callable[..., Any] | None = None,
    vectorize: bool = False,
    enable_double_precision: bool = True,
    validate: bool = False,
    validation_inputs: Sequence[Any] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    record_primitive_calls_file: str | Path | None = None,
) -> OnnxExportResult:
    """Export an array-callable inference function to ONNX.

    This exports a single learned inference boundary, not a full solver, loss, or
    constraint graph. Passing `key=None` keeps deterministic exports free of
    evaluation-time PRNG operations.
    """
    if validate and validation_inputs is None:
        raise ValueError("validation_inputs must be provided when validate=True.")

    try:
        jax2onnx = import_module("jax2onnx")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ONNX export requires `phydrax[onnx-export]`."
        ) from error

    out_path = Path(path)
    export_fn = make_inference_export_callable(
        fn,
        key=key,
        preprocess=preprocess,
        postprocess=postprocess,
        vectorize=bool(vectorize),
    )

    with TemporaryDirectory(prefix="phydrax-onnx-") as temporary_directory:
        staged_path = Path(temporary_directory) / out_path.name
        kwargs: dict[str, Any] = {
            "inputs": list(inputs),
            "return_mode": "file",
            "output_path": staged_path,
            "enable_double_precision": bool(enable_double_precision),
        }
        if input_names is not None:
            kwargs["input_names"] = list(input_names)
        if output_names is not None:
            kwargs["output_names"] = list(output_names)
        if model_name is not None:
            kwargs["model_name"] = str(model_name)
        if record_primitive_calls_file is not None:
            primitive_path = Path(record_primitive_calls_file)
            primitive_path.parent.mkdir(parents=True, exist_ok=True)
            kwargs["record_primitive_calls_file"] = str(primitive_path)

        exported = jax2onnx.to_onnx(export_fn, **kwargs)
        exported_path = staged_path if exported is None else Path(exported)
        validation_ok: bool | None = None
        validation_message: str | None = None
        if validate:
            assert validation_inputs is not None
            ok, message = jax2onnx.allclose(
                export_fn,
                str(exported_path),
                inputs=list(validation_inputs),
                rtol=float(rtol),
                atol=float(atol),
                enable_double_precision=bool(enable_double_precision),
            )
            if not ok:
                raise RuntimeError(message)
            validation_ok = bool(ok)
            validation_message = str(message)
        module_bytes = exported_path.read_bytes()
    publication = publish_bytes(
        out_path,
        module_bytes,
        maximum_bytes=4 * 1024 * 1024 * 1024,
        mode="atomic_replace",
    )
    return OnnxExportResult(
        path=out_path,
        validation_ok=validation_ok,
        validation_message=validation_message,
        publication=publication,
    )


def _onnx_schema(
    arguments: Sequence[Any], role: str, /
) -> tuple[ExternalTensorSpec, ...]:
    specs = []
    for argument in arguments:
        shape = tuple(argument.shape)
        if any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
            raise ValueError(
                f"ONNX {role} {argument.name!r} has symbolic dimensions {shape}; "
                "host inference requires a fixed shape."
            )
        if argument.type not in _ONNX_TENSOR_DTYPES:
            raise ValueError(
                f"ONNX {role} {argument.name!r} has unsupported type {argument.type!r}."
            )
        specs.append(
            ExternalTensorSpec(argument.name, shape, _ONNX_TENSOR_DTYPES[argument.type])
        )
    return tuple(specs)


def _onnx_binding(
    model_sha256: str,
    inputs: tuple[ExternalTensorSpec, ...],
    outputs: tuple[ExternalTensorSpec, ...],
    runtime_version: str,
    /,
) -> ArtifactBindingIdentity:
    semantic = SemanticProvenance(
        {
            "kind": "onnx-inference-model",
            "inputs": [[spec.name, list(spec.shape), spec.dtype] for spec in inputs],
            "outputs": [[spec.name, list(spec.shape), spec.dtype] for spec in outputs],
        },
        resource_ids={"model": model_sha256},
    )
    names = tuple(f"input:{spec.name}" for spec in inputs) + tuple(
        f"output:{spec.name}" for spec in outputs
    )
    specs = inputs + outputs
    return ArtifactBindingIdentity(
        semantic,
        NumericRevision(semantic, {"model_sha256": model_sha256}),
        ExecutableSignature(
            shapes=zip(names, (spec.shape for spec in specs), strict=True),
            dtypes=zip(names, (spec.dtype for spec in specs), strict=True),
            backend_facts={
                "runtime": "onnxruntime",
                "runtime_version": runtime_version,
                "execution_provider": _ONNX_PROVIDER,
            },
        ),
    )


def load_onnx(
    path: str | Path,
    /,
    *,
    trusted_model_sha256: str,
    transport: ExternalTransport = "copy",
) -> HostInferenceAdapter:
    """Load an ONNX model for eager host inference through ONNX Runtime.

    Requires `phydrax[onnx-inference]`. The model bytes must match the
    caller-supplied SHA-256 pin. Inputs and outputs must have fixed shapes and
    supported tensor types; they become the adapter's schemas. The session runs
    on ONNX Runtime's CPU execution provider. The returned `HostInferenceAdapter`
    is host-only, has no adjoint, and is bound to the model digest, its schemas,
    and the ONNX Runtime version.
    """
    _host_only()
    if (
        not isinstance(trusted_model_sha256, str)
        or len(trusted_model_sha256) != 64
        or any(character not in "0123456789abcdef" for character in trusted_model_sha256)
    ):
        raise ValueError("trusted_model_sha256 must be a lowercase SHA-256 digest.")
    _require_optional("onnxruntime", "phydrax[onnx-inference]")
    source = Path(path).expanduser().absolute()
    resource = read_bounded_resource(
        source.name,
        trusted_root=source.parent,
        limits=ResourceLimits(4 * 1024 * 1024 * 1024, 1, 1, 1, 0),
    )
    model_sha256 = resource.manifest.content_sha256
    if model_sha256 != trusted_model_sha256:
        raise PermissionError(
            "ONNX model is not authorized by the caller-supplied trusted pin."
        )
    onnxruntime = import_module("onnxruntime")
    session = onnxruntime.InferenceSession(resource.data, providers=[_ONNX_PROVIDER])
    inputs = _onnx_schema(session.get_inputs(), "input")
    outputs = _onnx_schema(session.get_outputs(), "output")
    input_names = tuple(spec.name for spec in inputs)
    output_names = [spec.name for spec in outputs]

    def run(values: tuple[np.ndarray, ...]) -> list[np.ndarray]:
        return session.run(output_names, dict(zip(input_names, values, strict=True)))

    return HostInferenceAdapter(
        run,
        inputs,
        outputs,
        _onnx_binding(model_sha256, inputs, outputs, onnxruntime.__version__),
        transport,
    )


__all__ = ["OnnxExportResult", "load_onnx", "save_onnx"]
