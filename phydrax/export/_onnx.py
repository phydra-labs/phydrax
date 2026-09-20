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

from phydrax.domain import DomainFunction

from .._publication import PublicationReceipt, publish_bytes
from ._inference import make_inference_export_callable


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


__all__ = ["OnnxExportResult", "save_onnx"]
