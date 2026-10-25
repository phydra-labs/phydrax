#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-side AIMNet2 program executed by a caller-pinned interpreter.

This file is never executed inside Phydrax. ``_aimnet2_checkpoint`` stages its
exact bytes into a private directory and runs it through
``run_pinned_command`` with a fixed isolated bootstrap (``python -I -S -c``)
that adds only the declared provider ``site-packages``::

    worker.py <site-packages> request.json result.zip

Operations:

* ``extract-jit``: a legacy TorchScript ``.jpt`` is executable. The host only
  requests it under an explicit digest-bound trust capability; the staged
  bytes are re-hashed here, the ZIP directory is bounded, and
  ``torch.jit.load`` then executes with the caller's privileges. The state
  tensors and the attributes the source ``load_v1_model`` reads (``cutoff``,
  module names, ``lrcoulomb.rc`` and DFT-D3 ``s6/s8/a1/a2``) are returned.
* ``evaluate``: an isolated source oracle. The installed ``aimnet`` release is
  verified file by file against its RECORD digests; the model is rebuilt from
  the staged ``model_yaml`` (``aimnet.`` classes only) and the host's
  pickle-free state archive with the source key rules; each configuration
  runs in the source's dense single-molecule mode.

Torch is verified by release. The process is neither a security nor a memory
sandbox: the host bounds wall time, logs and output bytes, and the limits in
the request are enforced here before results are written. The result is the
canonical Phydrax array archive (stored ZIP of NPY members plus
``manifest.json``), streamed one member at a time.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import math
import os
import stat
import struct
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


_REFUSED_EXIT = 3
_RESULT_FORMAT = "phydrax-aimnet2-provider-result"
_ZIP_END = struct.Struct("<4s4H2LH")
_NPY_CHUNK_BYTES = 16_777_216
_DFTD3_ATTRIBUTES = ("s8", "a1", "a2", "s6")


class ProviderRefusal(ValueError):
    """A source or request outside the admitted provider contract."""


def _refuse(message: str, /) -> ProviderRefusal:
    return ProviderRefusal(message)


@dataclass(frozen=True, slots=True)
class _Limits:
    max_source_bytes: int
    max_result_bytes: int
    max_tensor_bytes: int
    max_tensor_elements: int
    max_members: int
    max_manifest_bytes: int
    max_array_rank: int
    max_total_elements: int


def _limits(request: Mapping[str, Any], /) -> _Limits:
    declared = request.get("limits")
    names = [field.name for field in dataclasses.fields(_Limits)]
    if (
        not isinstance(declared, Mapping)
        or set(declared) != set(names)
        or any(type(declared[name]) is not int or declared[name] <= 0 for name in names)
    ):
        raise _refuse("Provider requests must declare explicit positive limits.")
    return _Limits(**{name: declared[name] for name in names})


def _normalized(name: str, /) -> str:
    return name.lower().replace("_", "-")


def _distribution(name: str, site_packages: str, /) -> metadata.Distribution:
    matches = [
        item
        for item in metadata.distributions(path=[site_packages])
        if _normalized(item.metadata["Name"]) == _normalized(name)
    ]
    if len(matches) != 1:
        raise _refuse(f"Provider distribution {name!r} is not uniquely installed.")
    return matches[0]


def _record_digest(distribution: metadata.Distribution, /) -> str:
    record = distribution.read_text("RECORD")
    if record is None:
        raise _refuse("Provider distribution has no RECORD manifest.")
    return hashlib.sha256(record.encode("utf-8")).hexdigest()


def _verify_record_files(distribution: metadata.Distribution, /) -> int:
    files = distribution.files
    if files is None:
        raise _refuse("Provider distribution has no file inventory.")
    verified = 0
    for package_path in files:
        recorded = package_path.hash
        if recorded is None:
            continue
        if recorded.mode != "sha256":
            raise _refuse("Provider RECORD uses an unsupported digest.")
        location = Path(str(distribution.locate_file(package_path)))
        with location.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise _refuse(f"Provider file {package_path} is not a regular file.")
            if package_path.size is not None and before.st_size != package_path.size:
                raise _refuse(f"Provider file {package_path} differs from its RECORD.")
            digest = hashlib.file_digest(stream, "sha256").digest()
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise _refuse(f"Provider file {package_path} changed while hashed.")
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if encoded != recorded.value:
            raise _refuse(f"Provider file {package_path} differs from its RECORD.")
        verified += 1
    return verified


def _identity(
    site_packages: str, required: Mapping[str, Any], verify: Sequence[str], /
) -> dict[str, Any]:
    if not isinstance(required, Mapping) or "torch" not in required:
        raise _refuse("The provider request must declare its torch release.")
    identity: dict[str, Any] = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "site_packages": site_packages,
    }
    for name in ("torch", *verify):
        if name not in required:
            raise _refuse(f"The provider request must declare the {name} release.")
        distribution = _distribution(name, site_packages)
        if distribution.version != required[name]:
            raise _refuse(
                f"Provider {name} {distribution.version} differs from {required[name]}."
            )
        identity[name] = {
            "version": distribution.version,
            "record_sha256": _record_digest(distribution),
            "verified_files": _verify_record_files(distribution) if name in verify else 0,
        }
    return identity


def _bounded_zip(data: bytes, limits: _Limits, label: str, /) -> zipfile.ZipFile:
    """Bound a ZIP directory and its expanded records before any is read."""
    end = len(data) - _ZIP_END.size
    if end < 0 or b"PK\x05\x06" not in data[max(0, len(data) - 65_557) :]:
        raise _refuse(f"The {label} is not a ZIP container.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise _refuse(f"The {label} ZIP directory is malformed.") from error
    members = archive.infolist()
    if len(members) > limits.max_members:
        archive.close()
        raise _refuse(f"The {label} has more records than the member limit.")
    expanded = 0
    for member in members:
        if member.flag_bits & 0x1 or member.file_size > max(
            limits.max_tensor_bytes, limits.max_manifest_bytes
        ):
            archive.close()
            raise _refuse(f"The {label} record {member.filename!r} is not admitted.")
        expanded += member.file_size
    if expanded > limits.max_source_bytes:
        archive.close()
        raise _refuse(f"The {label} expands beyond the source byte limit.")
    return archive


def _staged_source(request: Mapping[str, Any], root: Path, limits: _Limits, /) -> bytes:
    source = request.get("source")
    if not isinstance(source, Mapping):
        raise _refuse("The request does not describe its source.")
    data = (root / "source.bin").read_bytes()
    if len(data) != source.get("byte_size") or hashlib.sha256(
        data
    ).hexdigest() != source.get("sha256"):
        raise _refuse("Staged source bytes differ from the admitted digest.")
    if len(data) > limits.max_source_bytes:
        raise _refuse("The source exceeds the source byte limit.")
    return data


def _jit_attribute(module: Any, name: str, /) -> float | None:
    """``models.utils.get_jit_attr`` without its default substitution."""
    value = None
    try:
        value = getattr(module, name, None)
    except Exception:
        value = None
    if value is None:
        try:
            value = module.__getattr__(name)
        except (AttributeError, RuntimeError):
            value = None
    if value is None:
        return None
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise _refuse(f"JIT attribute {name!r} is not a scalar.")
        return float(value.item())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise _refuse(f"JIT attribute {name!r} has an unadmitted type.")


def _children(module: Any, /) -> Iterator[tuple[str, Any]]:
    for name, child in module.named_children():
        yield name, child
        yield from _children(child)


def _run_extract_jit(
    request: Mapping[str, Any], root: Path, limits: _Limits, /
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    source = request["source"]
    if source.get("trusted_executable") is not True:
        raise _refuse("TorchScript extraction requires the explicit trust capability.")
    data = _staged_source(request, root, limits)
    _bounded_zip(data, limits, "TorchScript source").close()
    # Executable deserialization of the admitted digest, explicitly trusted by
    # the caller; never reached by a refused or failed safe load.
    model = torch.jit.load(io.BytesIO(data), map_location="cpu")
    cutoff = _jit_attribute(model, "cutoff")
    if cutoff is None or not math.isfinite(cutoff):
        raise _refuse("The TorchScript model declares no finite cutoff.")
    children = list(_children(model))
    names = [name for name, _ in children]
    attributes: dict[str, Any] = {"cutoff": cutoff, "children": names}
    for name, child in children:
        if name == "lrcoulomb":
            attributes["lrcoulomb_rc"] = _jit_attribute(child, "rc")
        if name in ("dftd3", "d3bj") and "d3_params" not in attributes:
            values = {key: _jit_attribute(child, key) for key in _DFTD3_ATTRIBUTES}
            attributes["d3_params"] = {
                "s8": values["s8"] if values["s8"] is not None else 0.0,
                "a1": values["a1"] if values["a1"] is not None else 0.0,
                "a2": values["a2"] if values["a2"] is not None else 0.0,
                "s6": values["s6"] if values["s6"] is not None else 1.0,
            }
    arrays = {f"state/{name}": value for name, value in model.state_dict().items()}
    return {"attributes": attributes}, arrays


def _read_state_archive(root: Path, limits: _Limits, /) -> dict[str, np.ndarray]:
    data = (root / "state.zip").read_bytes()
    if len(data) > limits.max_source_bytes:
        raise _refuse("The staged state exceeds the source byte limit.")
    with _bounded_zip(data, limits, "state archive") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        inventory = manifest.get("arrays")
        if not isinstance(inventory, Mapping):
            raise _refuse("The state archive has no array inventory.")
        arrays = {}
        for name, entry in inventory.items():
            payload = archive.read(entry["member"])
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise _refuse(f"State array {name!r} fails its digest.")
            value = np.load(io.BytesIO(payload), allow_pickle=False)
            if not name.startswith("state/"):
                raise _refuse(f"State archive member {name!r} is not a state tensor.")
            arrays[name.removeprefix("state/")] = value
    return arrays


def _check_classes(value: Any, /) -> None:
    if isinstance(value, Mapping):
        if "class" in value and not str(value["class"]).startswith("aimnet."):
            raise _refuse(f"Untrusted class {value['class']!r} in model_yaml.")
        for item in value.values():
            _check_classes(item)
    elif isinstance(value, list):
        for item in value:
            _check_classes(item)


def _run_evaluate(
    request: Mapping[str, Any], root: Path, limits: _Limits, /
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    import yaml
    from aimnet.config import build_module
    from aimnet.models.utils import validate_state_dict_keys

    dtype_name = request.get("evaluation_dtype")
    if dtype_name not in ("float32", "float64"):
        raise _refuse("evaluation_dtype must be float32 or float64.")
    dtype = getattr(torch, dtype_name)
    model_yaml = request.get("model_yaml")
    if not isinstance(model_yaml, str) or len(model_yaml) > limits.max_manifest_bytes:
        raise _refuse("model_yaml must be bounded text.")
    declaration = yaml.safe_load(model_yaml)
    _check_classes(declaration)
    state = _read_state_archive(root, limits)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = build_module(declaration)
        if hasattr(model, "outputs") and hasattr(model.outputs, "atomic_shift"):
            model.outputs.atomic_shift.double()
        tensors = {
            name: torch.from_numpy(np.array(value)) for name, value in state.items()
        }
        result = model.load_state_dict(tensors, strict=False)
        missing, unexpected = validate_state_dict_keys(
            result.missing_keys, result.unexpected_keys
        )
        if missing or unexpected:
            raise _refuse(f"State keys differ from model_yaml: {missing} {unexpected}.")
        model = model.to(dtype)
        if hasattr(model, "outputs") and hasattr(model.outputs, "atomic_shift"):
            model.outputs.atomic_shift.double()
            shift = "outputs.atomic_shift.shifts.weight"
            if shift in tensors:
                model.outputs.atomic_shift.shifts.weight.data.copy_(
                    tensors[shift].double()
                )
        model.eval()
        configurations = request.get("configurations")
        if not isinstance(configurations, list) or not configurations:
            raise _refuse("Oracle evaluation needs configurations.")
        energies = []
        arrays: dict[str, Any] = {}
        for index, case in enumerate(configurations):
            numbers = torch.tensor([case["numbers"]], dtype=torch.int64)
            if numbers.numel() > limits.max_tensor_elements:
                raise _refuse("An oracle configuration exceeds the element limit.")
            coord = torch.tensor([case["positions"]], dtype=dtype, requires_grad=True)
            data = {
                "coord": coord,
                "numbers": numbers,
                "charge": torch.tensor([case["charge"]], dtype=dtype),
            }
            if case.get("multiplicity") is not None:
                data["mult"] = torch.tensor([case["multiplicity"]], dtype=dtype)
            output = model(data)
            energy = output["energy"].sum()
            (gradient,) = torch.autograd.grad(energy, coord)
            energies.append(float(output["energy"].detach().double().sum()))
            arrays[f"forces/{index}"] = (-gradient[0]).detach().double().numpy()
            arrays[f"charges/{index}"] = output["charges"][0].detach().double().numpy()
            if "spin_charges" in output:
                arrays[f"spin_charges/{index}"] = (
                    output["spin_charges"][0].detach().double().numpy()
                )
        arrays["energy"] = np.asarray(energies, dtype=np.float64)
    finally:
        torch.set_default_dtype(previous)
    return {"evaluation_dtype": dtype_name}, arrays


def _array_view(value: Any, /) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        if value.dtype in (torch.bfloat16, torch.float16, torch.complex32):
            raise _refuse("Reduced-precision provider tensors are refused.")
        return value.detach().cpu().resolve_conj().resolve_neg().numpy()
    return np.asarray(value)


def _npy_header(dtype: np.dtype[Any], shape: tuple[int, ...], /) -> bytes:
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": np.lib.format.dtype_to_descr(dtype),
            "fortran_order": False,
            "shape": shape,
        },
    )
    return header.getvalue()


def _stored(name: str, size: int, /) -> zipfile.ZipInfo:
    information = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    information.compress_type = zipfile.ZIP_STORED
    information.create_system = 3
    information.external_attr = 0o600 << 16
    information.file_size = size
    return information


def _write_archive(
    path: Path, manifest: Mapping[str, Any], arrays: Mapping[str, Any], limits: _Limits, /
) -> None:
    """Admit every array from its metadata, then stream the stored archive."""
    views = {
        name: np.ascontiguousarray(_array_view(value)) for name, value in arrays.items()
    }
    if len(views) + 1 > limits.max_members:
        raise _refuse("The provider result exceeds the member limit.")
    planned = []
    total_elements = 0
    total_bytes = 0
    for index, name in enumerate(sorted(views)):
        view = views[name]
        if view.dtype.hasobject or view.dtype.kind not in "biuf":
            raise _refuse(f"Provider array {name!r} has an unsupported dtype.")
        if view.ndim > limits.max_array_rank or view.size > limits.max_tensor_elements:
            raise _refuse(f"Provider array {name!r} exceeds the tensor limits.")
        header = _npy_header(view.dtype, view.shape)
        size = len(header) + view.nbytes
        if size > limits.max_tensor_bytes:
            raise _refuse(f"Provider array {name!r} exceeds the tensor byte limit.")
        total_elements += view.size
        total_bytes += size
        planned.append((name, f"arrays/{index:06d}.npy", header, size))
    if (
        total_elements > limits.max_total_elements
        or total_bytes > limits.max_result_bytes
    ):
        raise _refuse("The provider result exceeds its aggregate limits.")
    inventory = {}
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name, member, header, size in planned:
            view = views[name]
            digest = hashlib.sha256()
            with archive.open(_stored(member, size), mode="w") as stream:
                for chunk in _chunks(header, view):
                    digest.update(chunk)
                    stream.write(chunk)
            inventory[name] = {
                "member": member,
                "shape": list(view.shape),
                "dtype": view.dtype.str,
                "sha256": digest.hexdigest(),
                "order": "C",
            }
        payload = json.dumps(
            {**manifest, "arrays": inventory}, allow_nan=False, indent=2, sort_keys=True
        ).encode("utf-8")
        if len(payload) > limits.max_manifest_bytes:
            raise _refuse("The provider result manifest exceeds its byte limit.")
        archive.writestr(_stored("manifest.json", len(payload)), payload)


def _chunks(header: bytes, array: np.ndarray, /) -> Iterator[bytes]:
    yield header
    if array.ndim == 0:
        yield array.tobytes("C")
        return
    slice_bytes = array.itemsize * math.prod(array.shape[1:])
    step = max(_NPY_CHUNK_BYTES // max(slice_bytes, 1), 1)
    for start in range(0, array.shape[0], step):
        yield array[start : start + step].tobytes("C")


def _execute(site_packages: str, request_path: Path, output: Path, /) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise _refuse("Provider request must be a JSON object.")
    limits = _limits(request)
    operation = request.get("operation")
    verify = ("aimnet",) if operation == "evaluate" else ()
    provider = _identity(site_packages, request.get("provider"), verify)
    manifest: dict[str, Any] = {
        "format": _RESULT_FORMAT,
        "operation": operation,
        "provider": provider,
    }
    match operation:
        case "extract-jit":
            record, arrays = _run_extract_jit(request, request_path.parent, limits)
        case "evaluate":
            record, arrays = _run_evaluate(request, request_path.parent, limits)
        case other:
            raise _refuse(f"Unsupported provider operation {other!r}.")
    manifest.update(record)
    _write_archive(output, manifest, arrays, limits)


def main(arguments: Sequence[str], /) -> int:
    if len(arguments) != 3:
        sys.stderr.write("usage: worker.py <site-packages> request.json result.zip\n")
        return 2
    site_packages, request_path, output = arguments
    try:
        _execute(site_packages, Path(request_path), Path(output))
    except ProviderRefusal as refusal:
        sys.stderr.write(f"REFUSED: {refusal}\n")
        return _REFUSED_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
