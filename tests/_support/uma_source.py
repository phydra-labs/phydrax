"""Small UMA source fixtures: plain configurations and torch.save byte layouts.

The configurations follow the pinned fairchem ``bb0a062`` Hydra layout. The
torch.save writer emits the pickle protocol-2 opcode stream and stored ZIP
records that ``torch.save`` writes for an ``MLIPInferenceCheckpoint`` record,
without importing torch, so isolated-extraction tests run anywhere.
"""

from __future__ import annotations

import io
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


DATASETS = ("omat", "omol")


def uma_config(
    *,
    experts: int = 3,
    lmax: int = 2,
    mmax: int = 1,
    max_neighbors: int = 30,
    act_type: str = "gate",
    ff_type: str = "grid",
    norm_type: str = "rms_norm_sh",
    embedding: str = "pos_emb",
    direct: bool = False,
    model_id: str | None = "UMA-S-1.1",
    **backbone: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A tiny conservative (or direct) UMA source configuration and its tasks."""
    values: dict[str, Any] = {
        "model": "escnmd_moe_backbone" if experts else "escnmd_backbone",
        "max_num_elements": 10,
        "sphere_channels": 8,
        "lmax": lmax,
        "mmax": mmax,
        "cutoff": 2.5,
        "max_neighbors": max_neighbors,
        "edge_channels": 4,
        "num_distance_basis": 6,
        "direct_forces": direct,
        "regress_forces": True,
        "regress_stress": not direct,
        "num_layers": 2,
        "hidden_channels": 8,
        "norm_type": norm_type,
        "act_type": act_type,
        "ff_type": ff_type,
        "chg_spin_emb_type": embedding,
        "dataset_list": list(DATASETS),
        "charge_balanced_channels": [0],
        "spin_balanced_channels": [1],
        "otf_graph": True,
        "activation_checkpointing": True,
    }
    if experts:
        values.update(
            num_experts=experts,
            use_composition_embedding=True,
            moe_type="so2",
            moe_dropout=0.05,
        )
    values.update(backbone)
    references = {
        "omat": [0.0, -3.3, 0.0, 0.0, 0.0, 0.0, -8.3, -8.4, -5.0, 0.0],
        "omol": [0.0, -13.6, 0.0, 0.0, 0.0, 0.0, -1029.8, -1484.3, -2041.8, 0.0],
    }
    rmsd = {"omat": 1.3, "omol": 0.9}

    def task(
        name: str, prop: str, datasets: Sequence[str], scale: float, refs: Any
    ) -> dict:
        entry: dict[str, Any] = {
            "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
            "name": name,
            "level": "atom" if prop == "forces" else "system",
            "property": prop,
            "loss_fn": {"_target_": "fairchem.core.modules.loss.DDPMTLoss"},
            "out_spec": {"dim": [1], "dtype": "float32"},
            "normalizer": {
                "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
                "mean": 0.0,
                "rmsd": scale,
            },
            "datasets": list(datasets),
            "metrics": ["mae"],
        }
        if refs is not None:
            entry["element_references"] = {
                "_target_": (
                    "fairchem.core.modules.normalization.element_references."
                    "ElementReferences"
                ),
                "element_references": {
                    "_target_": "torch.DoubleTensor",
                    "_args_": [refs],
                },
            }
        return entry

    if direct:
        heads = {
            f"{name}_energy": {
                "module": "fairchem.core.models.uma.escn_md.MLP_Energy_Head"
            }
            for name in DATASETS
        }
        heads["forces"] = {"module": "fairchem.core.models.uma.escn_md.Linear_Force_Head"}
        tasks = [
            task(f"{name}_energy", "energy", [name], rmsd[name], references[name])
            for name in DATASETS
        ]
        tasks.append(task("forces", "forces", list(DATASETS), 1.1, None))
        pass_through = False
    else:
        heads = {
            "efs": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "dataset_names": list(DATASETS),
                "wrap_property": True,
            }
        }
        tasks = []
        for name in DATASETS:
            tasks.append(
                task(f"{name}_energy", "energy", [name], rmsd[name], references[name])
            )
            tasks.append(task(f"{name}_forces", "forces", [name], rmsd[name], None))
        tasks.append(task("omat_stress", "stress", ["omat"], rmsd["omat"], None))
        pass_through = True
    model_config: dict[str, Any] = {
        "_target_": "fairchem.core.models.base.HydraModel",
        "backbone": values,
        "heads": heads,
        "pass_through_head_outputs": pass_through,
        "otf_graph": True,
        "supports_single_atoms": False,
    }
    if model_id is not None:
        model_config["model_id"] = model_id
    return model_config, tasks


@dataclass(frozen=True)
class SavedTensor:
    """A tensor view as torch.save stores it: storage key, offset, shape, stride."""

    key: str
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage: str
    numel: int


_STORAGE_TYPES = {
    np.dtype("<f4"): "FloatStorage",
    np.dtype("<f8"): "DoubleStorage",
    np.dtype("<i8"): "LongStorage",
}


class _Pickle:
    """Protocol-2 opcode writer for the admitted torch.save subset."""

    def __init__(self) -> None:
        self.parts = [b"\x80\x02"]

    def raw(self, data: bytes) -> None:
        self.parts.append(data)

    def global_(self, module: str, name: str) -> None:
        self.raw(b"c" + module.encode() + b"\n" + name.encode() + b"\n")

    def value(self, value: Any) -> None:
        if isinstance(value, SavedTensor):
            self.tensor(value)
        elif value is None:
            self.raw(b"N")
        elif value is True:
            self.raw(b"\x88")
        elif value is False:
            self.raw(b"\x89")
        elif isinstance(value, int):
            if -(2**31) <= value < 2**31:
                self.raw(b"J" + struct.pack("<i", value))
            else:
                data = value.to_bytes(16, "little", signed=True)
                self.raw(b"\x8a" + bytes([len(data)]) + data)
        elif isinstance(value, float):
            self.raw(b"G" + struct.pack(">d", value))
        elif isinstance(value, str):
            data = value.encode("utf-8")
            self.raw(b"X" + struct.pack("<I", len(data)) + data)
        elif isinstance(value, tuple):
            self.raw(b"(")
            for item in value:
                self.value(item)
            self.raw(b"t")
        elif isinstance(value, list):
            self.raw(b"]")
            if value:
                self.raw(b"(")
                for item in value:
                    self.value(item)
                self.raw(b"e")
        elif isinstance(value, dict):
            self.raw(b"}")
            if value:
                self.raw(b"(")
                for key, item in value.items():
                    self.value(key)
                    self.value(item)
                self.raw(b"u")
        else:
            raise TypeError(type(value))

    def ordered(self, items: Mapping[str, Any]) -> None:
        self.global_("collections", "OrderedDict")
        self.raw(b")R")
        if items:
            self.raw(b"(")
            for key, item in items.items():
                self.value(key)
                self.value(item)
            self.raw(b"u")

    def tensor(self, tensor: SavedTensor) -> None:
        self.global_("torch._utils", "_rebuild_tensor_v2")
        self.raw(b"(")
        self.raw(b"(")
        self.value("storage")
        self.global_("torch", tensor.storage)
        self.value(tensor.key)
        self.value("cpu")
        self.value(tensor.numel)
        self.raw(b"tQ")
        self.value(tensor.offset)
        self.value(tensor.shape)
        self.value(tensor.stride)
        self.value(False)
        self.ordered({})
        self.raw(b"tR")


def torch_save_bytes(
    *,
    model_config: Mapping[str, Any],
    tasks_config: Sequence[Mapping[str, Any]],
    ema: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    n_averaged: int = 7,
    stride_override: Mapping[str, tuple[int, ...]] | None = None,
    record_global: tuple[str, str] = (
        "fairchem.core.units.mlip_unit.api.inference",
        "MLIPInferenceCheckpoint",
    ),
    compress: bool = False,
) -> bytes:
    """torch.save bytes of an ``MLIPInferenceCheckpoint`` with the given arrays."""
    storages: dict[str, bytes] = {}

    def saved(array: np.ndarray, name: str) -> SavedTensor:
        # ``np.ascontiguousarray`` promotes 0-d to 1-d; torch saves the EMA
        # ``n_averaged`` buffer as a 0-d tensor (shape ``()``, stride ``()``).
        value = np.array(array, order="C", copy=True)
        key = str(len(storages))
        storages[key] = value.tobytes()
        strides = tuple(int(s // value.itemsize) for s in value.strides)
        if stride_override and name in stride_override:
            strides = stride_override[name]
        return SavedTensor(
            key,
            0,
            tuple(value.shape),
            strides,
            _STORAGE_TYPES[value.dtype],
            int(value.size),
        )

    ema_tensors = {name: saved(value, name) for name, value in ema.items()}
    ema_tensors["n_averaged"] = saved(
        np.asarray(n_averaged, dtype=np.int64), "n_averaged"
    )
    model_tensors = {name: saved(value, f"model:{name}") for name, value in model.items()}
    writer = _Pickle()
    writer.global_(*record_global)
    writer.raw(b")\x81")
    writer.raw(b"}(")
    writer.value("model_config")
    writer.value(dict(model_config))
    writer.value("model_state_dict")
    writer.ordered(model_tensors)
    writer.value("ema_state_dict")
    writer.ordered(ema_tensors)
    writer.value("tasks_config")
    writer.value([dict(task) for task in tasks_config])
    writer.raw(b"ub.")
    payload = b"".join(writer.parts)
    buffer = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", compression=method) as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/byteorder", b"little")
        for key, data in storages.items():
            archive.writestr(f"archive/data/{key}", data)
        archive.writestr("archive/version", b"3\n")
    return buffer.getvalue()


__all__ = ["DATASETS", "SavedTensor", "torch_save_bytes", "uma_config"]
