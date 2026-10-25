#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact native reconstruction of fairchem UMA/eSCN-MD sources (``bb0a062``).

`parse_uma_source_config` interprets the plain source ``model_config`` and
``tasks_config`` declaratively (no Hydra instantiation): every key is admitted
into the architecture, mapped to a named scientific policy, recorded as an
explicit non-numerical exclusion with its reason, or refused. Source defaults
are the pinned constructor defaults.

`uma_model_from_source` builds the actual model kind (`UMAPotential` or
`UMAPropertyModel`) from the source parameter
inventory with exact names, shapes and finiteness; every tensor is consumed
once or the reconstruction refuses. Non-persistent source buffers (Wigner
kernels, grids, coefficient maps, Gaussian offsets) are not parameters; they
are rebuilt natively from their definitions, and the persistent e3nn ``Jd_l``
buffers are consumed as basis evidence (`uma_source_basis_residual`).

`uma_initial_source_tensors` draws a source-layout parameter inventory with
the source initialization rules, so a natively initialized model is built by
the same factory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ...atomistic._types import AtomisticPrecisionPolicy, AtomisticScaleContract
from ...sparse._streamed import StreamedRelationPlan
from ...typing import parse, PRNGKey
from ._uma import (
    UMAArchitecture,
    UMADirectStressHead,
    UMAEnergyHead,
    UMAEnergyHeadKind,
    UMAEnergyReduction,
    UMAModel,
    UMAModelKind,
    UMANetwork,
    UMAPotential,
    UMAPropertyModel,
    UMATaskReferences,
)
from ._uma_conditioning import (
    UMAChargeSpinEmbedding,
    UMADatasetEmbedding,
    UMAFrozen,
    UMAMoLERouter,
    UMASystemEmbedding,
)
from ._uma_layers import (
    UMABlock,
    UMAEdgeDegreeEmbedding,
    UMAEdgewise,
    UMAGridAtomwise,
    UMALayerNorm,
    UMALinear,
    UMALinearMap,
    UMAMixtureLinear,
    UMANormalization,
    UMARadialMLP,
    UMASharedMixture,
    UMASO2Convolution,
    UMASO3Linear,
    UMASpectralAtomwise,
)
from ._uma_so3 import (
    uma_source_basis_residual,
    uma_source_wigner,
    UMAEdgeRollGauge,
    UMASphericalGrid,
)


UMA_SOURCE_COMMIT = "bb0a062a9a08d8b3ad3995aca5b9765e07804990"
# Stored e3nn J matrices are float32; the relabelled native basis must agree.
UMA_BASIS_TOLERANCE = 1.0e-5

_BACKBONE_MODELS = {
    "escnmd_backbone": False,
    "fairchem.core.models.uma.escn_md.eSCNMDBackbone": False,
    "escnmd_moe_backbone": True,
    "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone": True,
}
_HYDRA_MODELS = frozenset({"hydra", "fairchem.core.models.base.HydraModel"})
# Pinned eSCNMDBackbone.__init__ defaults.
_BACKBONE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "max_num_elements": 100,
        "sphere_channels": 128,
        "lmax": 2,
        "mmax": 2,
        "grid_resolution": None,
        "num_sphere_samples": 128,
        "otf_graph": False,
        "max_neighbors": 300,
        "use_pbc": True,
        "use_pbc_single": True,
        "cutoff": 5.0,
        "edge_channels": 128,
        "distance_function": "gaussian",
        "num_distance_basis": 512,
        "direct_forces": True,
        "regress_forces": True,
        "direct_stress": False,
        "regress_stress": False,
        "regress_hessian": False,
        "hessian_vmap": True,
        "num_layers": 2,
        "hidden_channels": 128,
        "norm_type": "rms_norm_sh",
        "act_type": "gate",
        "ff_type": "grid",
        "activation_checkpointing": False,
        "chg_spin_emb_type": "pos_emb",
        "cs_emb_grad": False,
        "dataset_emb_grad": False,
        "dataset_list": None,
        "dataset_mapping": None,
        "use_dataset_embedding": True,
        "use_cuda_graph_wigner": False,
        "use_quaternion_wigner": True,
        "radius_pbc_version": 2,
        "always_use_pbc": True,
        "charge_balanced_channels": None,
        "spin_balanced_channels": None,
        "edge_chunk_size": 1,
        "execution_mode": "general",
    }
)
# Pinned eSCNMDMoeBackbone.__init__ defaults (``act`` must stay torch SiLU).
_MOE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {
        "num_experts": 8,
        "moe_dropout": 0.0,
        "use_global_embedding": False,
        "use_composition_embedding": False,
        "composition_dropout": 0.0,
        "moe_expert_coefficient_norm": "softmax",
        "layers_moe": None,
        "moe_layer_type": "pytorch",
        "moe_single": False,
        "moe_type": "so2",
        "model_version": 1.0,
    }
)
# Source settings that do not change the scientific function, with the reason.
_EXCLUDED_BACKBONE: Mapping[str, str] = MappingProxyType(
    {
        "num_sphere_samples": "unused by the pinned source",
        "otf_graph": "source graph generation; the native graph binds the declared selection",
        "use_pbc": "deprecated source flag, unused",
        "use_pbc_single": "deprecated source flag, unused",
        "always_use_pbc": "source graph generation periodicity override",
        "activation_checkpointing": "source memory execution policy",
        "edge_chunk_size": "source execution chunking",
        "execution_mode": "source execution backend with identical formulas",
        "use_cuda_graph_wigner": "source Wigner kernel implementation",
        "use_quaternion_wigner": (
            "source Wigner implementation and random roll; native uses the exact "
            "native O(3) recurrence with the declared zero-roll gauge"
        ),
        "regress_hessian": "source autograd Hessian request; native responses differentiate energy",
        "hessian_vmap": "source autograd execution option",
        "radius_pbc_version": "admitted only as 2; mapped to the declared selection policy",
        "moe_dropout": "training-only dropout; inference identity",
        "composition_dropout": "training-only dropout; inference identity",
        "moe_layer_type": "source MoLE kernel (pytorch/dgl) with identical formulas",
        "model_version": "source generation tag, recorded as provenance",
        "dataset_list": "deprecated alias resolved into dataset_mapping",
    }
)
_TASK_TRAINING_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        "loss_fn": "training objective",
        "out_spec": "source output buffer declaration",
        "metrics": "training metrics",
        "train_on_free_atoms": "training mask policy",
        "eval_on_free_atoms": "evaluation mask policy",
        "inference_only": "admitted false; untrained derivative tasks are native responses",
    }
)
_TENSOR_TARGETS = frozenset(
    {"torch.DoubleTensor", "torch.FloatTensor", "torch.Tensor", "torch.tensor"}
)
_NORMALIZER_TARGETS = frozenset(
    {
        "fairchem.core.modules.normalization.normalizer.Normalizer",
        "fairchem.core.modules.normalization.normalizer.create_normalizer",
    }
)
_REFERENCE_TARGETS = frozenset(
    {
        "fairchem.core.modules.normalization.element_references.ElementReferences",
        "fairchem.core.modules.normalization.element_references.create_element_references",
    }
)
_TASK_TARGET = "fairchem.core.units.mlip_unit.mlip_unit.Task"
_HEAD_CLASSES = frozenset(
    {
        "MLP_EFS_Head",
        "MLP_Energy_Head",
        "Linear_Energy_Head",
        "Linear_Force_Head",
        "MLP_Stress_Head",
        "DatasetSpecificSingleHeadWrapper",
        "DatasetSpecificMoEWrapper",
    }
)

UMAHeadRole: TypeAlias = Literal["energy", "direct-forces", "direct-stress"]
UMAHeadKind: TypeAlias = Literal["mlp", "linear", "force", "stress"]
UMAHeadWrapping: TypeAlias = Literal["none", "single", "experts"]


class UMASourceConfigurationError(ValueError):
    """A UMA source configuration outside the admitted pinned source contract."""


def _refuse(message: str, /) -> UMASourceConfigurationError:
    return UMASourceConfigurationError(message)


@dataclass(frozen=True, slots=True)
class UMASourceTask:
    """One source ``Task``: property, datasets, normalizer and element references."""

    name: str
    property: Literal["energy", "forces", "stress"]
    datasets: tuple[str, ...]
    mean: float
    rmsd: float
    element_references: np.ndarray | None


@dataclass(frozen=True, slots=True)
class UMAHeadDeclaration:
    """One source output head and its native role."""

    name: str
    role: UMAHeadRole
    kind: UMAHeadKind
    wrapping: UMAHeadWrapping
    datasets: tuple[str, ...]
    experts: int
    expert_of_dataset: Mapping[str, int]
    reduce: Literal["sum", "mean"]
    outputs: Mapping[str, str]

    @property
    def tensor_prefix(self) -> str:
        return f"output_heads.{self.name}." + ("" if self.wrapping == "none" else "head.")


@dataclass(frozen=True, slots=True)
class UMASourceDeclaration:
    """The admitted source declaration: architecture, heads, tasks and exclusions."""

    architecture: UMAArchitecture
    model_id: str | None
    heads: tuple[UMAHeadDeclaration, ...]
    energy_heads: tuple[str, ...]
    tasks: tuple[UMASourceTask, ...]
    excluded: Mapping[str, str]
    source_dataset_keys: tuple[str, ...]
    record: Mapping[str, Any] = field(repr=False)


def _class_name(value: Any, /, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise _refuse(f"{context} must name a source class.")
    return value.rsplit(".", 1)[-1]


def _plain(value: Any, /) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": value.dtype.str}
    return value


def _array_value(
    value: Any, arrays: Mapping[str, np.ndarray], name: str, /
) -> np.ndarray:
    """A configuration tensor: ``{"_target_": torch.*Tensor, "_args_": [list]}`` or array."""
    if isinstance(value, Mapping) and set(value) == {"__array__"}:
        reference = value["__array__"]
        if reference not in arrays:
            raise _refuse(f"{name} references an undelivered configuration array.")
        return np.asarray(arrays[reference], dtype=np.float64)
    if isinstance(value, Mapping):
        if (
            set(value) != {"_target_", "_args_"}
            or value["_target_"] not in _TENSOR_TARGETS
        ):
            raise _refuse(f"{name} is not a plain tensor construction.")
        arguments = value["_args_"]
        if not isinstance(arguments, list) or len(arguments) != 1:
            raise _refuse(f"{name} must construct from exactly one list.")
        value = arguments[0]
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise _refuse(f"{name} is not numeric.") from error
    if not np.all(np.isfinite(array)):
        raise _refuse(f"{name} must be finite.")
    return array


def _scalar(value: Any, arrays: Mapping[str, np.ndarray], name: str, /) -> float:
    array = (
        _array_value(value, arrays, name)
        if not isinstance(value, (int, float))
        else (np.asarray(float(value)))
    )
    if array.size != 1 or isinstance(value, bool):
        raise _refuse(f"{name} must be one finite number.")
    result = float(array.reshape(()))
    if not math.isfinite(result):
        raise _refuse(f"{name} must be finite.")
    return result


def _parse_task(
    entry: Any, arrays: Mapping[str, np.ndarray], excluded: dict[str, str], index: int, /
) -> UMASourceTask:
    if not isinstance(entry, Mapping):
        raise _refuse(f"tasks_config[{index}] must be a mapping.")
    allowed = {
        "_target_",
        "name",
        "level",
        "property",
        "normalizer",
        "element_references",
        "datasets",
        *_TASK_TRAINING_FIELDS,
    }
    unknown = set(entry) - allowed
    if unknown:
        raise _refuse(f"Task {index} has unadmitted fields {sorted(unknown)!r}.")
    if entry.get("_target_") != _TASK_TARGET:
        raise _refuse(f"Task {index} must instantiate {_TASK_TARGET}.")
    name = entry.get("name")
    prop = entry.get("property")
    if not isinstance(name, str) or prop not in ("energy", "forces", "stress"):
        raise _refuse(f"Task {index} must name an energy, forces or stress property.")
    level = entry.get("level")
    if level != ("atom" if prop == "forces" else "system"):
        raise _refuse(f"Task {name!r} has a level inconsistent with {prop!r}.")
    if entry.get("inference_only", False):
        raise _refuse(f"Task {name!r} is inference-only; native derivatives replace it.")
    datasets = entry.get("datasets")
    if (
        not isinstance(datasets, list)
        or not datasets
        or not all(isinstance(item, str) for item in datasets)
    ):
        raise _refuse(f"Task {name!r} must list its datasets.")
    for key, reason in _TASK_TRAINING_FIELDS.items():
        if key in entry:
            excluded[f"tasks_config.{name}.{key}"] = reason
    normalizer = entry.get("normalizer")
    if normalizer is None:
        mean, rmsd = 0.0, 1.0
    else:
        if not isinstance(normalizer, Mapping) or normalizer.get("_target_") not in (
            _NORMALIZER_TARGETS
        ):
            raise _refuse(f"Task {name!r} normalizer is not an admitted Normalizer.")
        fields = set(normalizer) - {"_target_"}
        if fields - {"mean", "rmsd", "stdev"} or "file" in normalizer:
            raise _refuse(f"Task {name!r} normalizer must give mean and rmsd values.")
        scale_key = "rmsd" if "rmsd" in normalizer else "stdev"
        if scale_key not in normalizer or "mean" not in normalizer:
            raise _refuse(f"Task {name!r} normalizer must give mean and rmsd.")
        mean = _scalar(normalizer["mean"], arrays, f"{name}.normalizer.mean")
        rmsd = _scalar(normalizer[scale_key], arrays, f"{name}.normalizer.rmsd")
        if rmsd == 0.0:
            raise _refuse(f"Task {name!r} normalizer rmsd must be nonzero.")
    references = entry.get("element_references")
    reference_array = None
    if references is not None:
        if (
            not isinstance(references, Mapping)
            or references.get("_target_") not in _REFERENCE_TARGETS
            or set(references) != {"_target_", "element_references"}
        ):
            raise _refuse(f"Task {name!r} element references are not ElementReferences.")
        reference_array = _array_value(
            references["element_references"], arrays, f"{name}.element_references"
        )
        if reference_array.ndim != 1:
            raise _refuse(f"Task {name!r} element references must be one vector.")
        if prop != "energy":
            raise _refuse(f"Task {name!r}: only energy tasks carry element references.")
    return UMASourceTask(name, prop, tuple(datasets), mean, rmsd, reference_array)


def _resolve_mapping(names: Any, mapping: Any, parameter: str, /) -> dict[str, str]:
    """Source ``resolve_dataset_mapping``."""
    if names is not None and mapping is not None:
        raise _refuse(f"Both {parameter!r} and 'dataset_mapping' are declared.")
    if names is None and mapping is None:
        raise _refuse("'dataset_mapping' must be declared.")
    if names is not None:
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise _refuse(f"{parameter!r} must list dataset names.")
        mapping = {name: name for name in names}
    if not isinstance(mapping, Mapping) or not mapping:
        raise _refuse("'dataset_mapping' must be a nonempty mapping.")
    resolved = {str(key): str(value) for key, value in mapping.items()}
    if not set(resolved.values()) <= set(resolved):
        raise _refuse("dataset_mapping values must be a subset of its keys.")
    return resolved


def _parse_head(
    name: str, config: Any, regress: Mapping[str, bool], /
) -> tuple[UMAHeadDeclaration, dict[str, str]]:
    if not isinstance(config, Mapping):
        raise _refuse(f"Head {name!r} must be a mapping.")
    excluded: dict[str, str] = {}
    module = _class_name(config.get("module"), context=f"head {name!r} module")
    if module not in _HEAD_CLASSES:
        raise _refuse(f"Head {name!r} module {module!r} is not an admitted UMA head.")
    wrapping: UMAHeadWrapping = "none"
    datasets: tuple[str, ...] = ()
    expert_of: dict[str, int] = {}
    experts = 0
    inner_config: Mapping[str, Any] = config
    inner = module
    if module in ("DatasetSpecificSingleHeadWrapper", "DatasetSpecificMoEWrapper"):
        allowed = {"module", "head_cls", "wrap_property", "head_kwargs", "dataset_names"}
        if module == "DatasetSpecificMoEWrapper":
            allowed.add("dataset_mapping")
        unknown = set(config) - allowed
        if unknown:
            raise _refuse(f"Head {name!r} has unadmitted fields {sorted(unknown)!r}.")
        inner = _class_name(config.get("head_cls"), context=f"head {name!r} head_cls")
        inner_config = config.get("head_kwargs") or {}
        if not isinstance(inner_config, Mapping):
            raise _refuse(f"Head {name!r} head_kwargs must be a mapping.")
        if module == "DatasetSpecificSingleHeadWrapper":
            names = config.get("dataset_names")
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise _refuse(f"Head {name!r} must list its dataset names.")
            datasets = tuple(sorted(names))
            wrapping = "single"
        else:
            mapping = _resolve_mapping(
                config.get("dataset_names"),
                config.get("dataset_mapping"),
                "dataset_names",
            )
            datasets = tuple(sorted(mapping))
            targets = sorted(set(mapping.values()))
            expert_of = {key: targets.index(mapping[key]) for key in datasets}
            experts = len(datasets)
            wrapping = "experts"
        wrap_property = bool(config.get("wrap_property", True))
    else:
        wrap_property = None
    match inner:
        case "MLP_EFS_Head" | "MLP_Energy_Head":
            allowed_inner = {"reduce", "prefix", "wrap_property"}
            kind: UMAHeadKind = "mlp"
            role: UMAHeadRole = "energy"
        case "Linear_Energy_Head":
            allowed_inner = {"reduce"}
            kind, role = "linear", "energy"
        case "Linear_Force_Head":
            allowed_inner = set()
            kind, role = "force", "direct-forces"
        case "MLP_Stress_Head":
            allowed_inner = {"reduce"}
            kind, role = "stress", "direct-stress"
        case _:
            raise _refuse(f"Head {name!r} wraps an unadmitted head {inner!r}.")
    fields = set(inner_config) - ({"module"} if wrapping == "none" else set())
    if fields - allowed_inner:
        raise _refuse(
            f"Head {name!r} has unadmitted fields {sorted(fields - allowed_inner)!r}."
        )
    try:
        reduce = parse(
            inner_config.get("reduce", "mean" if kind == "stress" else "sum"),
            UMAEnergyReduction,
            f"Head {name!r} reduce",
        )
    except (TypeError, ValueError) as error:
        raise _refuse(f"Head {name!r} reduce must be sum or mean.") from error
    # Linear_Energy_Head with reduce="mean" only reports sum / n; with direct
    # forces it is a property-only source configuration (UMAPropertyModel).
    prefix = inner_config.get("prefix")
    if prefix is not None and not isinstance(prefix, str):
        raise _refuse(f"Head {name!r} prefix must be a string.")
    outputs: dict[str, str] = {}
    if role == "energy":
        outputs["energy"] = f"{prefix}_energy" if prefix else "energy"
        if inner == "MLP_EFS_Head" and not regress["direct_forces"]:
            if regress["regress_stress"] and not regress["direct_stress"]:
                outputs["forces"] = f"{prefix}_forces" if prefix else "forces"
                outputs["stress"] = f"{prefix}_stress" if prefix else "stress"
            elif regress["regress_forces"]:
                outputs["forces"] = f"{prefix}_forces" if prefix else "forces"
    elif role == "direct-forces":
        outputs["forces"] = "forces"
    else:
        outputs["stress"] = "stress"
    if wrap_property is not None:
        excluded[f"heads.{name}.wrap_property"] = "source output dictionary nesting"
    if "wrap_property" in inner_config:
        excluded[f"heads.{name}.wrap_property"] = "source output dictionary nesting"
    return (
        UMAHeadDeclaration(
            name=name,
            role=role,
            kind=kind,
            wrapping=wrapping,
            datasets=datasets,
            experts=experts,
            expert_of_dataset=MappingProxyType(expert_of),
            reduce=reduce,
            outputs=MappingProxyType(outputs),
        ),
        excluded,
    )


def _output_keys(
    heads: Sequence[UMAHeadDeclaration], pass_through: bool, /
) -> dict[str, tuple[str, str, str | None]]:
    """Source output key -> (head, property, dataset or None)."""
    keys: dict[str, tuple[str, str, str | None]] = {}

    def add(key: str, value: tuple[str, str, str | None], /) -> None:
        if key in keys:
            raise _refuse(f"Source output key {key!r} is produced twice.")
        keys[key] = value

    for head in heads:
        for prop, inner in head.outputs.items():
            if pass_through and head.wrapping != "none":
                for dataset in head.datasets:
                    add(f"{dataset}_{inner}", (head.name, prop, dataset))
            elif pass_through:
                add(inner, (head.name, prop, None))
            elif inner == prop:
                add(f"{head.name}:{prop}", (head.name, prop, None))
    return keys


_MODEL_CONFIG_FIELDS = frozenset(
    {
        "_target_",
        "model",
        "backbone",
        "heads",
        "otf_graph",
        "pass_through_head_outputs",
        "freeze_backbone",
        "supports_single_atoms",
        "model_id",
        "finetune_config",
    }
)


def _admit_model_config(model_config: Any, excluded: dict[str, str], /) -> None:
    """The HydraModel envelope: admitted fields, target and non-numerical exclusions."""
    if not isinstance(model_config, Mapping):
        raise _refuse("model_config must be a mapping.")
    unknown = set(model_config) - _MODEL_CONFIG_FIELDS
    if unknown:
        raise _refuse(f"model_config has unadmitted fields {sorted(unknown)!r}.")
    target = model_config.get("_target_", model_config.get("model"))
    if target not in _HYDRA_MODELS:
        raise _refuse("model_config must declare the fairchem HydraModel.")
    if model_config.get("finetune_config") is not None:
        raise _refuse("Fine-tuning configurations load other checkpoints; refused.")
    for key in ("otf_graph", "freeze_backbone"):
        if key in model_config:
            excluded[f"model_config.{key}"] = (
                "source graph generation"
                if key == "otf_graph"
                else "training parameter freeze"
            )


def _admit_backbone(
    model_config: Mapping[str, Any], excluded: dict[str, str], /
) -> tuple[bool, dict[str, Any]]:
    """Backbone class and pinned constructor values: ``(moe, values)``."""
    backbone = model_config.get("backbone")
    if not isinstance(backbone, Mapping):
        raise _refuse("model_config.backbone must be a mapping.")
    model_name = backbone.get("model")
    if model_name not in _BACKBONE_MODELS:
        raise _refuse(f"Backbone {model_name!r} is not an admitted UMA/eSCN-MD backbone.")
    moe = _BACKBONE_MODELS[model_name]
    defaults = {**_BACKBONE_DEFAULTS, **(_MOE_DEFAULTS if moe else {})}
    unknown = set(backbone) - set(defaults) - {"model"}
    if unknown:
        raise _refuse(f"Backbone has unadmitted fields {sorted(unknown)!r}.")
    values = {key: backbone.get(key, default) for key, default in defaults.items()}
    for key, reason in _EXCLUDED_BACKBONE.items():
        if key in values:
            excluded[f"backbone.{key}"] = reason
    if values["distance_function"] != "gaussian":
        raise _refuse("Only the gaussian distance function exists in the source.")
    if values["radius_pbc_version"] != 2:
        raise _refuse("Only radius_pbc_version 2 graph selection is admitted.")
    return moe, values


def _admit_model_identity(
    model_config: Mapping[str, Any], values: Mapping[str, Any], moe: bool, /
) -> tuple[str | None, int, Literal["mean", "sum-over-count-plus-one"]]:
    """Model id and MoLE admission: ``(model_id, experts, composition_policy)``."""
    model_id = model_config.get("model_id")
    if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
        raise _refuse("model_id must be a nonempty string when present.")
    experts = int(values["num_experts"]) if moe else 0
    composition_policy: Literal["mean", "sum-over-count-plus-one"] = "mean"
    if experts:
        if values["use_global_embedding"]:
            raise _refuse(
                "The obsolete global embedding is not part of the pinned source."
            )
        # moe_single: the source caches MOLE layers by (in, out, bias); only
        # bias-free layers of equal shape coincide (biased keys hold their own
        # bias tensor), so those share one parameter. Reconstructed exactly.
        if values["moe_type"] in ("all", "notso2"):
            # The source shares one per-system *edge* count (set_MOLE_sizes) with
            # every MOLE layer, so node-row MOLE layers cannot execute there.
            raise _refuse(
                f"moe_type {values['moe_type']!r} applies edge-count MoLE segments to "
                "node rows in the pinned source; it has no executable source meaning."
            )
        if values["moe_type"] not in ("so2", "so2m0"):
            raise _refuse("Unknown moe_type.")
        if values["moe_expert_coefficient_norm"] not in ("softmax", "pnorm"):
            raise _refuse("Unknown MoLE coefficient normalization.")
        if model_id is None:
            if values["model_version"] in (1.1, "1.1"):
                model_id = "UMA-1.1"
            else:
                raise _refuse(
                    "UMA MoE checkpoints without model_id or model_version 1.1 are "
                    "refused by the pinned source."
                )
        composition_policy = (
            "sum-over-count-plus-one" if model_id == "UMA-S-1.2" else "mean"
        )
    return model_id, experts, composition_policy


def _parse_heads(
    heads_config: Any, regress: Mapping[str, bool], excluded: dict[str, str], /
) -> list[UMAHeadDeclaration]:
    """Every declared head in name order, with its exclusions recorded."""
    if not isinstance(heads_config, Mapping) or not heads_config:
        raise _refuse("model_config.heads must declare the source heads.")
    heads: list[UMAHeadDeclaration] = []
    for head_name in sorted(heads_config):
        head, head_excluded = _parse_head(head_name, heads_config[head_name], regress)
        heads.append(head)
        excluded.update(head_excluded)
    return heads


_TaskRoutes: TypeAlias = dict[str, tuple[UMASourceTask, str]]


def _route_tasks(
    tasks: Sequence[UMASourceTask],
    keys: Mapping[str, tuple[str, str, str | None]],
    dataset_names: Sequence[str],
    by_name: Mapping[str, UMAHeadDeclaration],
    pass_through: bool,
    /,
) -> tuple[_TaskRoutes, _TaskRoutes, _TaskRoutes]:
    """Bind each task to the head output it reads; one task per dataset and property.

    Returns the energy, forces and stress routes ``dataset -> (task, head)``.
    """
    energy_tasks: _TaskRoutes = {}
    force_tasks: _TaskRoutes = {}
    stress_tasks: _TaskRoutes = {}
    for task in tasks:
        key = task.name if pass_through else f"{task.name}:{task.property}"
        if key not in keys:
            raise _refuse(
                f"Task {task.name!r} reads an output no admitted head produces."
            )
        head_name, prop, dataset = keys[key]
        if prop != task.property:
            raise _refuse(f"Task {task.name!r} reads {prop!r} as {task.property!r}.")
        for name in task.datasets:
            if name not in dataset_names:
                raise _refuse(f"Task {task.name!r} names unknown dataset {name!r}.")
            if dataset is not None and name != dataset:
                raise _refuse(f"Task {task.name!r} reads another dataset's output.")
            if (
                by_name[head_name].wrapping != "none"
                and name not in by_name[head_name].datasets
            ):
                raise _refuse(f"Head {head_name!r} does not serve dataset {name!r}.")
            table = {
                "energy": energy_tasks,
                "forces": force_tasks,
                "stress": stress_tasks,
            }[task.property]
            if name in table:
                raise _refuse(f"Dataset {name!r} has two {task.property} tasks.")
            table[name] = (task, head_name)
    return energy_tasks, force_tasks, stress_tasks


def _served_heads(
    heads: Sequence[UMAHeadDeclaration],
    by_name: Mapping[str, UMAHeadDeclaration],
    energy_tasks: _TaskRoutes,
    force_tasks: _TaskRoutes,
    stress_tasks: _TaskRoutes,
    /,
) -> tuple[tuple[str, ...], set[str], set[str]]:
    """Energy, direct-force and direct-stress heads; every head must serve a task.

    Gradient tasks read from an energy head must be its exact derivative.
    """
    energy_head_names = tuple(sorted({head for _, head in energy_tasks.values()}))
    for gradient_tasks in (force_tasks, stress_tasks):
        for dataset, (task, head_name) in gradient_tasks.items():
            if by_name[head_name].role == "energy":
                _require_conservative(dataset, task, energy_tasks, head_name)
    direct_force_heads = {
        head for _, head in force_tasks.values() if by_name[head].role != "energy"
    }
    direct_stress_heads = {
        head for _, head in stress_tasks.values() if by_name[head].role != "energy"
    }
    if len(direct_force_heads) > 1 or len(direct_stress_heads) > 1:
        raise _refuse("At most one direct force and one direct stress head are admitted.")
    used = set(energy_head_names) | direct_force_heads | direct_stress_heads
    for head in heads:
        if head.name not in used:
            raise _refuse(
                f"Head {head.name!r} serves no admitted task; its parameters would be "
                "silently unused."
            )
    return energy_head_names, direct_force_heads, direct_stress_heads


@dataclass(frozen=True, slots=True)
class _UMAEnergyReadout:
    """The admitted energy readout: shared head kind/reduction and task maps."""

    kind: str
    reduction: str
    model_kind: UMAModelKind
    head_of_task: tuple[int, ...]
    head_experts: tuple[int, ...]
    stress_head: UMAHeadDeclaration | None


def _energy_readout(
    by_name: Mapping[str, UMAHeadDeclaration],
    energy_head_names: tuple[str, ...],
    energy_tasks: _TaskRoutes,
    force_tasks: _TaskRoutes,
    dataset_names: Sequence[str],
    direct_stress_heads: set[str],
    /,
) -> _UMAEnergyReadout:
    """Energy heads share one kind and reduction; the model kind follows the source."""
    energy_kinds = {by_name[name].kind for name in energy_head_names}
    energy_reductions = {by_name[name].reduce for name in energy_head_names}
    if len(energy_reductions) > 1:
        raise _refuse("Energy heads must share one reduction.")
    model_kind: UMAModelKind = "potential"
    if energy_reductions == {"mean"}:
        # The source reports sum / n but its forces/stress differentiate the
        # unreduced sum (compute_energy's energy_part). A scalar potential exists
        # exactly when every energy dataset's forces are that gradient; otherwise
        # the source only reports properties (the pinned Toolkit UMAWrapper still
        # exposes them as "energy"/"forces"), realized by UMAPropertyModel.
        for dataset, (_, head_name) in energy_tasks.items():
            forces = force_tasks.get(dataset)
            if forces is None or forces[1] != head_name:
                model_kind = "property"
    energy_wrappings = {by_name[name].wrapping for name in energy_head_names}
    if not energy_head_names:
        raise _refuse(
            "A UMA potential needs an energy head; property-only heads refused."
        )
    if len(energy_kinds) != 1 or (
        "experts" in energy_wrappings and len(energy_head_names) != 1
    ):
        raise _refuse("Energy heads must share one kind; an expert wrapper stands alone.")
    head_of_task = tuple(
        energy_head_names.index(energy_tasks[name][1]) if name in energy_tasks else 0
        for name in dataset_names
    )
    expert_head = by_name[energy_head_names[0]]
    head_experts = (
        tuple(expert_head.expert_of_dataset[name] for name in dataset_names)
        if expert_head.wrapping == "experts"
        else ()
    )
    stress_head = (
        by_name[next(iter(direct_stress_heads))] if direct_stress_heads else None
    )
    if stress_head is not None and stress_head.wrapping == "experts" and not head_experts:
        raise _refuse("An expert stress head needs the expert energy head's task map.")
    return _UMAEnergyReadout(
        kind=next(iter(energy_kinds)),
        reduction=next(iter(energy_reductions)),
        model_kind=model_kind,
        head_of_task=head_of_task,
        head_experts=head_experts,
        stress_head=stress_head,
    )


def parse_uma_source_config(
    model_config: Mapping[str, Any],
    tasks_config: Sequence[Mapping[str, Any]],
    /,
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    roll_gauge: UMAEdgeRollGauge = "two-chart-nlerp-zero-roll",
) -> UMASourceDeclaration:
    """Admit a plain source configuration; refuse anything not exactly understood."""
    arrays_ = {} if arrays is None else dict(arrays)
    excluded: dict[str, str] = {}
    _admit_model_config(model_config, excluded)
    moe, values = _admit_backbone(model_config, excluded)
    model_id, experts, composition_policy = _admit_model_identity(
        model_config, values, moe
    )
    if values["use_dataset_embedding"]:
        mapping = _resolve_mapping(
            values["dataset_list"], values["dataset_mapping"], "dataset_list"
        )
    else:
        mapping = None
    regress = {
        name: bool(values[name])
        for name in ("direct_forces", "regress_forces", "direct_stress", "regress_stress")
    }
    pass_through = bool(model_config.get("pass_through_head_outputs", False))
    heads = _parse_heads(model_config.get("heads"), regress, excluded)
    tasks = tuple(
        _parse_task(entry, arrays_, excluded, index)
        for index, entry in enumerate(tasks_config)
    )
    if len({task.name for task in tasks}) != len(tasks):
        raise _refuse("Task names must be unique.")
    keys = _output_keys(heads, pass_through)
    dataset_names = sorted(
        mapping if mapping is not None else {d for task in tasks for d in task.datasets}
    )
    if not dataset_names:
        raise _refuse("No datasets are declared.")
    by_name = {head.name: head for head in heads}
    energy_tasks, force_tasks, stress_tasks = _route_tasks(
        tasks, keys, dataset_names, by_name, pass_through
    )
    energy_head_names, direct_force_heads, direct_stress_heads = _served_heads(
        heads, by_name, energy_tasks, force_tasks, stress_tasks
    )
    readout = _energy_readout(
        by_name,
        energy_head_names,
        energy_tasks,
        force_tasks,
        dataset_names,
        direct_stress_heads,
    )
    if values["direct_forces"] and not direct_force_heads:
        excluded["backbone.direct_forces"] = (
            "declares direct forces but no direct force task is served"
        )
    architecture = UMAArchitecture(
        max_elements=values["max_num_elements"],
        sphere_channels=values["sphere_channels"],
        lmax=values["lmax"],
        mmax=values["mmax"],
        grid_resolution=values["grid_resolution"],
        cutoff=values["cutoff"],
        max_neighbors=values["max_neighbors"],
        strict_neighbors=False,
        edge_channels=values["edge_channels"],
        distance_basis=values["num_distance_basis"],
        layer_count=values["num_layers"],
        hidden_channels=values["hidden_channels"],
        normalization=values["norm_type"],
        edge_activation=values["act_type"],
        feed_forward=values["ff_type"],
        charge_spin_embedding=values["chg_spin_emb_type"],
        charge_spin_trainable=bool(values["cs_emb_grad"]),
        dataset_trainable=bool(values["dataset_emb_grad"]),
        tasks=dataset_names,
        dataset_mapping=None if mapping is None else tuple(mapping.items()),
        charge_channels=tuple(values["charge_balanced_channels"] or ()),
        spin_channels=tuple(values["spin_balanced_channels"] or ()),
        experts=experts,
        mixture_kind=values["moe_type"] if moe else "so2",
        mixture_layers=values["layers_moe"] if moe else None,
        mixture_sharing=bool(values["moe_single"]) if moe else False,
        composition_embedding=bool(values["use_composition_embedding"]) if moe else False,
        composition_policy=composition_policy,
        coefficient_norm=(
            "pnorm-batch-count"
            if moe and values["moe_expert_coefficient_norm"] == "pnorm"
            else "softmax-plus-0.005"
        ),
        energy_head=parse(readout.kind, UMAEnergyHeadKind, "energy head kind"),
        energy_reduction=parse(readout.reduction, UMAEnergyReduction, "energy reduction"),
        head_wrapper="dataset-experts" if readout.head_experts else "shared",
        head_experts=readout.head_experts,
        energy_head_of_task=readout.head_of_task,
        force_route="direct" if values["direct_forces"] else "conservative",
        direct_force_head=bool(direct_force_heads),
        direct_stress_head=(
            None if readout.stress_head is None else readout.stress_head.reduce
        ),
        isolated_atoms=(
            "network"
            if bool(model_config.get("supports_single_atoms", False))
            else "external-reference-lookup"
        ),
        roll_gauge=roll_gauge,
        model_kind=readout.model_kind,
    )
    record = {
        "source_commit": UMA_SOURCE_COMMIT,
        "model_config": _plain(model_config),
        "tasks_config": _plain(list(tasks_config)),
        "model_id": model_id,
    }
    return UMASourceDeclaration(
        architecture=architecture,
        model_id=model_id,
        heads=tuple(heads),
        energy_heads=energy_head_names,
        tasks=tasks,
        excluded=MappingProxyType(excluded),
        source_dataset_keys=tuple(mapping) if mapping is not None else (),
        record=MappingProxyType(record),
    )


def _require_conservative(
    dataset: str,
    task: UMASourceTask,
    energy_tasks: Mapping[str, tuple[UMASourceTask, str]],
    head: str,
    /,
) -> None:
    """A gradient task equals the native energy derivative only for equal scaling."""
    energy = energy_tasks.get(dataset)
    if energy is None or energy[1] != head:
        raise _refuse(f"Gradient task {task.name!r} has no energy task on head {head!r}.")
    if task.mean != 0.0 or task.rmsd != energy[0].rmsd:
        raise _refuse(
            f"Gradient task {task.name!r} is denormalized differently from its energy; "
            "it is not the derivative of the native energy."
        )


# ---------------------------------------------------------------------------
# Tensor inventories


class _Inventory:
    """Exact source tensors: each consumed once with its exact shape."""

    def __init__(self, tensors: Mapping[str, Any], dtype: np.dtype, /) -> None:
        self._tensors = {str(name): value for name, value in tensors.items()}
        self._consumed: set[str] = set()
        self.dtype = dtype

    def take(self, name: str, shape: tuple[int, ...], init: str, /) -> np.ndarray:
        del init
        if name not in self._tensors:
            raise ValueError(f"Missing UMA source tensor {name!r}.")
        if name in self._consumed:
            raise RuntimeError(f"UMA source tensor {name!r} consumed twice.")
        array = np.asarray(self._tensors[name])
        if array.shape != shape:
            raise ValueError(
                f"UMA source tensor {name!r} has shape {array.shape}, not {shape}."
            )
        if array.dtype.kind != "f" or not np.all(np.isfinite(array)):
            raise ValueError(
                f"UMA source tensor {name!r} must be finite floating values."
            )
        self._consumed.add(name)
        return array.astype(self.dtype)

    def present(self, name: str, /) -> bool:
        return name in self._tensors

    def finish(self) -> None:
        extra = sorted(set(self._tensors) - self._consumed)
        if extra:
            raise ValueError(f"Unadmitted UMA source tensors: {extra}.")


class _InitialInventory:
    """Source-layout tensors drawn with the source initialization rules."""

    def __init__(self, key: PRNGKey, dtype: np.dtype, /) -> None:
        self._key = key
        self.dtype = dtype
        self.tensors: dict[str, np.ndarray] = {}

    def _next(self) -> PRNGKey:
        self._key, key = jr.split(self._key)
        return key

    def take(self, name: str, shape: tuple[int, ...], init: str, /) -> np.ndarray:
        key = self._next()
        match init.split(":"):
            case ["uniform", fan]:
                bound = 1.0 / math.sqrt(float(fan))
                value = jr.uniform(
                    key, shape, minval=-bound, maxval=bound, dtype=jnp.float64
                )
            case ["small"]:
                value = jr.uniform(
                    key, shape, minval=-0.001, maxval=0.001, dtype=jnp.float64
                )
            case ["normal"]:
                value = jr.normal(key, shape, dtype=jnp.float64)
            case ["ones"]:
                value = jnp.ones(shape, dtype=jnp.float64)
            case ["zeros"]:
                value = jnp.zeros(shape, dtype=jnp.float64)
            case ["jd", degree]:
                value = _reference_j(int(degree))
            case _:
                raise ValueError(f"Unknown UMA initialization {init!r}.")
        array = np.asarray(
            jax.device_get(value),
            dtype=np.float32 if init.startswith("jd") else np.float64,
        )
        self.tensors[name] = array
        return array.astype(self.dtype)

    def present(self, name: str, /) -> bool:
        del name
        return False

    def finish(self) -> None:
        return None


# e3nn's J rotation: degree-one block, a proper involution exchanging x and y.
_REFERENCE_J_ROTATION = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)


def _reference_j(degree: int, /) -> np.ndarray:
    full = np.asarray(uma_source_wigner(jnp.asarray(_REFERENCE_J_ROTATION), degree))
    start = degree * degree
    return full[start:, start:]


_InventoryLike: TypeAlias = _Inventory | _InitialInventory


def _linear(
    inventory: _InventoryLike,
    name: str,
    out_features: int,
    in_features: int,
    /,
    *,
    bias: bool,
    mixture: int,
) -> UMALinearMap:
    init = f"uniform:{in_features}"
    if mixture:
        weights = inventory.take(
            f"{name}.weights", (mixture, out_features, in_features), init
        )
        bias_ = inventory.take(f"{name}.bias", (out_features,), init) if bias else None
        return UMAMixtureLinear(weights, bias_)
    weight = inventory.take(f"{name}.weight", (out_features, in_features), init)
    bias_ = inventory.take(f"{name}.bias", (out_features,), init) if bias else None
    return UMALinear(weight, bias_)


def _radial(
    inventory: _InventoryLike, name: str, widths: Sequence[int], mixture: int, /
) -> UMARadialMLP:
    linears = []
    norms = []
    for layer, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        linears.append(
            _linear(
                inventory,
                f"{name}.net.{3 * layer}",
                fan_out,
                fan_in,
                bias=True,
                mixture=mixture,
            )
        )
        if layer < len(widths) - 2:
            norms.append(
                UMALayerNorm(
                    inventory.take(
                        f"{name}.net.{3 * layer + 1}.weight", (fan_out,), "ones"
                    ),
                    inventory.take(
                        f"{name}.net.{3 * layer + 1}.bias", (fan_out,), "zeros"
                    ),
                )
            )
    return UMARadialMLP(linears, norms)


def _normalization(
    inventory: _InventoryLike, name: str, architecture: UMAArchitecture, /
) -> UMANormalization:
    channels = architecture.sphere_channels
    lmax = architecture.lmax
    kind = architecture.normalization
    if kind == "layer_norm_sh":
        return UMANormalization(
            kind,
            lmax,
            weight=inventory.take(f"{name}.affine_weight", (lmax, channels), "ones"),
            scalar_weight=inventory.take(f"{name}.norm_l0.weight", (channels,), "ones"),
            scalar_bias=inventory.take(f"{name}.norm_l0.bias", (channels,), "zeros"),
        )
    return UMANormalization(
        kind,
        lmax,
        weight=inventory.take(f"{name}.affine_weight", (lmax + 1, channels), "ones"),
        bias=inventory.take(f"{name}.affine_bias", (channels,), "zeros"),
    )


def _so3_linear(
    inventory: _InventoryLike,
    name: str,
    lmax: int,
    out_features: int,
    in_features: int,
    /,
) -> UMASO3Linear:
    return UMASO3Linear(
        inventory.take(
            f"{name}.weight",
            (lmax + 1, out_features, in_features),
            f"uniform:{in_features}",
        ),
        inventory.take(f"{name}.bias", (out_features,), "zeros"),
    )


def _mixed(kind: str, layer_mixed: bool, slot: str, /) -> bool:
    """Whether the source ``mole_type`` replaces one linear slot of a block."""
    if not layer_mixed:
        return False
    match kind:
        case "so2":
            return slot in ("m0", "m")
        case "so2m0":
            return slot == "m0"
    raise ValueError(f"Unadmitted mixture kind {kind!r}.")


class _SharedMixtures:
    """Source ``moe_single`` cache: one bias-free mixture per ``(in, out)``."""

    def __init__(self) -> None:
        self.slots: dict[tuple[int, int], tuple[int, str]] = {}
        self.linears: list[UMAMixtureLinear] = []

    def reference(
        self,
        inventory: _InventoryLike,
        name: str,
        out_features: int,
        in_features: int,
        experts: int,
        /,
    ) -> UMASharedMixture:
        key = (in_features, out_features)
        tensor = f"{name}.weights"
        shape = (experts, out_features, in_features)
        if key not in self.slots:
            weights = inventory.take(tensor, shape, f"uniform:{in_features}")
            self.slots[key] = (len(self.linears), tensor)
            self.linears.append(UMAMixtureLinear(weights))
        else:
            slot, first = self.slots[key]
            if isinstance(inventory, _InitialInventory):
                inventory.tensors[tensor] = inventory.tensors[first].copy()
            else:
                weights = inventory.take(tensor, shape, "")
                if not np.array_equal(weights, np.asarray(self.linears[slot].weights)):
                    raise ValueError(
                        f"moe_single tensors {tensor!r} and {first!r} name one shared "
                        "source layer but differ."
                    )
        return UMASharedMixture(self.slots[key][0], in_features, out_features, experts)


def _block(
    inventory: _InventoryLike,
    architecture: UMAArchitecture,
    index: int,
    edge_grid: UMASphericalGrid | None,
    node_grid: UMASphericalGrid | None,
    shared: _SharedMixtures | None,
    /,
) -> UMABlock:
    prefix = f"backbone.blocks.{index}"
    layout = architecture.layout
    channels = architecture.sphere_channels
    hidden = architecture.hidden_channels
    mixed_layer = index in architecture.mixture_layers
    experts = architecture.experts
    kind = architecture.mixture_kind

    def experts_for(slot: str, /) -> int:
        return experts if _mixed(kind, mixed_layer, slot) else 0

    edge_input = architecture.distance_basis + 2 * architecture.edge_channels

    def convolution(
        name: str, cin: int, cout: int, extra: int, radial: bool, /
    ) -> UMASO2Convolution:
        degrees = layout.lmax + 1
        m0 = _linear(
            inventory,
            f"{name}.fc_m0",
            degrees * cout + extra,
            degrees * cin,
            bias=True,
            mixture=experts_for("m0"),
        )
        orders = []
        widths = [degrees * cin]
        for order in range(1, layout.mmax + 1):
            count = layout.lmax - order + 1
            mixture = experts_for("m")
            orders.append(
                shared.reference(
                    inventory,
                    f"{name}.so2_m_conv.{order - 1}.fc",
                    2 * count * cout,
                    count * cin,
                    mixture,
                )
                if shared is not None and mixture
                else _linear(
                    inventory,
                    f"{name}.so2_m_conv.{order - 1}.fc",
                    2 * count * cout,
                    count * cin,
                    bias=False,
                    mixture=mixture,
                )
            )
            widths.append(count * cin)
        radial_mlp = (
            _radial(
                inventory,
                f"{name}.rad_func",
                (
                    edge_input,
                    architecture.edge_channels,
                    architecture.edge_channels,
                    sum(widths),
                ),
                experts_for("radial"),
            )
            if radial
            else None
        )
        return UMASO2Convolution(
            layout,
            m0,
            orders,
            channels_in=cin,
            channels_out=cout,
            extra=extra,
            radial=radial_mlp,
        )

    extra = layout.lmax * hidden if architecture.edge_activation == "gate" else hidden
    edgewise = UMAEdgewise(
        convolution(f"{prefix}.edge_wise.so2_conv_1", 2 * channels, hidden, extra, True),
        convolution(f"{prefix}.edge_wise.so2_conv_2", hidden, channels, 0, False),
        activation=architecture.edge_activation,
        grid=edge_grid if architecture.edge_activation == "s2" else None,
    )
    atomwise_experts = experts_for("atomwise")
    if architecture.feed_forward == "grid":
        assert node_grid is not None
        atomwise = UMAGridAtomwise(
            (
                _linear(
                    inventory,
                    f"{prefix}.atom_wise.grid_mlp.0",
                    hidden,
                    channels,
                    bias=False,
                    mixture=atomwise_experts,
                ),
                _linear(
                    inventory,
                    f"{prefix}.atom_wise.grid_mlp.2",
                    hidden,
                    hidden,
                    bias=False,
                    mixture=atomwise_experts,
                ),
                _linear(
                    inventory,
                    f"{prefix}.atom_wise.grid_mlp.4",
                    channels,
                    hidden,
                    bias=False,
                    mixture=atomwise_experts,
                ),
            ),
            node_grid,
        )
    else:
        atomwise = UMASpectralAtomwise(
            _linear(
                inventory,
                f"{prefix}.atom_wise.scalar_mlp.0",
                layout.lmax * hidden,
                channels,
                bias=True,
                mixture=atomwise_experts,
            ),
            _so3_linear(
                inventory,
                f"{prefix}.atom_wise.so3_linear_1",
                layout.lmax,
                hidden,
                channels,
            ),
            _so3_linear(
                inventory,
                f"{prefix}.atom_wise.so3_linear_2",
                layout.lmax,
                channels,
                hidden,
            ),
        )
    return UMABlock(
        _normalization(inventory, f"{prefix}.norm_1", architecture),
        edgewise,
        _normalization(inventory, f"{prefix}.norm_2", architecture),
        atomwise,
    )


def _charge_spin(
    inventory: _InventoryLike,
    architecture: UMAArchitecture,
    target: Literal["charge", "spin"],
    /,
) -> UMAChargeSpinEmbedding | UMAFrozen:
    name = f"backbone.{target}_embedding"
    channels = architecture.sphere_channels
    kind = architecture.charge_spin_embedding
    match kind:
        case "pos_emb":
            embedding = UMAChargeSpinEmbedding(
                kind, target, inventory.take(f"{name}.W", (channels // 2,), "normal")
            )
        case "lin_emb":
            embedding = UMAChargeSpinEmbedding(
                kind,
                target,
                inventory.take(f"{name}.lin_emb.weight", (channels, 1), "uniform:1"),
                bias=inventory.take(f"{name}.lin_emb.bias", (channels,), "uniform:1"),
            )
        case "rand_emb":
            rows = 201 if target == "charge" else 101
            embedding = UMAChargeSpinEmbedding(
                kind,
                target,
                inventory.take(f"{name}.rand_emb.weight", (rows, channels), "normal"),
            )
    return embedding if architecture.charge_spin_trainable else UMAFrozen(embedding)


def _head_linears(
    inventory: _InventoryLike,
    prefix: str,
    kind: UMAEnergyHeadKind,
    channels: int,
    hidden: int,
    experts: int,
    /,
) -> UMAEnergyHead:
    if kind == "linear":
        return UMAEnergyHead(
            (_linear(inventory, prefix, 1, channels, bias=True, mixture=experts),)
        )
    return UMAEnergyHead(
        (
            _linear(
                inventory, f"{prefix}.0", hidden, channels, bias=True, mixture=experts
            ),
            _linear(inventory, f"{prefix}.2", hidden, hidden, bias=True, mixture=experts),
            _linear(inventory, f"{prefix}.4", 1, hidden, bias=True, mixture=experts),
        )
    )


def _references(declaration: UMASourceDeclaration, /) -> UMATaskReferences:
    architecture = declaration.architecture
    tasks = architecture.tasks
    heads = {head.name: head for head in declaration.heads}
    energy: list[tuple[int, float, float, np.ndarray | None]] = []
    forces: list[tuple[int, float, float]] = []
    stress: list[tuple[int, float, float]] = []
    for task in declaration.tasks:
        for dataset in task.datasets:
            code = tasks.index(dataset)
            match task.property:
                case "energy":
                    energy.append((code, task.rmsd, task.mean, task.element_references))
                case "forces" | "stress":
                    direct = any(
                        head.role != "energy"
                        and task.property in head.outputs
                        and (head.wrapping == "none" or dataset in head.datasets)
                        for head in heads.values()
                    )
                    if direct:
                        (forces if task.property == "forces" else stress).append(
                            (code, task.rmsd, task.mean)
                        )
    return UMATaskReferences(
        tasks=len(tasks),
        elements=architecture.max_elements,
        energy=energy,
        forces=forces,
        stress=stress,
    )


def _build(
    scale: AtomisticScaleContract,
    declaration: UMASourceDeclaration,
    inventory: _InventoryLike,
    /,
    *,
    precision: AtomisticPrecisionPolicy | None,
    streaming: StreamedRelationPlan | None,
    node_block: int,
) -> tuple[UMAModel, float]:
    architecture = declaration.architecture
    dtype = inventory.dtype
    channels = architecture.sphere_channels
    hidden = architecture.hidden_channels
    elements = architecture.max_elements
    layout = architecture.layout
    jd = [
        inventory.take(
            f"backbone.Jd_{degree}", (2 * degree + 1, 2 * degree + 1), f"jd:{degree}"
        )
        for degree in range(architecture.lmax + 1)
    ]
    residual = uma_source_basis_residual(jd) if architecture.lmax >= 1 else 0.0
    if residual > UMA_BASIS_TOLERANCE:
        raise ValueError(
            f"Stored e3nn J matrices deviate from the relabelled native basis by {residual}."
        )
    sphere = inventory.take(
        "backbone.sphere_embedding.weight", (elements, channels), "normal"
    )
    source_table = inventory.take(
        "backbone.source_embedding.weight",
        (elements, architecture.edge_channels),
        "small",
    )
    target_table = inventory.take(
        "backbone.target_embedding.weight",
        (elements, architecture.edge_channels),
        "small",
    )
    charge = _charge_spin(inventory, architecture, "charge")
    spin = _charge_spin(inventory, architecture, "spin")
    dataset = None
    if architecture.dataset_mapping is not None:
        rows = [
            inventory.take(
                f"backbone.dataset_embedding.dataset_emb_dict.{key}.weight",
                (1, channels),
                "normal",
            )[0]
            for key, _ in architecture.dataset_mapping
        ]
        table = UMADatasetEmbedding(np.stack(rows), architecture.dataset_rows)
        dataset = table if architecture.dataset_trainable else UMAFrozen(table)
    mix_width = (3 if dataset is not None else 2) * channels
    mix = _linear(
        inventory, "backbone.mix_csd", channels, mix_width, bias=True, mixture=0
    )
    assert isinstance(mix, UMALinear)
    system = UMASystemEmbedding(charge, spin, dataset, mix)
    router = None
    if architecture.experts:
        width = (2 if architecture.composition_embedding else 1) * channels
        experts = architecture.experts
        linears = []
        for index, (fan_in, fan_out) in enumerate(
            ((width, 2 * experts), (2 * experts, 2 * experts), (2 * experts, experts))
        ):
            linear = _linear(
                inventory,
                f"backbone.routing_mlp.{2 * index}",
                fan_out,
                fan_in,
                bias=True,
                mixture=0,
            )
            assert isinstance(linear, UMALinear)
            linears.append(linear)
        composition = (
            inventory.take(
                "backbone.composition_embedding.weight", (elements, channels), "normal"
            )
            if architecture.composition_embedding
            else None
        )
        router = UMAMoLERouter(
            linears,
            composition=composition,
            composition_policy=architecture.composition_policy,
            normalization=architecture.coefficient_norm,
        )
    edge_input = architecture.distance_basis + 2 * architecture.edge_channels
    edge_degree = UMAEdgeDegreeEmbedding(
        _radial(
            inventory,
            "backbone.edge_degree_embedding.rad_func",
            (
                edge_input,
                architecture.edge_channels,
                architecture.edge_channels,
                layout.m_sizes[0] * channels,
            ),
            0,
        ),
        rescale=5.0,
    )
    edge_grid = (
        UMASphericalGrid(
            architecture.lmax,
            architecture.mmax,
            resolution=architecture.grid_resolution,
            dtype=dtype,
        )
        if architecture.edge_activation == "s2"
        else None
    )
    node_grid = (
        UMASphericalGrid(
            architecture.lmax,
            architecture.lmax,
            resolution=architecture.grid_resolution,
            dtype=dtype,
        )
        if architecture.feed_forward == "grid"
        else None
    )
    shared = _SharedMixtures() if architecture.mixture_sharing else None
    blocks = [
        _block(inventory, architecture, index, edge_grid, node_grid, shared)
        for index in range(architecture.layer_count)
    ]
    norm = _normalization(inventory, "backbone.norm", architecture)
    heads = {head.name: head for head in declaration.heads}
    energy_heads = []
    for name in declaration.energy_heads:
        head = heads[name]
        block = "energy_block"
        energy_heads.append(
            _head_linears(
                inventory,
                f"{head.tensor_prefix}{block}",
                architecture.energy_head,
                channels,
                hidden,
                head.experts,
            )
        )
    force_head = None
    stress_head = None
    for head in declaration.heads:
        if head.role == "direct-forces":
            force_head = _so3_linear(
                inventory, f"{head.tensor_prefix}linear", 1, 1, channels
            )
        elif head.role == "direct-stress":
            stress_head = UMADirectStressHead(
                _head_linears(
                    inventory,
                    f"{head.tensor_prefix}scalar_block",
                    "mlp",
                    channels,
                    hidden,
                    head.experts,
                ),
                _so3_linear(inventory, f"{head.tensor_prefix}l2_linear", 2, 1, channels),
            )
    inventory.finish()
    network = UMANetwork(
        architecture,
        dtype=dtype,
        sphere_embedding=sphere,
        source_embedding=source_table,
        target_embedding=target_table,
        system=system,
        router=router,
        edge_degree=edge_degree,
        blocks=blocks,
        norm=norm,
        energy_heads=energy_heads,
        references=_references(declaration),
        force_head=force_head,
        stress_head=stress_head,
        streaming=streaming,
        node_block=node_block,
        shared_mixtures=() if shared is None else tuple(shared.linears),
    )
    model: UMAModel = (
        UMAPotential(scale, network, precision=precision)
        if architecture.model_kind == "potential"
        else UMAPropertyModel(scale, network, precision=precision)
    )
    return model, residual


def _precision_dtype(precision: AtomisticPrecisionPolicy | None, /) -> np.dtype:
    policy = AtomisticPrecisionPolicy() if precision is None else precision
    return np.dtype(policy.compute_dtype)


def uma_model_from_source(
    scale: AtomisticScaleContract,
    declaration: UMASourceDeclaration,
    tensors: Mapping[str, Any],
    /,
    *,
    precision: AtomisticPrecisionPolicy | None = None,
    streaming: StreamedRelationPlan | None = None,
    node_block: int = 1024,
) -> tuple[UMAModel, float]:
    """The declared model kind and stored-basis residual from exact source tensors.

    ``tensors`` maps source parameter names (``backbone.*``, ``output_heads.*``,
    without an EMA ``module.`` prefix) to arrays; each is consumed exactly once.
    A `UMAPotential` is returned when a scalar potential exists, otherwise a
    `UMAPropertyModel`.
    """
    if not isinstance(declaration, UMASourceDeclaration):
        raise TypeError("declaration must be a UMASourceDeclaration.")
    inventory = _Inventory(tensors, _precision_dtype(precision))
    return _build(
        scale,
        declaration,
        inventory,
        precision=precision,
        streaming=streaming,
        node_block=node_block,
    )


def uma_initial_source_tensors(
    declaration: UMASourceDeclaration, /, *, key: PRNGKey
) -> dict[str, np.ndarray]:
    """A complete source-layout inventory drawn with the source initializers."""
    from ...units import ANGSTROM, ELECTRONVOLT

    inventory = _InitialInventory(key, np.dtype(np.float64))
    _build(
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        declaration,
        inventory,
        precision=None,
        streaming=None,
        node_block=1024,
    )
    return dict(inventory.tensors)


__all__ = [
    "parse_uma_source_config",
    "UMA_BASIS_TOLERANCE",
    "UMA_SOURCE_COMMIT",
    "UMAHeadDeclaration",
    "UMASourceConfigurationError",
    "UMASourceDeclaration",
    "UMASourceTask",
    "uma_initial_source_tensors",
    "uma_model_from_source",
]
