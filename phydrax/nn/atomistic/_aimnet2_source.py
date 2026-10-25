#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact reconstruction of a native AIMNet2 model from a source declaration.

The input is the source model declaration (the safe-loaded ``model_yaml``
mapping ``{"class": ..., "kwargs": ...}`` of AIMNetcentral 0.2.0), the source
state tensors keyed by their state-dictionary names, and the source metadata.
Every declaration key is interpreted or refused and every tensor is consumed
once, recorded under a named source rule, or refused. Weights keep their source
layout (``Linear`` ``(out, in)``); nothing is re-derived from shapes.

Source rules, each recorded in ``AIMNet2SourceReconstruction``:

* ``aev.{rc,eta,shifts}_v`` are registered but never read by the forward (the
  vector channels use the scalar basis): recorded inert;
* all-NaN ``afv.weight`` rows mark unimplemented species (source export
  convention): zero rows behind an explicit per-case species refusal;
* ``outputs.{dipole,quadrupole}.mass`` without those modules and
  ``outputs.{lrcoulomb,dftd3,d3bj}.*`` of absent modules are the loader's
  expected-unexpected keys: recorded as dropped;
* a buffer missing from the state executes its constructor value (the loader's
  expected-missing ``outputs.srcoulomb.*`` and, for ``load_model``'s warning
  path, ``SRRep.rc`` and ``DispParam`` reference arguments); recorded;
* embedded ``LRCoulomb`` and ``DFTD3`` are calculator-equivalent long-range
  terms: ``externalize_source_long_range`` moves them into the calculator
  composition with an embedded ``SRCoulomb`` exactly where ``LRCoulomb``
  subtracted it, so total energies are unchanged and nothing is counted twice.

Refusals are limited to declarations the source cannot execute (a YAML string
where torch needs a ``dtype`` object), configurations the source forward fails
on (unequal ``nshifts_s``/``nshifts_v``), executable external files whose
contents are needed but not admitted (``DispParam.ptfile`` without a state
buffer), untyped auxiliary outputs written to other keys, and activations or
modules outside the admitted list.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from jax.typing import ArrayLike

from ...atomistic._dispersion_ts import TSDispersionPotential, TSDispersionReference
from ...atomistic._types import AtomisticPrecisionPolicy, AtomisticScaleContract
from ._aimnet2 import (
    AIMNET2_SPECIES_CAPACITY,
    AIMNet2Architecture,
    AIMNet2AtomicShifts,
    AIMNet2FrozenAtomicShifts,
    AIMNet2MassTable,
    AIMNet2MLP,
    AIMNet2OutputKind,
    AIMNet2OutputStage,
    AIMNet2Potential,
    AIMNet2Repulsion,
    AIMNet2ShortRangeRadius,
    StageValue,
)
from ._aimnet2_features import AIMNet2ConvSV, AIMNet2RadialBasis, SOURCE_MINIMUM_SHIFT


class AIMNet2SourceRefusedError(ValueError):
    """A source declaration or tensor outside the admitted AIMNet2 contract."""


_MODEL_CLASSES = frozenset({"aimnet.models.AIMNet2", "aimnet.models.aimnet2.AIMNet2"})
_MODEL_KEYS = frozenset(
    {
        "aev",
        "nfeature",
        "d2features",
        "ncomb_v",
        "hidden",
        "aim_size",
        "outputs",
        "num_charge_channels",
    }
)
_AEV_KEYS = frozenset(
    {
        "rmin",
        "rc_s",
        "nshifts_s",
        "eta_s",
        "rc_v",
        "nshifts_v",
        "eta_v",
        "shifts_s",
        "shifts_v",
    }
)
_MODULE_PATHS = {
    "output": ("aimnet.modules.Output", "aimnet.modules.core.Output"),
    "atomic-shift": ("aimnet.modules.AtomicShift", "aimnet.modules.core.AtomicShift"),
    "atomic-sum": ("aimnet.modules.AtomicSum", "aimnet.modules.core.AtomicSum"),
    "srcoulomb": ("aimnet.modules.SRCoulomb", "aimnet.modules.lr.SRCoulomb"),
    "srrep": ("aimnet.modules.core.SRRep",),
    "dipole": ("aimnet.modules.Dipole", "aimnet.modules.core.Dipole"),
    "quadrupole": ("aimnet.modules.Quadrupole", "aimnet.modules.core.Quadrupole"),
    "dispparam": ("aimnet.modules.DispParam", "aimnet.modules.lr.DispParam"),
    "d3ts": ("aimnet.modules.D3TS", "aimnet.modules.lr.D3TS"),
    "lrcoulomb": ("aimnet.modules.LRCoulomb", "aimnet.modules.lr.LRCoulomb"),
    "dftd3": ("aimnet.modules.DFTD3", "aimnet.modules.lr.DFTD3"),
}
# ``torch.nn`` activation classes (and their keyword arguments) with native
# semantics in ``apply_activation``; ``inplace`` does not change values.
_ACTIVATIONS = {
    "GELU": ("gelu", frozenset({"approximate"})),
    "SiLU": ("silu", frozenset({"inplace"})),
    "ReLU": ("relu", frozenset({"inplace"})),
    "Tanh": ("tanh", frozenset()),
    "Sigmoid": ("sigmoid", frozenset()),
    "Softplus": ("softplus", frozenset({"beta", "threshold"})),
}
# ``models/utils.validate_state_dict_keys``: keys the source loader drops.
_EXPECTED_DROPPED_PREFIXES = ("outputs.lrcoulomb.", "outputs.dftd3.", "outputs.d3bj.")
_EXPECTED_DROPPED_KEYS = ("outputs.dipole.mass", "outputs.quadrupole.mass")
_DFTD3_BUFFERS = ("rcov", "r4r2", "c6ab", "cn_ref")
# ``calculators/calculator.py``: ``_WB97M_D3_PARAMS`` and the external DFTD3
# defaults the calculator constructs.
SOURCE_RXN_D3_PARAMETERS = MappingProxyType(
    {"s6": 1.0, "s8": 0.3908, "a1": 0.566, "a2": 3.128}
)
SOURCE_EXTERNAL_D3_CUTOFF = 15.0
SOURCE_EXTERNAL_D3_SMOOTHING_FRACTION = 0.2
SOURCE_COULOMB_CONSTANT = 27.211386024367243 * 0.5291772105638411
_LRCOULOMB_DEFAULTS = {
    "key_in": "charges",
    "key_out": "e_h",
    "rc": 4.6,
    "method": "simple",
    "dsf_alpha": 0.2,
    "dsf_rc": 15.0,
    "ewald_accuracy": 1e-6,
    "subtract_sr": True,
    "envelope": "exp",
}


def _refuse(message: str, /) -> AIMNet2SourceRefusedError:
    return AIMNet2SourceRefusedError(message)


def _class_kind(path: Any, /) -> str:
    if not isinstance(path, str):
        raise _refuse("Source module classes must be import-path strings.")
    for kind, paths in _MODULE_PATHS.items():
        if path in paths:
            return kind
    raise _refuse(f"Source output module {path!r} is not an admitted AIMNet2 module.")


def _integer(value: Any, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _refuse(f"{name} must be a positive integer.")
    return value


def _real(value: Any, name: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(f"{name} must be a real number.")
    if not math.isfinite(float(value)):
        raise _refuse(f"{name} must be finite.")
    return float(value)


def _flag(value: Any, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise _refuse(f"{name} must be a boolean.")
    return value


def _keys(mapping: Any, allowed: frozenset[str] | set[str], name: str, /) -> dict:
    if not isinstance(mapping, Mapping):
        raise _refuse(f"{name} must be a mapping.")
    extra = sorted(set(mapping) - set(allowed))
    if extra:
        raise _refuse(f"{name} declares unadmitted keys {extra}.")
    return dict(mapping)


def _digest(value: np.ndarray, /) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)})
    return hashlib.sha256(header.encode("utf-8") + array.tobytes()).hexdigest()


class _Inventory:
    """Every tensor consumed once, recorded under a named rule, or refused."""

    def __init__(self, tensors: Mapping[str, ArrayLike], /) -> None:
        if not isinstance(tensors, Mapping):
            raise TypeError("tensors must map source state names to arrays.")
        self._tensors = {str(name): np.asarray(value) for name, value in tensors.items()}
        self._consumed: set[str] = set()
        self.inert: dict[str, str] = {}

    def present(self, name: str, /) -> bool:
        return name in self._tensors

    def shape(self, name: str, /) -> tuple[int, ...]:
        if name not in self._tensors:
            raise _refuse(f"Source tensor {name!r} is missing.")
        return tuple(self._tensors[name].shape)

    def array(
        self,
        name: str,
        shape: tuple[int, ...],
        /,
        *,
        finite: bool = True,
    ) -> np.ndarray:
        if name not in self._tensors:
            raise _refuse(f"Source tensor {name!r} is missing.")
        if name in self._consumed or name in self.inert:
            raise RuntimeError(f"Source tensor {name!r} was consumed twice.")
        value = self._tensors[name]
        if value.dtype.kind != "f":
            raise _refuse(f"Source tensor {name!r} must be floating point.")
        if value.shape != shape:
            raise _refuse(
                f"Source tensor {name!r} has shape {value.shape}; expected {shape}."
            )
        if finite and not np.all(np.isfinite(value)):
            raise _refuse(f"Source tensor {name!r} must be finite.")
        self._consumed.add(name)
        return value

    def mark_inert(self, name: str, rule: str, /) -> np.ndarray:
        if name in self._consumed or name in self.inert:
            raise RuntimeError(f"Source tensor {name!r} was consumed twice.")
        self.inert[name] = rule
        return self._tensors[name]

    def finish(self) -> None:
        remaining = sorted(set(self._tensors) - self._consumed - set(self.inert))
        for name in remaining:
            if name in _EXPECTED_DROPPED_KEYS or name.startswith(
                _EXPECTED_DROPPED_PREFIXES
            ):
                self.inert[name] = "source-loader-expected-unexpected-key"
        extra = sorted(set(self._tensors) - self._consumed - set(self.inert))
        if extra:
            raise _refuse(f"Unadmitted source tensors: {extra}.")


@dataclass(frozen=True, slots=True)
class AIMNet2ExternalizedTerms:
    """Embedded long-range modules moved into the calculator composition.

    ``coulomb`` records the ``LRCoulomb`` options (method, executed ``rc``,
    envelope, ``subtract_sr`` and the DSF/Ewald/PME parameters); when
    ``subtract_sr`` is true an ``SRCoulomb`` with the same ``rc`` and envelope
    replaces it in the model, so model plus full external Coulomb equals the
    source. ``dispersion`` records ``DFTD3`` BJ constants, cutoff/smoothing and
    digests of the externalized reference buffers the source carried.
    """

    coulomb: Mapping[str, Any] | None
    dispersion: Mapping[str, Any] | None
    rule: str

    def to_record(self) -> dict[str, Any]:
        return {
            "coulomb": None if self.coulomb is None else dict(self.coulomb),
            "dispersion": None if self.dispersion is None else dict(self.dispersion),
            "rule": self.rule,
        }


def _externalized_coulomb(
    name: str,
    options: Any,
    state: dict[str, np.ndarray],
    legacy_attributes: Mapping[str, Any] | None,
    /,
) -> dict[str, Any]:
    """One embedded ``LRCoulomb``'s options and executed ``rc`` (popped from state)."""
    options = {
        **_LRCOULOMB_DEFAULTS,
        **_keys(options, set(_LRCOULOMB_DEFAULTS), f"outputs.{name}"),
    }
    if options["key_in"] != "charges" or options["key_out"] != "energy":
        raise _refuse("LRCoulomb writing another key is an untyped auxiliary output.")
    if options["method"] not in ("simple", "dsf", "ewald", "pme"):
        raise _refuse(f"Unknown LRCoulomb method {options['method']!r}.")
    if options["envelope"] not in ("exp", "cosine"):
        raise _refuse("LRCoulomb envelope must be 'exp' or 'cosine'.")
    buffer = f"outputs.{name}.rc"
    if legacy_attributes is not None:
        radius = legacy_attributes.get("lrcoulomb_rc")
        if radius is None:
            raise _refuse("The legacy LRCoulomb rc could not be determined.")
        envelope = "exp"
        state.pop(buffer, None)
    elif buffer in state:
        rc = state.pop(buffer)
        if rc.shape != () or rc.dtype.kind != "f":
            raise _refuse(f"{buffer} must be a floating scalar.")
        radius, envelope = float(rc), options["envelope"]
    else:
        radius = float(np.float32(_real(options["rc"], "LRCoulomb rc")))
        envelope = options["envelope"]
    subtract = _flag(options["subtract_sr"], "subtract_sr")
    return {
        "module": name,
        "method": options["method"],
        "rc": float(radius),
        "envelope": envelope,
        "subtract_sr": subtract,
        "dsf_alpha": _real(options["dsf_alpha"], "dsf_alpha"),
        "dsf_rc": _real(options["dsf_rc"], "dsf_rc"),
        "ewald_accuracy": _real(options["ewald_accuracy"], "ewald_accuracy"),
    }


def _externalized_dispersion(
    name: str,
    options: Any,
    state: dict[str, np.ndarray],
    legacy_attributes: Mapping[str, Any] | None,
    /,
) -> dict[str, Any]:
    """One embedded ``DFTD3``'s BJ constants and reference-buffer digests."""
    options = _keys(
        options,
        {"s8", "a1", "a2", "s6", "cutoff", "smoothing_fraction", "key_out"},
        f"outputs.{name}",
    )
    if options.get("key_out", "energy") != "energy":
        raise _refuse("DFTD3 writing another key is an untyped auxiliary output.")
    values = {
        "s8": options.get("s8", 0.0),
        "a1": options.get("a1", 0.0),
        "a2": options.get("a2", 0.0),
        "s6": options.get("s6", 1.0),
    }
    attributes = (legacy_attributes or {}).get("d3_params")
    if attributes and not (
        attributes.get("s8") == 0.0
        and attributes.get("a1") == 0.0
        and attributes.get("a2") == 0.0
    ):
        values = dict(attributes)
    buffers = {}
    for buffer in _DFTD3_BUFFERS:
        key = f"outputs.{name}.{buffer}"
        if key in state:
            buffers[key] = _digest(state.pop(key))
    return {
        "module": name,
        **{key: _real(values[key], f"DFTD3 {key}") for key in values},
        "cutoff": _real(options.get("cutoff", 15.0), "DFTD3 cutoff"),
        "smoothing_fraction": _real(
            options.get("smoothing_fraction", 0.2), "DFTD3 smoothing_fraction"
        ),
        "reference_buffers": buffers,
    }


def _embed_short_range_coulomb(
    outputs: list[tuple[str, Any]],
    state: dict[str, np.ndarray],
    coulomb: Mapping[str, Any],
    /,
) -> None:
    """``SRCoulomb`` exactly where the externalized ``LRCoulomb`` subtracted it."""
    if any(name == "srcoulomb" for name, _ in outputs):
        raise _refuse("The model embeds both LRCoulomb and SRCoulomb.")
    outputs.append(
        (
            "srcoulomb",
            {
                "class": "aimnet.modules.SRCoulomb",
                "kwargs": {
                    "rc": coulomb["rc"],
                    "key_in": "charges",
                    "key_out": "energy",
                    "envelope": coulomb["envelope"],
                },
            },
        )
    )
    state["outputs.srcoulomb.rc"] = np.asarray(coulomb["rc"], dtype=np.float64)


def externalize_source_long_range(
    declaration: Mapping[str, Any],
    tensors: Mapping[str, ArrayLike],
    /,
    *,
    legacy_attributes: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], AIMNet2ExternalizedTerms]:
    """Move embedded ``LRCoulomb``/``DFTD3`` out of the model, exactly.

    Mirrors ``models/utils.strip_lr_modules_from_yaml`` generalized to every
    ``LRCoulomb`` option: the executed ``rc`` is the state buffer when present
    (else the constructor value); legacy TorchScript models use the JIT module
    attributes and the source's legacy ``"exp"`` envelope. ``DFTD3`` BJ
    constants come from the JIT attributes when nonzero (legacy) or the YAML.
    """
    root = _keys(declaration, {"class", "kwargs"}, "model declaration")
    kwargs = _keys(root.get("kwargs"), _MODEL_KEYS, "AIMNet2 kwargs")
    entries = _output_entries(kwargs.get("outputs"))
    state = {str(name): np.asarray(value) for name, value in tensors.items()}
    outputs: list[tuple[str, Any]] = []
    coulomb = None
    dispersion = None
    for name, entry in entries:
        kind = _class_kind(entry["class"])
        options = entry.get("kwargs") or {}
        if kind == "lrcoulomb":
            if coulomb is not None:
                raise _refuse("At most one embedded LRCoulomb is admitted.")
            coulomb = _externalized_coulomb(name, options, state, legacy_attributes)
        elif kind == "dftd3":
            if dispersion is not None:
                raise _refuse("At most one embedded DFTD3 is admitted.")
            dispersion = _externalized_dispersion(name, options, state, legacy_attributes)
        else:
            outputs.append((name, entry))
    if coulomb is not None and coulomb["subtract_sr"]:
        _embed_short_range_coulomb(outputs, state, coulomb)
    if (
        any(_class_kind(entry["class"]) == "d3ts" for _, entry in outputs)
        and dispersion is not None
    ):
        raise _refuse("The model embeds both D3TS and tabulated DFT-D3.")
    core = {
        "class": root["class"],
        "kwargs": {**kwargs, "outputs": dict(outputs)},
    }
    terms = AIMNet2ExternalizedTerms(
        coulomb=None if coulomb is None else MappingProxyType(coulomb),
        dispersion=None if dispersion is None else MappingProxyType(dispersion),
        rule="aimnet.models.utils.strip_lr_modules_from_yaml (generalized)",
    )
    return core, state, terms


@dataclass(frozen=True, slots=True)
class AIMNet2CalculatorComposition:
    """Source calculator terms composed around the model energy (never inside it).

    ``coulomb_mode`` names what the model embeds: ``"sr_embedded"`` (the model
    subtracts the short-range part) or ``"none"``. With ``external_coulomb`` the
    calculator adds full Coulomb of the predicted charges by
    ``external_coulomb_method`` (calculator default ``"simple"``: all pairs,
    ``k = Hartree * Bohr``; externalized DSF/Ewald/PME keep their options), and
    subtracts the short-range part only when ``subtract_short_range``.
    ``external_dispersion`` is tabulated DFT-D3(BJ) without three-body terms.
    ``family_defaults`` names calculator family rules that changed the stored
    metadata (the rxn family forces neutral charge and wB97M D3 parameters
    unless D3TS is embedded); ``externalized`` records modules moved out of the
    model declaration.
    """

    coulomb_mode: str
    external_coulomb: bool
    external_coulomb_method: str
    external_coulomb_options: Mapping[str, float]
    coulomb_constant: float
    subtract_short_range: bool
    short_range_radius: float | None
    short_range_envelope: str | None
    external_dispersion: bool
    d3_parameters: Mapping[str, float] | None
    d3_cutoff: float
    d3_smoothing_fraction: float
    embedded_ts_dispersion: bool
    family: str | None
    supports_charged_systems: bool | None
    family_defaults: tuple[str, ...]
    externalized: Mapping[str, Any] | None

    def to_record(self) -> dict[str, Any]:
        return {
            "coulomb_mode": self.coulomb_mode,
            "external_coulomb": self.external_coulomb,
            "external_coulomb_method": self.external_coulomb_method,
            "external_coulomb_options": dict(self.external_coulomb_options),
            "coulomb_constant": self.coulomb_constant,
            "subtract_short_range": self.subtract_short_range,
            "short_range_radius": self.short_range_radius,
            "short_range_envelope": self.short_range_envelope,
            "external_dispersion": self.external_dispersion,
            "d3_parameters": None
            if self.d3_parameters is None
            else dict(self.d3_parameters),
            "d3_cutoff": self.d3_cutoff,
            "d3_smoothing_fraction": self.d3_smoothing_fraction,
            "embedded_ts_dispersion": self.embedded_ts_dispersion,
            "family": self.family,
            "supports_charged_systems": self.supports_charged_systems,
            "family_defaults": list(self.family_defaults),
            "externalized": None
            if self.externalized is None
            else dict(self.externalized),
        }


def _externalized_coulomb_settings(
    coulomb: Mapping[str, Any], /
) -> tuple[str, dict[str, float], str, Any, Any]:
    """Calculator Coulomb for an externalized ``LRCoulomb``.

    Returns ``(method, options, coulomb_mode, sr_rc, sr_envelope)``; the model
    embeds the short-range part exactly when ``LRCoulomb`` subtracted it.
    """
    subtract = coulomb["subtract_sr"]
    return (
        coulomb["method"],
        {key: coulomb[key] for key in ("dsf_alpha", "dsf_rc", "ewald_accuracy")},
        "sr_embedded" if subtract else "none",
        coulomb["rc"] if subtract else None,
        coulomb["envelope"] if subtract else None,
    )


def _apply_family_defaults(
    family: Any,
    supports: Any,
    embedded_ts: bool,
    needs_dispersion: bool,
    d3: Any,
    /,
) -> tuple[Any, bool, Any, list[str]]:
    """Source ``_apply_family_defaults``: ``(supports, needs_dispersion, d3, defaults)``.

    The rxn family is neutral-only and needs wB97M DFT-D3 unless D3TS is embedded.
    """
    defaults: list[str] = []
    if family is not None and not isinstance(family, str):
        raise _refuse("Source family must be text.")
    if family == "rxn":
        if supports is None:
            supports = False
            defaults.append("rxn-supports-charged-systems-false")
        elif supports is not False:
            raise _refuse(
                "aimnet2-rxn models must declare supports_charged_systems=False."
            )
        if not embedded_ts:
            if not needs_dispersion:
                defaults.append("rxn-needs-dispersion")
            needs_dispersion = True
            if d3 is None:
                d3 = dict(SOURCE_RXN_D3_PARAMETERS)
                defaults.append("rxn-wb97m-d3-parameters")
    if supports is not None and not isinstance(supports, bool):
        raise _refuse("supports_charged_systems must be a boolean or absent.")
    return supports, needs_dispersion, d3, defaults


def _d3_parameters(d3: Any, /) -> Mapping[str, float]:
    """The external DFT-D3(BJ) constants; ``s6`` defaults to one."""
    if not isinstance(d3, Mapping) or not {"s8", "a1", "a2"} <= set(d3):
        raise _refuse("needs_dispersion requires d3_params with s8, a1 and a2.")
    return MappingProxyType(
        {
            "s6": _real(d3.get("s6", 1.0), "d3_params.s6"),
            "s8": _real(d3["s8"], "d3_params.s8"),
            "a1": _real(d3["a1"], "d3_params.a1"),
            "a2": _real(d3["a2"], "d3_params.a2"),
        }
    )


def aimnet2_calculator_composition(
    metadata: Mapping[str, Any],
    /,
    *,
    externalized: AIMNet2ExternalizedTerms | None = None,
) -> AIMNet2CalculatorComposition:
    """Apply the source calculator's metadata and family rules exactly.

    ``metadata`` is the source model metadata (``.pt`` top-level fields or
    ``config.json``). Mirrors ``AIMNet2Calculator.__init__`` and
    ``_apply_family_defaults`` without runtime overrides. Externalized
    embedded modules join the composition; metadata that already requests the
    same external term refuses (it would be counted twice).
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")
    family = metadata.get("family")
    supports = metadata.get("supports_charged_systems")
    needs_coulomb = bool(metadata.get("needs_coulomb", False))
    needs_dispersion = bool(metadata.get("needs_dispersion", False))
    mode = metadata.get("coulomb_mode", "none")
    sr_rc = metadata.get("coulomb_sr_rc")
    envelope = metadata.get("coulomb_sr_envelope")
    d3 = metadata.get("d3_params")
    cutoff = SOURCE_EXTERNAL_D3_CUTOFF
    smoothing = SOURCE_EXTERNAL_D3_SMOOTHING_FRACTION
    method = "simple"
    coulomb_options: dict[str, float] = {}
    if mode not in ("sr_embedded", "full_embedded", "none"):
        raise _refuse(f"Unknown source coulomb_mode {mode!r}.")
    if externalized is not None and externalized.coulomb is not None:
        if needs_coulomb:
            raise _refuse(
                "Metadata requests external Coulomb for a model that embeds it."
            )
        needs_coulomb = True
        method, coulomb_options, mode, sr_rc, envelope = _externalized_coulomb_settings(
            externalized.coulomb
        )
    if externalized is not None and externalized.dispersion is not None:
        if needs_dispersion:
            raise _refuse("Metadata requests external DFT-D3 for a model that embeds it.")
        dispersion = externalized.dispersion
        needs_dispersion = True
        d3 = {key: dispersion[key] for key in ("s6", "s8", "a1", "a2")}
        cutoff = dispersion["cutoff"]
        smoothing = dispersion["smoothing_fraction"]
    embedded_ts = bool(metadata.get("has_embedded_d3ts", False))
    supports, needs_dispersion, d3, defaults = _apply_family_defaults(
        family, supports, embedded_ts, needs_dispersion, d3
    )
    if needs_coulomb and mode == "sr_embedded" and sr_rc is None:
        sr_rc = 4.6
    if needs_coulomb and mode == "sr_embedded" and envelope is None:
        envelope = "exp"
    parameters = _d3_parameters(d3) if needs_dispersion else None
    sr_embedded = mode == "sr_embedded"
    return AIMNet2CalculatorComposition(
        coulomb_mode=mode,
        external_coulomb=needs_coulomb,
        external_coulomb_method=method,
        external_coulomb_options=MappingProxyType(coulomb_options),
        coulomb_constant=SOURCE_COULOMB_CONSTANT,
        subtract_short_range=needs_coulomb
        and not sr_embedded
        and (externalized is None or externalized.coulomb is None),
        short_range_radius=_real(sr_rc, "coulomb_sr_rc")
        if sr_embedded and sr_rc is not None
        else None,
        short_range_envelope=envelope if sr_embedded else None,
        external_dispersion=needs_dispersion,
        d3_parameters=parameters,
        d3_cutoff=float(cutoff),
        d3_smoothing_fraction=float(smoothing),
        embedded_ts_dispersion=embedded_ts,
        family=family,
        supports_charged_systems=supports,
        family_defaults=tuple(defaults),
        externalized=None
        if externalized is None
        else MappingProxyType(externalized.to_record()),
    )


@dataclass(frozen=True, slots=True)
class AIMNet2SourceReconstruction:
    """A reconstructed native model and every source rule applied to it."""

    potential: AIMNet2Potential
    architecture: AIMNet2Architecture
    inert_tensors: Mapping[str, str]
    unimplemented_rows: tuple[int, ...]
    constructor_values: Mapping[str, Any]
    superseded_declarations: Mapping[str, str]
    declared_aev: Mapping[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture.architecture_id,
            "potential_architecture_id": self.potential.architecture_id,
            "inert_tensors": dict(sorted(self.inert_tensors.items())),
            "unimplemented_rows": list(self.unimplemented_rows),
            "constructor_values": dict(self.constructor_values),
            "superseded_declarations": dict(self.superseded_declarations),
            "declared_aev": dict(self.declared_aev),
        }


def _output_entries(outputs: Any, /) -> list[tuple[str, Mapping[str, Any]]]:
    if isinstance(outputs, Mapping):
        entries = [(str(name), value) for name, value in outputs.items()]
    elif isinstance(outputs, Sequence) and not isinstance(outputs, str):
        entries = [(str(index), value) for index, value in enumerate(outputs)]
    else:
        raise _refuse("`outputs` is neither a list nor a dict.")
    for name, value in entries:
        _keys(value, {"class", "kwargs"}, f"outputs.{name}")
        if "class" not in value:
            raise _refuse(f"outputs.{name} declares no class.")
    return entries


def _activation(mlp: Mapping[str, Any], name: str, /) -> tuple[str, tuple[float, ...]]:
    path = mlp.get("activation_fn", "torch.nn.GELU")
    if not isinstance(path, str):
        raise _refuse(f"outputs.{name} activation must be an import path.")
    class_name = path.rsplit(".", 1)[-1]
    module = path.rsplit(".", 1)[0] if "." in path else ""
    if module not in ("torch.nn", "torch.nn.modules.activation") or (
        class_name not in _ACTIVATIONS
    ):
        raise _refuse(f"outputs.{name} activation {path!r} is not admitted.")
    native, allowed = _ACTIVATIONS[class_name]
    options = _keys(mlp.get("activation_kwargs") or {}, allowed, f"outputs.{name}")
    if "inplace" in options:
        _flag(options["inplace"], "inplace")
    if native == "gelu":
        approximate = options.get("approximate", "none")
        if approximate not in ("none", "tanh"):
            raise _refuse(f"GELU approximation {approximate!r} is unknown.")
        return ("gelu-tanh" if approximate == "tanh" else "gelu", ())
    if native == "softplus":
        return (
            "softplus",
            (
                _real(options.get("beta", 1.0), "Softplus beta"),
                _real(options.get("threshold", 20.0), "Softplus threshold"),
            ),
        )
    return (native, ())


def _head_mlp(
    kwargs: Mapping[str, Any], name: str, /
) -> tuple[tuple[int, ...], bool, bool, tuple[str, tuple[float, ...]]]:
    mlp = _keys(
        kwargs.get("mlp"),
        {
            "hidden",
            "activation_fn",
            "activation_kwargs",
            "weight_init_fn",
            "bias",
            "last_linear",
        },
        f"outputs.{name}.mlp",
    )
    if "weight_init_fn" in mlp and not isinstance(mlp["weight_init_fn"], str):
        raise _refuse("weight_init_fn must be an import path (initialization only).")
    hidden = mlp.get("hidden") or []
    if not isinstance(hidden, Sequence) or isinstance(hidden, str):
        raise _refuse(f"outputs.{name}.mlp.hidden must be a list.")
    widths = []
    for width in hidden:
        if isinstance(width, bool) or not isinstance(width, int):
            raise _refuse(f"outputs.{name}.mlp.hidden must contain integers.")
        if width > 0:
            widths.append(width)
    return (
        tuple(widths),
        _flag(mlp.get("last_linear", True), f"outputs.{name}.last_linear"),
        _flag(mlp.get("bias", True), f"outputs.{name}.bias"),
        _activation(mlp, name),
    )


def _admit_aev_shifts(aev_declaration: Any, /) -> tuple[int, Any]:
    """Declared ``(nshifts_s, nshifts_v)``; the vector basis must match the forward."""
    aev = _keys(aev_declaration, _AEV_KEYS, "aev")
    if "nshifts_s" not in aev:
        raise _refuse("aev must declare nshifts_s (the source indexes it directly).")
    shifts_s = _integer(aev["nshifts_s"], "aev.nshifts_s")
    shifts_v = aev.get("nshifts_v") or shifts_s
    if aev.get("rc_v") is not None:
        if _real(aev["rc_v"], "aev.rc_v") > _real(aev.get("rc_s", 5.0), "aev.rc_s"):
            raise _refuse("rc_v must be less than or equal to rc_s.")
        if aev.get("nshifts_v") is None:
            raise _refuse("A declared vector basis needs nshifts_v.")
    if shifts_v != shifts_s:
        # The source forward contracts agh (nshifts_v) with scalar-basis
        # vector channels (nshifts_s); unequal counts fail in the source.
        raise _refuse("nshifts_v differs from nshifts_s; the source forward fails.")
    return shifts_s, shifts_v


def _admit_hidden(hidden: Any, /) -> list[tuple[int, ...]]:
    """The interaction-pass hidden width lists."""
    if not isinstance(hidden, Sequence) or isinstance(hidden, str) or not hidden:
        raise _refuse("hidden must be a nonempty list of width lists.")
    parsed_hidden = []
    for widths in hidden:
        if not isinstance(widths, Sequence) or isinstance(widths, str):
            raise _refuse("hidden entries must be width lists.")
        if any(isinstance(width, bool) or not isinstance(width, int) for width in widths):
            raise _refuse("hidden widths must be integers.")
        parsed_hidden.append(tuple(widths))
    return parsed_hidden


def _head_stage(
    name: str, options: Any, aim_size: int, consumers: set[Any], /
) -> AIMNet2OutputStage:
    """An ``Output`` MLP of the AIM vector: the energy head or a DispParam input."""
    options = _keys(options, {"mlp", "n_in", "n_out", "key_in", "key_out"}, name)
    if options.get("key_in") != "aim" or options.get("n_in") != aim_size:
        raise _refuse(f"Output {name!r} must read the AIM vector.")
    widths, last_linear, bias, activation = _head_mlp(options, name)
    key_out = options.get("key_out")
    if key_out == "energy" and options.get("n_out") == 1:
        head = AIMNet2OutputKind.ENERGY_HEAD
    elif key_out in consumers and options.get("n_out") == 2:
        head = AIMNet2OutputKind.DISPERSION_HEAD
    else:
        raise _refuse(
            f"Output {name!r} ({key_out!r}, n_out={options.get('n_out')}) "
            "is an untyped auxiliary output, not the energy head or a "
            "DispParam input."
        )
    return AIMNet2OutputStage(
        name,
        head,
        hidden=widths,
        last_linear=last_linear,
        bias=bias,
        activation=activation,
        key_in="aim",
        key_out=key_out,
    )


def _atomic_shift_stage(name: str, options: Any, /) -> AIMNet2OutputStage:
    """``AtomicShift``: per-species shifts, frozen when ``requires_grad`` is false."""
    options = _keys(
        options,
        {
            "key_in",
            "key_out",
            "num_types",
            "dtype",
            "requires_grad",
            "reduce_sum",
        },
        name,
    )
    if "dtype" in options:
        raise _refuse(
            "AtomicShift dtype must be a torch dtype object; a YAML value "
            "cannot express one and fails in the source constructor."
        )
    rows = options.get("num_types", AIMNET2_SPECIES_CAPACITY)
    return AIMNet2OutputStage(
        name,
        AIMNet2OutputKind.ATOMIC_SHIFT,
        reduce_sum=_flag(options.get("reduce_sum", False), "reduce_sum"),
        frozen=not _flag(options.get("requires_grad", True), "requires_grad"),
        species_rows=_integer(rows, "num_types"),
        key_in=options.get("key_in"),
        key_out=options.get("key_out"),
    )


def _repulsion_stage(
    name: str,
    options: Any,
    radii: Mapping[str, float],
    neighbor_cutoff: float | None,
    /,
) -> AIMNet2OutputStage:
    """Energy-bearing ``SRRep`` over the model neighbor list."""
    options = _keys(options, {"key_out", "cutoff_fn", "rc", "reduce_sum"}, name)
    if options.get("key_out", "e_rep") != "energy":
        raise _refuse(
            "SRRep writing its default 'e_rep' key is an untyped auxiliary "
            "output; only energy-bearing SRRep is admitted."
        )
    function = options.get("cutoff_fn", "none")
    if neighbor_cutoff is None or name not in radii:
        raise _refuse("SRRep sums the model neighbor list; declare neighbor_cutoff.")
    route = (
        float(neighbor_cutoff)
        if function == "none"
        else min(radii[name], float(neighbor_cutoff))
    )
    return AIMNet2OutputStage(
        name,
        AIMNet2OutputKind.SHORT_RANGE_REPULSION,
        cutoff_function=function,
        reduce_sum=_flag(options.get("reduce_sum", True), "reduce_sum"),
        pair_cutoff=route,
        key_out="energy",
    )


def _ts_dispersion_stage(
    name: str, options: Any, ts_pair_cutoff: float | None, /
) -> AIMNet2OutputStage:
    """Embedded ``D3TS`` over the declared hard calculator route radius."""
    options = _keys(options, {"a1", "a2", "s8", "s6", "key_in", "key_out"}, name)
    if ts_pair_cutoff is None:
        raise _refuse("D3TS sums its calculator neighbor list; declare ts_pair_cutoff.")
    return AIMNet2OutputStage(
        name,
        AIMNet2OutputKind.TS_DISPERSION,
        s6=_real(options.get("s6", 1.0), "s6"),
        s8=_real(options.get("s8"), "s8"),
        a1=_real(options.get("a1"), "a1"),
        a2=_real(options.get("a2"), "a2"),
        pair_cutoff=float(ts_pair_cutoff),
        key_in=options.get("key_in", "disp_param"),
        key_out=options.get("key_out", "energy"),
    )


def _output_stages(
    entries: Sequence[tuple[str, Mapping[str, Any]]],
    aim_size: int,
    radii: Mapping[str, float],
    ts_pair_cutoff: float | None,
    neighbor_cutoff: float | None,
    /,
) -> list[AIMNet2OutputStage]:
    """Output modules in source order; every admitted kind is typed or refused."""
    consumers = {
        (entry.get("kwargs") or {}).get("key_in", "disp_param")
        for _, entry in entries
        if _class_kind(entry["class"]) == "dispparam"
    }
    stages = []
    for name, entry in entries:
        kind = _class_kind(entry["class"])
        options = entry.get("kwargs") or {}
        match kind:
            case "output":
                stages.append(_head_stage(name, options, aim_size, consumers))
            case "atomic-shift":
                stages.append(_atomic_shift_stage(name, options))
            case "atomic-sum":
                options = _keys(options, {"key_in", "key_out"}, name)
                stages.append(
                    AIMNet2OutputStage(
                        name,
                        AIMNet2OutputKind.ATOMIC_SUM,
                        key_in=options.get("key_in"),
                        key_out=options.get("key_out"),
                    )
                )
            case "srcoulomb":
                options = _keys(options, {"rc", "key_in", "key_out", "envelope"}, name)
                if name not in radii:
                    raise _refuse(f"SRCoulomb {name!r} has no executed radius.")
                stages.append(
                    AIMNet2OutputStage(
                        name,
                        AIMNet2OutputKind.SHORT_RANGE_COULOMB,
                        envelope=options.get("envelope", "exp"),
                        pair_cutoff=radii[name],
                        key_in=options.get("key_in", "charges"),
                        key_out=options.get("key_out", "energy"),
                    )
                )
            case "srrep":
                stages.append(_repulsion_stage(name, options, radii, neighbor_cutoff))
            case "dipole" | "quadrupole":
                options = _keys(options, {"key_in", "key_out", "center_coord"}, name)
                if options.get("key_in", "charges") != "charges":
                    raise _refuse(
                        f"{kind} of {options['key_in']!r} has no typed observable "
                        "(only charge moments are admitted)."
                    )
                stages.append(
                    AIMNet2OutputStage(
                        name,
                        AIMNet2OutputKind.DIPOLE
                        if kind == "dipole"
                        else AIMNet2OutputKind.QUADRUPOLE,
                        center_coordinates=_flag(
                            options.get("center_coord", False), "center_coord"
                        ),
                        key_in="charges",
                        key_out=options.get("key_out", kind),
                    )
                )
            case "dispparam":
                options = _keys(
                    options, {"ref_c6", "ref_alpha", "ptfile", "key_in", "key_out"}, name
                )
                stages.append(
                    AIMNet2OutputStage(
                        name,
                        AIMNet2OutputKind.DISPERSION_PARAMETERS,
                        key_in=options.get("key_in", "disp_param"),
                        key_out=options.get("key_out", "disp_param"),
                    )
                )
            case "d3ts":
                stages.append(_ts_dispersion_stage(name, options, ts_pair_cutoff))
            case "lrcoulomb" | "dftd3":
                raise _refuse(
                    f"Embedded {kind} must first be externalized with "
                    "externalize_source_long_range."
                )
    return stages


def aimnet2_architecture_from_source(
    declaration: Mapping[str, Any],
    /,
    *,
    aev_cutoff: float,
    implemented_species: Sequence[int],
    charged_systems: bool = True,
    pair_radii: Mapping[str, float] | None = None,
    ts_pair_cutoff: float | None = None,
    neighbor_cutoff: float | None = None,
) -> AIMNet2Architecture:
    """Interpret one source model declaration; every key is admitted or refused.

    ``aev_cutoff`` is the executed ``aev.rc_s`` value and ``pair_radii`` the
    executed ``rc`` buffers of ``SRCoulomb``/``SRRep`` stages by name.
    ``ts_pair_cutoff`` declares the hard D3TS route radius the source takes
    from its dispersion neighbor list; ``neighbor_cutoff`` is the model neighbor
    radius over which ``SRRep`` sums (the source metadata ``cutoff``).
    """
    radii = dict(pair_radii or {})
    root = _keys(declaration, {"class", "kwargs"}, "model declaration")
    if root.get("class") not in _MODEL_CLASSES:
        raise _refuse(f"Model class {root.get('class')!r} is not aimnet.models.AIMNet2.")
    kwargs = _keys(root.get("kwargs"), _MODEL_KEYS, "AIMNet2 kwargs")
    for required in (
        "aev",
        "nfeature",
        "d2features",
        "ncomb_v",
        "hidden",
        "aim_size",
        "outputs",
    ):
        if required not in kwargs:
            raise _refuse(f"AIMNet2 kwargs lack {required!r}.")
    shifts_s, shifts_v = _admit_aev_shifts(kwargs["aev"])
    feature_count = _integer(kwargs["nfeature"], "nfeature")
    d2 = _flag(kwargs["d2features"], "d2features")
    combinations = kwargs["ncomb_v"]
    combinations = shifts_v if not combinations else _integer(combinations, "ncomb_v")
    channels = kwargs.get("num_charge_channels", 1)
    if channels not in (1, 2) or isinstance(channels, bool):
        raise _refuse("num_charge_channels must be 1 or 2.")
    parsed_hidden = _admit_hidden(kwargs["hidden"])
    aim_size = _integer(kwargs["aim_size"], "aim_size")
    stages = _output_stages(
        _output_entries(kwargs["outputs"]),
        aim_size,
        radii,
        ts_pair_cutoff,
        neighbor_cutoff,
    )
    try:
        return AIMNet2Architecture(
            feature_count=feature_count,
            d2_features=d2,
            shift_count=shifts_s,
            combination_count=combinations,
            hidden=parsed_hidden,
            aim_size=aim_size,
            charge_channels=channels,
            aev_cutoff=aev_cutoff,
            implemented_species=implemented_species,
            outputs=stages,
            charged_systems=charged_systems,
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, AIMNet2SourceRefusedError):
            raise
        raise _refuse(f"Source declaration is not admitted: {error}") from error


def _mlp(
    inventory: _Inventory,
    prefix: str,
    sizes: Sequence[int],
    last_linear: bool,
    bias: bool,
    activation: tuple[str, tuple[float, ...]],
    dtype: Any,
    /,
) -> AIMNet2MLP:
    """Source ``nn.Sequential``: Linear at even indices, activations between."""
    weights = []
    biases = []
    for layer in range(len(sizes) - 1):
        index = 2 * layer
        weights.append(
            inventory.array(f"{prefix}.{index}.weight", (sizes[layer + 1], sizes[layer]))
        )
        if bias:
            biases.append(inventory.array(f"{prefix}.{index}.bias", (sizes[layer + 1],)))
    return AIMNet2MLP(
        weights,
        biases if bias else None,
        last_linear=last_linear,
        activation=activation,
        dtype=dtype,
    )


def _constructor_reference(options: Mapping[str, Any], /) -> np.ndarray | None:
    """``DispParam.__init__``'s table from ``ref_c6``/``ref_alpha`` (87 rows)."""
    if options.get("ref_c6") is None and options.get("ref_alpha") is None:
        return None
    table = np.zeros((87, 2), dtype=np.float32)
    for column, key in enumerate(("ref_c6", "ref_alpha")):
        value = options.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            for species, item in value.items():
                index = int(species)
                if not 0 <= index < 87:
                    raise _refuse(f"DispParam {key} species {species} is out of range.")
                table[index, column] = _real(item, key)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            values = [_real(item, key) for item in value]
            if len(values) > 87:
                raise _refuse(f"DispParam {key} has more than 87 rows.")
            table[: len(values), column] = values
        else:
            raise _refuse(f"DispParam {key} must be a mapping or list.")
    table[0] = (0.0, 1.0)
    return table


def _radius(
    inventory: _Inventory,
    name: str,
    declared: Any,
    constructor_values: dict[str, Any],
    /,
) -> np.ndarray:
    if inventory.present(name):
        return inventory.array(name, ())
    value = np.asarray(np.float32(_real(declared, name)))
    constructor_values[name] = float(value)
    return value


def _scalar_basis(
    inventory: _Inventory, aev_declaration: Mapping[str, Any], /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executed ``aev`` ``(rc_s, eta_s, shifts_s)``; the vector basis is inert."""
    shift_count = aev_declaration.get("nshifts_s")
    if isinstance(shift_count, bool) or not isinstance(shift_count, int):
        raise _refuse("aev must declare an integer nshifts_s.")
    rc_s = inventory.array("aev.rc_s", ())
    eta_s = inventory.array("aev.eta_s", ())
    shifts_s = inventory.array("aev.shifts_s", (shift_count,))
    for name in ("aev.rc_v", "aev.eta_v", "aev.shifts_v"):
        if not inventory.present(name):
            raise _refuse(f"Source tensor {name!r} is missing.")
        inventory.mark_inert(name, "registered-vector-basis-unused-by-source-forward")
    return rc_s, eta_s, shifts_s


def _species_embedding(
    inventory: _Inventory, declared_species: Sequence[int] | None, /
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """``afv.weight`` with all-NaN unimplemented rows: ``(afv, all_nan, species)``."""
    afv_shape = inventory.shape("afv.weight")
    if len(afv_shape) != 2 or afv_shape[0] != AIMNET2_SPECIES_CAPACITY:
        raise _refuse("afv.weight must have 64 species rows.")
    afv = inventory.array("afv.weight", afv_shape, finite=False)
    nan_rows = np.isnan(afv)
    all_nan = np.all(nan_rows, axis=1)
    if np.any(nan_rows & ~all_nan[:, None]) or np.any(np.isinf(afv)):
        raise _refuse("afv.weight rows must be finite or entirely NaN (unimplemented).")
    if all_nan[0]:
        raise _refuse("The padding embedding row must be finite.")
    available = tuple(int(row) for row in np.nonzero(~all_nan)[0] if row > 0)
    if declared_species is None:
        species = available
    else:
        species = tuple(sorted({int(value) for value in declared_species}))
        missing = sorted(set(species) - set(available))
        if missing:
            raise _refuse(f"Declared species {missing} have unimplemented (NaN) rows.")
    return afv, all_nan, species


def _executed_radii(
    inventory: _Inventory,
    entries: Mapping[str, Mapping[str, Any]],
    constructor_values: dict[str, Any],
    /,
) -> dict[str, np.ndarray]:
    """Executed ``SRCoulomb``/``SRRep`` ``rc``: state buffer or constructor value."""
    radii: dict[str, np.ndarray] = {}
    for name, entry in entries.items():
        kind = _class_kind(entry["class"])
        options = entry.get("kwargs") or {}
        if kind == "srcoulomb":
            radii[name] = _radius(
                inventory,
                f"outputs.{name}.rc",
                options.get("rc", 4.6),
                constructor_values,
            )
        elif kind == "srrep":
            radii[name] = _radius(
                inventory,
                f"outputs.{name}.rc",
                options.get("rc", 5.2),
                constructor_values,
            )
    return radii


def _interaction_network(
    inventory: _Inventory, architecture: AIMNet2Architecture, dtype: Any, /
) -> tuple[AIMNet2ConvSV, AIMNet2ConvSV, tuple[AIMNet2MLP, ...]]:
    """Source ``conv_a``/``conv_q`` and the interaction-pass MLPs, in that order."""
    atom_conv = AIMNet2ConvSV(
        inventory.array(
            "conv_a.agh",
            (
                architecture.feature_count,
                architecture.shift_count,
                architecture.combination_count,
            ),
        ),
        d2_features=architecture.d2_features,
        dtype=dtype,
    )
    charge_conv = AIMNet2ConvSV(
        inventory.array(
            "conv_q.agh",
            (
                architecture.charge_channels,
                architecture.shift_count,
                architecture.combination_count,
            ),
        ),
        d2_features=False,
        dtype=dtype,
    )
    passes = tuple(
        _mlp(
            inventory,
            f"mlps.{index}",
            (inputs, *widths, outputs),
            last_linear,
            True,
            ("gelu", ()),
            dtype,
        )
        for index, (widths, (inputs, outputs, last_linear)) in enumerate(
            zip(architecture.pass_hidden, architecture.pass_shapes(), strict=True)
        )
    )
    return atom_conv, charge_conv, passes


def _atomic_shift_table(
    inventory: _Inventory,
    prefix: str,
    stage: AIMNet2OutputStage,
    all_nan: np.ndarray,
    reduction: Any,
    /,
) -> AIMNet2AtomicShifts | AIMNet2FrozenAtomicShifts:
    """Atomic shifts; rows of unimplemented species are zeroed, others finite."""
    shifts = inventory.array(
        f"{prefix}.shifts.weight", (stage.species_rows, 1), finite=False
    )[:, 0]
    rows = np.arange(stage.species_rows)
    unused = np.isin(rows, np.nonzero(all_nan)[0])
    if not np.all(np.isfinite(shifts[~unused])):
        raise _refuse("Atomic shifts of implemented species must be finite.")
    shifts = np.where(unused, 0.0, shifts).astype(shifts.dtype)
    table_type = AIMNet2FrozenAtomicShifts if stage.frozen else AIMNet2AtomicShifts
    return table_type(shifts, dtype=reduction)


def _repulsion_value(
    inventory: _Inventory,
    prefix: str,
    stage: AIMNet2OutputStage,
    radii: Mapping[str, np.ndarray],
    dtype: Any,
    /,
) -> AIMNet2Repulsion:
    """``SRRep`` parameter table with its executed radius and route cutoff."""
    if stage.cutoff_function is None or stage.pair_cutoff is None:
        raise _refuse(f"SRRep {stage.name!r} lost its declared cutoff.")
    name = f"{prefix}.params.weight"
    table = inventory.array(name, inventory.shape(name))
    return AIMNet2Repulsion(
        table,
        radii[stage.name],
        cutoff_function=stage.cutoff_function,
        route_cutoff=stage.pair_cutoff,
        dtype=dtype,
    )


def _moment_masses(
    inventory: _Inventory, prefix: str, stage: AIMNet2OutputStage, dtype: Any, /
) -> AIMNet2MassTable | None:
    """Moment ``mass`` buffer: executed for centered moments, otherwise inert."""
    name = f"{prefix}.mass"
    if stage.center_coordinates:
        return AIMNet2MassTable(inventory.array(name, (119,)), dtype=dtype)
    if not inventory.present(name):
        raise _refuse(f"Source tensor {name!r} is missing.")
    inventory.mark_inert(name, "mass-buffer-unused-without-center-coord")
    return None


def _dispersion_reference_table(
    inventory: _Inventory,
    prefix: str,
    options: Mapping[str, Any],
    constructor_values: dict[str, Any],
    superseded: dict[str, str],
    /,
) -> np.ndarray:
    """``DispParam`` reference table: state buffer, else constructor references.

    An absent buffer with an external ``ptfile`` or no reference is refused.
    """
    name = f"{prefix}.disp_param0"
    if inventory.present(name):
        shape = inventory.shape(name)
        if len(shape) != 2 or shape[1] != 2:
            raise _refuse(f"{name} must have shape (S, 2).")
        table = inventory.array(name, shape)
        for key in ("ptfile", "ref_c6", "ref_alpha"):
            if options.get(key) is not None:
                superseded[f"{prefix}.{key}"] = (
                    "constructor-value-superseded-by-state-buffer"
                )
        return table
    if options.get("ptfile") is not None:
        raise _refuse(
            f"{name} is absent and DispParam.ptfile names an external "
            "file outside the admitted source; admit its table as the "
            "state buffer."
        )
    constructed = _constructor_reference(options)
    if constructed is None:
        raise _refuse(
            f"{name} is absent and DispParam declares no reference; the "
            "source would execute an all-zero placeholder."
        )
    constructor_values[name] = "ref_c6/ref_alpha"
    return constructed


def _ts_dispersion_value(
    inventory: _Inventory, prefix: str, stage: AIMNet2OutputStage, dtype: Any, /
) -> TSDispersionPotential:
    """Embedded D3TS constants with the source ``r4r2`` table."""
    if stage.dispersion is None or stage.pair_cutoff is None:
        raise _refuse(f"D3TS {stage.name!r} lost its declared constants.")
    s6, s8, a1, a2 = stage.dispersion
    return TSDispersionPotential(
        a1=a1,
        a2=a2,
        s8=s8,
        s6=s6,
        pair_cutoff=stage.pair_cutoff,
        r4r2=inventory.array(f"{prefix}.r4r2", (119,)),
        dtype=dtype,
    )


def aimnet2_potential_from_source(
    scale: AtomisticScaleContract,
    declaration: Mapping[str, Any],
    tensors: Mapping[str, ArrayLike],
    /,
    *,
    declared_species: Sequence[int] | None = None,
    charged_systems: bool = True,
    ts_pair_cutoff: float | None = None,
    neighbor_cutoff: float | None = None,
    precision: AtomisticPrecisionPolicy | None = None,
) -> AIMNet2SourceReconstruction:
    """Reconstruct a native AIMNet2 potential from one exact source inventory.

    The declaration must already have embedded long-range modules externalized
    (``externalize_source_long_range``). ``declared_species`` is the source
    metadata ``implemented_species``; when absent the non-NaN ``afv.weight``
    rows define it (``extract_species``). ``charged_systems=False`` encodes the
    rxn family's neutral-only contract.
    """
    precision_ = AtomisticPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, AtomisticPrecisionPolicy):
        raise TypeError("precision must be an AtomisticPrecisionPolicy or None.")
    dtype = np.dtype(precision_.compute_dtype)
    reduction = np.dtype(precision_.reduction_dtype)
    inventory = _Inventory(tensors)
    root = _keys(declaration, {"class", "kwargs"}, "model declaration")
    kwargs = _keys(root.get("kwargs"), _MODEL_KEYS, "AIMNet2 kwargs")
    aev_declaration = _keys(kwargs.get("aev") or {}, _AEV_KEYS, "aev")
    rc_s, eta_s, shifts_s = _scalar_basis(inventory, aev_declaration)
    afv, all_nan, species = _species_embedding(inventory, declared_species)
    unimplemented = tuple(int(row) for row in np.nonzero(all_nan)[0])
    constructor_values: dict[str, Any] = {}
    superseded: dict[str, str] = {}
    entries = dict(_output_entries(kwargs.get("outputs")))
    radii = _executed_radii(inventory, entries, constructor_values)
    architecture = aimnet2_architecture_from_source(
        declaration,
        aev_cutoff=float(rc_s),
        implemented_species=species,
        charged_systems=charged_systems,
        pair_radii={name: float(value) for name, value in radii.items()},
        ts_pair_cutoff=ts_pair_cutoff,
        neighbor_cutoff=neighbor_cutoff,
    )
    embedding = np.where(all_nan[:, None], 0.0, afv).astype(afv.dtype)
    basis = AIMNet2RadialBasis(shifts_s, eta_s, rc_s, dtype=dtype)
    atom_conv, charge_conv, passes = _interaction_network(inventory, architecture, dtype)
    values: list[StageValue] = []
    for stage in architecture.outputs:
        prefix = f"outputs.{stage.name}"
        options = entries[stage.name].get("kwargs") or {}
        match stage.kind:
            case AIMNet2OutputKind.ENERGY_HEAD | AIMNet2OutputKind.DISPERSION_HEAD:
                values.append(
                    _mlp(
                        inventory,
                        f"{prefix}.mlp",
                        (architecture.aim_size, *stage.hidden, stage.output_count),
                        stage.last_linear,
                        stage.bias,
                        stage.activation,
                        dtype,
                    )
                )
            case AIMNet2OutputKind.ATOMIC_SHIFT:
                values.append(
                    _atomic_shift_table(inventory, prefix, stage, all_nan, reduction)
                )
            case AIMNet2OutputKind.ATOMIC_SUM:
                values.append(None)
            case AIMNet2OutputKind.SHORT_RANGE_COULOMB:
                values.append(AIMNet2ShortRangeRadius(radii[stage.name], dtype=dtype))
            case AIMNet2OutputKind.SHORT_RANGE_REPULSION:
                values.append(_repulsion_value(inventory, prefix, stage, radii, dtype))
            case AIMNet2OutputKind.DIPOLE | AIMNet2OutputKind.QUADRUPOLE:
                values.append(_moment_masses(inventory, prefix, stage, dtype))
            case AIMNet2OutputKind.DISPERSION_PARAMETERS:
                table = _dispersion_reference_table(
                    inventory, prefix, options, constructor_values, superseded
                )
                values.append(
                    TSDispersionReference(table, implemented_species=species, dtype=dtype)
                )
            case AIMNet2OutputKind.TS_DISPERSION:
                values.append(_ts_dispersion_value(inventory, prefix, stage, dtype))
    inventory.finish()
    declared_aev = {
        "rmin": aev_declaration.get("rmin", SOURCE_MINIMUM_SHIFT),
        "rc_s": aev_declaration.get("rc_s", 5.0),
        "rc_v": aev_declaration.get("rc_v"),
        "nshifts_v": aev_declaration.get("nshifts_v"),
        "vector_basis_executed": False,
    }
    try:
        potential = AIMNet2Potential(
            scale,
            architecture,
            precision=precision_,
            embedding=embedding,
            basis=basis,
            atom_conv=atom_conv,
            charge_conv=charge_conv,
            passes=passes,
            stage_values=tuple(values),
        )
    except (TypeError, ValueError) as error:
        raise _refuse(f"Source components are not admitted: {error}") from error
    return AIMNet2SourceReconstruction(
        potential=potential,
        architecture=architecture,
        inert_tensors=MappingProxyType(dict(inventory.inert)),
        unimplemented_rows=unimplemented,
        constructor_values=MappingProxyType(constructor_values),
        superseded_declarations=MappingProxyType(superseded),
        declared_aev=MappingProxyType(declared_aev),
    )


__all__ = [
    "AIMNet2CalculatorComposition",
    "AIMNet2ExternalizedTerms",
    "aimnet2_architecture_from_source",
    "aimnet2_calculator_composition",
    "aimnet2_potential_from_source",
    "AIMNet2SourceReconstruction",
    "AIMNet2SourceRefusedError",
    "externalize_source_long_range",
    "SOURCE_COULOMB_CONSTANT",
    "SOURCE_EXTERNAL_D3_CUTOFF",
    "SOURCE_EXTERNAL_D3_SMOOTHING_FRACTION",
    "SOURCE_RXN_D3_PARAMETERS",
]
