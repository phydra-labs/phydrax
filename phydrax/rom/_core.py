#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


ParameterRecord = tuple[tuple[str, float], ...]
FloatArray = NDArray[np.floating]


def _canonical_json(value: object, /) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: object, /) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_array(
    value: ArrayLike | None,
    /,
    *,
    name: str,
    ndim: int | None = None,
    allow_empty: bool = False,
) -> FloatArray | None:
    if value is None:
        return None
    array = np.array(value, copy=True)
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be a numeric array.")
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued.")
    if not np.issubdtype(array.dtype, np.inexact):
        array = array.astype(float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}.")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    array.setflags(write=False)
    return array


def _array_record(value: FloatArray | None, /) -> dict[str, object] | None:
    if value is None:
        return None
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _parameter_record(value: Mapping[str, float] | ParameterRecord, /) -> ParameterRecord:
    items = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(sorted((str(name).strip(), float(item)) for name, item in items))
    names = tuple(name for name, _ in normalized)
    values = tuple(item for _, item in normalized)
    if not normalized or any(not name for name in names):
        raise ValueError("ROM parameters must be a non-empty named mapping.")
    if len(set(names)) != len(names):
        raise ValueError("ROM parameter names must be unique.")
    if not np.all(np.isfinite(values)):
        raise ValueError("ROM parameters must be finite.")
    return normalized


def parameters_mapping(value: ParameterRecord, /) -> Mapping[str, float]:
    """Return one immutable parameter mapping."""
    return MappingProxyType(dict(value))


class CorpusPartition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class GeometryRegistration:
    """Explicit linear pullback/pushforward between physical and reference states."""

    geometry_id: str
    reference_geometry_id: str
    pullback: FloatArray
    pushforward: FloatArray
    registration_id: str = field(init=False)

    def __post_init__(self) -> None:
        geometry_id = self.geometry_id.strip()
        reference_id = self.reference_geometry_id.strip()
        pullback = _freeze_array(self.pullback, name="pullback", ndim=2)
        pushforward = _freeze_array(self.pushforward, name="pushforward", ndim=2)
        assert pullback is not None and pushforward is not None
        if not geometry_id or not reference_id:
            raise ValueError("Geometry registration IDs must be non-empty.")
        if (
            pullback.shape[1] != pushforward.shape[0]
            or pullback.shape[0] != pushforward.shape[1]
        ):
            raise ValueError(
                "Geometry pullback and pushforward dimensions are incompatible."
            )
        identity = pullback @ pushforward
        if not np.allclose(identity, np.eye(identity.shape[0]), rtol=1e-8, atol=1e-10):
            raise ValueError(
                "Geometry pullback/pushforward must preserve reference states."
            )
        object.__setattr__(self, "geometry_id", geometry_id)
        object.__setattr__(self, "reference_geometry_id", reference_id)
        object.__setattr__(self, "pullback", pullback)
        object.__setattr__(self, "pushforward", pushforward)
        object.__setattr__(
            self,
            "registration_id",
            _fingerprint(
                {
                    "kind": "rom-geometry-registration",
                    "geometry_id": geometry_id,
                    "reference_geometry_id": reference_id,
                    "pullback": _array_record(pullback),
                    "pushforward": _array_record(pushforward),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ROMCaseSpec:
    """One requested truth case with explicit parameter and geometry identity."""

    case_id: str
    parameters: ParameterRecord
    geometry_id: str = "fixed"

    def __post_init__(self) -> None:
        case_id = self.case_id.strip()
        geometry_id = self.geometry_id.strip()
        if not case_id or not geometry_id:
            raise ValueError("ROM case and geometry IDs must be non-empty.")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "geometry_id", geometry_id)
        object.__setattr__(self, "parameters", _parameter_record(self.parameters))


@dataclass(frozen=True, slots=True)
class TruthSample:
    """Provider-neutral truth output consumed by corpus construction and fallback."""

    state: FloatArray
    truth_artifact_id: str
    operator: FloatArray | None = None
    rhs: FloatArray | None = None
    qoi_vector: FloatArray | None = None
    qoi: float | None = None
    dual_norm_inverse: FloatArray | None = None
    stability_lower_bound: float | None = None
    trajectory: FloatArray | None = None
    time_step: float | None = None
    nonlinear_snapshots: FloatArray | None = None
    residual_snapshots: FloatArray | None = None
    element_contributions: FloatArray | None = None
    low_fidelity_qoi: float | None = None
    fidelity_level: int | None = None
    registration: GeometryRegistration | None = None

    def __post_init__(self) -> None:
        state = _freeze_array(self.state, name="state", ndim=1)
        operator = _freeze_array(self.operator, name="operator", ndim=2)
        rhs = _freeze_array(self.rhs, name="rhs", ndim=1)
        qoi_vector = _freeze_array(self.qoi_vector, name="qoi_vector", ndim=1)
        dual = _freeze_array(self.dual_norm_inverse, name="dual_norm_inverse", ndim=2)
        trajectory = _freeze_array(self.trajectory, name="trajectory", ndim=2)
        nonlinear = _freeze_array(
            self.nonlinear_snapshots, name="nonlinear_snapshots", ndim=2
        )
        residual = _freeze_array(
            self.residual_snapshots, name="residual_snapshots", ndim=2
        )
        contributions = _freeze_array(
            self.element_contributions, name="element_contributions", ndim=2
        )
        assert state is not None
        state_size = state.size
        artifact_id = self.truth_artifact_id.strip()
        if not artifact_id:
            raise ValueError("Truth artifact IDs must be non-empty.")
        if operator is not None and operator.shape != (state_size, state_size):
            raise ValueError("Truth operators must be square on the state space.")
        if rhs is not None and rhs.shape != state.shape:
            raise ValueError("Truth right-hand sides must match the state shape.")
        if qoi_vector is not None and qoi_vector.shape != state.shape:
            raise ValueError("Truth QoI vectors must match the state shape.")
        if dual is not None:
            if dual.shape != (state_size, state_size):
                raise ValueError(
                    "Dual-norm inverse matrices must act on the state space."
                )
            if not np.allclose(dual, dual.T, rtol=1e-9, atol=1e-11):
                raise ValueError("Dual-norm inverse matrices must be symmetric.")
            if np.min(np.linalg.eigvalsh(dual)) <= 0.0:
                raise ValueError("Dual-norm inverse matrices must be positive definite.")
        if trajectory is not None and trajectory.shape[1] != state_size:
            raise ValueError("Trajectory snapshots must use the truth state dimension.")
        for name, snapshots in (
            ("nonlinear_snapshots", nonlinear),
            ("residual_snapshots", residual),
        ):
            if snapshots is not None and snapshots.shape[1] != state_size:
                raise ValueError(f"{name} must use the truth state dimension.")
        qoi = None if self.qoi is None else float(self.qoi)
        low_qoi = None if self.low_fidelity_qoi is None else float(self.low_fidelity_qoi)
        if qoi is not None and not np.isfinite(qoi):
            raise ValueError("Truth QoIs must be finite.")
        if low_qoi is not None and not np.isfinite(low_qoi):
            raise ValueError("Low-fidelity QoIs must be finite.")
        alpha = (
            None
            if self.stability_lower_bound is None
            else float(self.stability_lower_bound)
        )
        if alpha is not None and (not np.isfinite(alpha) or alpha <= 0.0):
            raise ValueError("Stability lower bounds must be finite and positive.")
        time_step = None if self.time_step is None else float(self.time_step)
        if time_step is not None and (not np.isfinite(time_step) or time_step <= 0.0):
            raise ValueError("Truth time steps must be finite and positive.")
        level = None if self.fidelity_level is None else int(self.fidelity_level)
        if level is not None and level < 0:
            raise ValueError("Fidelity levels must be nonnegative.")
        for name, value in (
            ("state", state),
            ("operator", operator),
            ("rhs", rhs),
            ("qoi_vector", qoi_vector),
            ("dual_norm_inverse", dual),
            ("trajectory", trajectory),
            ("nonlinear_snapshots", nonlinear),
            ("residual_snapshots", residual),
            ("element_contributions", contributions),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "truth_artifact_id", artifact_id)
        object.__setattr__(self, "qoi", qoi)
        object.__setattr__(self, "low_fidelity_qoi", low_qoi)
        object.__setattr__(self, "stability_lower_bound", alpha)
        object.__setattr__(self, "time_step", time_step)
        object.__setattr__(self, "fidelity_level", level)


@runtime_checkable
class TruthModel(Protocol):
    """Adapter to an existing truth stack; ROM never owns the truth solver."""

    def __call__(self, case: ROMCaseSpec, /) -> TruthSample: ...


@dataclass(frozen=True, slots=True)
class ROMCase:
    """One immutable truth case retained by a ROM corpus."""

    spec: ROMCaseSpec
    sample: TruthSample
    case_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.sample.registration is not None and (
            self.sample.registration.geometry_id != self.spec.geometry_id
        ):
            raise ValueError("Case geometry and registration geometry must match.")
        object.__setattr__(
            self,
            "case_digest",
            _fingerprint(
                {
                    "kind": "rom-case",
                    "case_id": self.spec.case_id,
                    "parameters": list(self.spec.parameters),
                    "geometry_id": self.spec.geometry_id,
                    "truth_artifact_id": self.sample.truth_artifact_id,
                    "state": _array_record(self.sample.state),
                    "operator": _array_record(self.sample.operator),
                    "rhs": _array_record(self.sample.rhs),
                    "qoi_vector": _array_record(self.sample.qoi_vector),
                    "qoi": self.sample.qoi,
                    "dual_norm_inverse": _array_record(self.sample.dual_norm_inverse),
                    "stability_lower_bound": self.sample.stability_lower_bound,
                    "trajectory": _array_record(self.sample.trajectory),
                    "time_step": self.sample.time_step,
                    "nonlinear_snapshots": _array_record(self.sample.nonlinear_snapshots),
                    "residual_snapshots": _array_record(self.sample.residual_snapshots),
                    "element_contributions": _array_record(
                        self.sample.element_contributions
                    ),
                    "low_fidelity_qoi": self.sample.low_fidelity_qoi,
                    "fidelity_level": self.sample.fidelity_level,
                    "registration_id": (
                        None
                        if self.sample.registration is None
                        else self.sample.registration.registration_id
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CorpusSplit:
    """Disjoint, exhaustive case partitions with a stable split identity."""

    train: tuple[str, ...]
    validation: tuple[str, ...] = ()
    test: tuple[str, ...] = ()
    split_id: str = field(init=False)

    def __post_init__(self) -> None:
        train = tuple(str(value).strip() for value in self.train)
        validation = tuple(str(value).strip() for value in self.validation)
        test = tuple(str(value).strip() for value in self.test)
        all_ids = (*train, *validation, *test)
        if not train or any(not value for value in all_ids):
            raise ValueError("Corpus splits require non-empty training case IDs.")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("Corpus split partitions must be disjoint.")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "test", test)
        object.__setattr__(
            self,
            "split_id",
            _fingerprint(
                {
                    "kind": "rom-corpus-split",
                    "train": list(train),
                    "validation": list(validation),
                    "test": list(test),
                }
            ),
        )

    def partition(self, case_id: str, /) -> CorpusPartition:
        if case_id in self.train:
            return CorpusPartition.TRAIN
        if case_id in self.validation:
            return CorpusPartition.VALIDATION
        if case_id in self.test:
            return CorpusPartition.TEST
        raise KeyError(f"Case {case_id!r} is not in this corpus split.")


@dataclass(frozen=True, slots=True)
class ValidityRegion:
    """Closed parameter box and explicit registered geometry support."""

    parameter_names: tuple[str, ...]
    lower: FloatArray
    upper: FloatArray
    geometry_ids: tuple[str, ...] = ("fixed",)
    validity_id: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(str(name).strip() for name in self.parameter_names)
        lower = _freeze_array(self.lower, name="validity lower", ndim=1)
        upper = _freeze_array(self.upper, name="validity upper", ndim=1)
        geometry_ids = tuple(str(value).strip() for value in self.geometry_ids)
        assert lower is not None and upper is not None
        if (
            not names
            or len(set(names)) != len(names)
            or any(not name for name in names)
            or lower.shape != (len(names),)
            or upper.shape != lower.shape
            or np.any(lower > upper)
        ):
            raise ValueError("Validity parameter bounds are invalid.")
        if not geometry_ids or any(not value for value in geometry_ids):
            raise ValueError("Validity geometry IDs must be non-empty.")
        if len(set(geometry_ids)) != len(geometry_ids):
            raise ValueError("Validity geometry IDs must be unique.")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "geometry_ids", geometry_ids)
        object.__setattr__(
            self,
            "validity_id",
            _fingerprint(
                {
                    "kind": "rom-validity-region",
                    "parameter_names": list(names),
                    "lower": lower.tolist(),
                    "upper": upper.tolist(),
                    "geometry_ids": list(geometry_ids),
                }
            ),
        )

    @classmethod
    def enclosing(cls, cases: Sequence[ROMCase], /) -> ValidityRegion:
        if not cases:
            raise ValueError("A validity region requires at least one case.")
        names = tuple(name for name, _ in cases[0].spec.parameters)
        if any(
            tuple(name for name, _ in case.spec.parameters) != names for case in cases
        ):
            raise ValueError("All corpus cases must use the same named parameters.")
        values = np.asarray(
            [[value for _, value in case.spec.parameters] for case in cases], dtype=float
        )
        geometry_ids = tuple(dict.fromkeys(case.spec.geometry_id for case in cases))
        return cls(names, values.min(axis=0), values.max(axis=0), geometry_ids)

    def contains(
        self,
        parameters: Mapping[str, float] | ParameterRecord,
        geometry_id: str = "fixed",
        /,
    ) -> bool:
        record = _parameter_record(parameters)
        if tuple(name for name, _ in record) != self.parameter_names:
            return False
        values = np.asarray([value for _, value in record])
        return bool(
            geometry_id in self.geometry_ids
            and np.all(values >= self.lower)
            and np.all(values <= self.upper)
        )


@dataclass(frozen=True, slots=True)
class CorpusCaseManifest:
    case_id: str
    partition: CorpusPartition
    parameters: ParameterRecord
    geometry_id: str
    truth_artifact_id: str
    case_digest: str


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    corpus_id: str
    truth_model_id: str
    truth_model_revision: str
    split_id: str
    validity_id: str
    cases: tuple[CorpusCaseManifest, ...]


@dataclass(frozen=True, slots=True)
class ROMCorpus:
    """Immutable truth data and its content-sensitive manifest."""

    manifest: CorpusManifest
    split: CorpusSplit
    validity: ValidityRegion
    cases: tuple[ROMCase, ...]

    def cases_in(self, partition: CorpusPartition, /) -> tuple[ROMCase, ...]:
        return tuple(
            case
            for case in self.cases
            if self.split.partition(case.spec.case_id) is partition
        )


def create_corpus(
    case_specs: Iterable[ROMCaseSpec],
    truth_model: TruthModel | Callable[[ROMCaseSpec], TruthSample],
    /,
    *,
    truth_model_id: str,
    truth_model_revision: str,
    split: CorpusSplit,
    validity: ValidityRegion | None = None,
) -> ROMCorpus:
    """Evaluate declared cases through an existing truth adapter and freeze a corpus."""
    if not callable(truth_model):
        raise TypeError("truth_model must implement the TruthModel call contract.")
    model_id = truth_model_id.strip()
    revision = truth_model_revision.strip()
    if not model_id or not revision:
        raise ValueError("Truth model identity and revision must be non-empty.")
    specs = tuple(case_specs)
    if not specs or any(not isinstance(spec, ROMCaseSpec) for spec in specs):
        raise TypeError("case_specs must contain ROMCaseSpec values.")
    if len({spec.case_id for spec in specs}) != len(specs):
        raise ValueError("ROM case IDs must be unique.")
    expected_ids = set((*split.train, *split.validation, *split.test))
    if {spec.case_id for spec in specs} != expected_ids:
        raise ValueError("Corpus split IDs must exactly cover the declared cases.")
    cases_list: list[ROMCase] = []
    for spec in specs:
        sample = truth_model(spec)
        if not isinstance(sample, TruthSample):
            raise TypeError("Truth adapters must return TruthSample values.")
        cases_list.append(ROMCase(spec, sample))
    cases = tuple(cases_list)
    region = ValidityRegion.enclosing(cases) if validity is None else validity
    if any(
        not region.contains(case.spec.parameters, case.spec.geometry_id) for case in cases
    ):
        raise ValueError("The declared validity region must contain every corpus case.")
    case_manifests = tuple(
        CorpusCaseManifest(
            case.spec.case_id,
            split.partition(case.spec.case_id),
            case.spec.parameters,
            case.spec.geometry_id,
            case.sample.truth_artifact_id,
            case.case_digest,
        )
        for case in cases
    )
    corpus_id = _fingerprint(
        {
            "kind": "rom-corpus",
            "truth_model_id": model_id,
            "truth_model_revision": revision,
            "split_id": split.split_id,
            "validity_id": region.validity_id,
            "cases": [
                {
                    "case_id": item.case_id,
                    "partition": item.partition.value,
                    "parameters": list(item.parameters),
                    "geometry_id": item.geometry_id,
                    "truth_artifact_id": item.truth_artifact_id,
                    "case_digest": item.case_digest,
                }
                for item in case_manifests
            ],
        }
    )
    manifest = CorpusManifest(
        corpus_id,
        model_id,
        revision,
        split.split_id,
        region.validity_id,
        case_manifests,
    )
    return ROMCorpus(manifest, split, region, cases)


__all__ = [
    "CorpusCaseManifest",
    "CorpusManifest",
    "CorpusPartition",
    "CorpusSplit",
    "GeometryRegistration",
    "ParameterRecord",
    "ROMCase",
    "ROMCaseSpec",
    "ROMCorpus",
    "TruthModel",
    "TruthSample",
    "ValidityRegion",
    "create_corpus",
    "parameters_mapping",
]
