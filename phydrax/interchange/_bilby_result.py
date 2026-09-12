#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule


if TYPE_CHECKING:
    from ..integration import WeightedSampleTarget
    from ..uq._population import EventPosterior
from ._resource import read_bounded_resource, ResourceLimits


_FORBIDDEN_POSTERIOR_MARKERS = {
    "__class__",
    "__function__",
    "__lal_dict__",
    "__module__",
    "__numpy_random_generator__",
    "__prior__",
    "__prior_dict__",
}


def _unique_object(pairs: list[tuple[str, Any]], /) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Bilby JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _finite_optional(value: Any, /) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("Bilby evidence must be finite when supplied.")
    return scalar


def _validate_structure(value: Any, limits: ResourceLimits, /) -> None:
    pending = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > limits.max_nodes or depth > limits.max_depth:
            raise ValueError("Bilby JSON exceeds structural resource limits.")
        if isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)


def _dataframe_content(value: Any, /) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Bilby posterior must be a JSON mapping.")
    if value.get("__dataframe__") is True:
        content = value.get("content")
    else:
        content = value
    if not isinstance(content, Mapping) or not content:
        raise ValueError("Bilby posterior contains no columns.")
    if _FORBIDDEN_POSTERIOR_MARKERS.intersection(content):
        raise ValueError("Bilby posterior contains an executable object marker.")
    return content


def _column(value: Any, name: str, /, *, finite: bool = True) -> np.ndarray:
    if isinstance(value, Mapping) and value.get("__array__") is True:
        value = value.get("content")
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.size == 0
        or array.dtype.hasobject
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(f"Bilby posterior column {name!r} must be a numeric vector.")
    if finite and np.any(~np.isfinite(array)):
        raise ValueError(f"Bilby posterior column {name!r} must be finite.")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


class ImportedBilbyResult(StrictModule):
    """Current-format Bilby JSON posterior decoded without importing Bilby objects."""

    parameters: frozendict[str, Array]
    log_weights: Array
    sampling_log_prior: Array | None
    log_likelihood: Array | None
    log_evidence: Array
    source_manifest_id: str = eqx.field(static=True)
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    sampler: str = eqx.field(static=True)
    bilby_version: str = eqx.field(static=True)
    label: str = eqx.field(static=True)
    evidence_kind: str = eqx.field(static=True)
    has_evidence: bool = eqx.field(static=True)
    import_id: str = eqx.field(static=True)

    @property
    def num_samples(self) -> int:
        return int(self.log_weights.shape[0])

    def posterior_measure(self) -> WeightedSampleTarget:
        from ..integration import WeightedSampleTarget

        return WeightedSampleTarget(
            self.parameters,
            self.log_weights,
            normalized=True,
            independent=False,
            sample_axes=0,
            provenance=f"bilby-json:{self.import_id}",
        )

    def to_event_posterior(
        self,
        /,
        *,
        event_id: str,
        parameterization_id: str,
        likelihood_id: str,
        provider_id: str,
        approximation: str = "external-import",
        sampling_log_prior: ArrayLike | None = None,
    ) -> EventPosterior:
        from ..uq._particle import effective_sample_size
        from ..uq._population import EventPosterior

        prior = (
            self.sampling_log_prior
            if sampling_log_prior is None
            else jnp.asarray(sampling_log_prior)
        )
        if prior is None:
            raise ValueError(
                "Event posterior import requires sample-level original prior values."
            )
        source_ess = effective_sample_size(self.log_weights)
        return EventPosterior(
            self.posterior_measure(),
            prior,
            event_id=event_id,
            parameterization_id=parameterization_id,
            likelihood_id=likelihood_id,
            provider_id=provider_id,
            inference_method=f"bilby:{self.sampler}",
            approximation=approximation,
            source_effective_sample_size=source_ess,
            log_evidence=self.log_evidence if self.has_evidence else None,
            evidence_kind=self.evidence_kind,
        )


def read_bilby_result_json(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
) -> ImportedBilbyResult:
    source = Path(path)
    if source.suffix.lower() != ".json":
        raise ValueError("Only current plain Bilby JSON results are supported.")
    resource = read_bounded_resource(source, trusted_root=trusted_root, limits=limits)
    payload = json.loads(resource.data.decode("utf-8"), object_pairs_hook=_unique_object)
    _validate_structure(payload, limits)
    if not isinstance(payload, Mapping):
        raise TypeError("Bilby result root must be a JSON object.")
    search_keys = payload.get("search_parameter_keys")
    if (
        not isinstance(search_keys, list)
        or not search_keys
        or any(not isinstance(value, str) or not value for value in search_keys)
        or len(set(search_keys)) != len(search_keys)
    ):
        raise ValueError(
            "Bilby result must declare distinct non-empty search_parameter_keys."
        )
    names = tuple(search_keys)
    content = _dataframe_content(payload.get("posterior"))
    if any(name not in content for name in names):
        raise ValueError("Bilby posterior is missing a sampled parameter column.")
    columns = {name: _column(content[name], name) for name in names}
    lengths = {int(value.size) for value in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Bilby posterior columns have inconsistent lengths.")
    count = lengths.pop()
    if "weights" in content:
        weights = _column(content["weights"], "weights")
        if weights.size != count or np.any(weights < 0.0) or not np.any(weights > 0.0):
            raise ValueError("Bilby posterior weights are invalid.")
        log_weights = np.full(weights.shape, -np.inf, dtype=float)
        positive = weights > 0.0
        log_weights[positive] = np.log(weights[positive])
    else:
        log_weights = np.zeros((count,), dtype=float)
    log_weights = log_weights - np.logaddexp.reduce(log_weights)
    log_prior = (
        None if "log_prior" not in content else _column(content["log_prior"], "log_prior")
    )
    log_likelihood = (
        None
        if "log_likelihood" not in content
        else _column(content["log_likelihood"], "log_likelihood")
    )
    for name, value in (("log_prior", log_prior), ("log_likelihood", log_likelihood)):
        if value is not None and value.size != count:
            raise ValueError(f"Bilby {name} column has inconsistent length.")
    evidence_value = _finite_optional(payload.get("log_evidence"))
    use_ratio = payload.get("use_ratio", False)
    if not isinstance(use_ratio, bool):
        raise TypeError("Bilby use_ratio metadata must be Boolean.")
    evidence_kind = (
        "omitted-constant"
        if evidence_value is None
        else "noise-relative"
        if use_ratio
        else "absolute"
    )
    sampler = str(payload.get("sampler", "unknown")).strip() or "unknown"
    version = str(payload.get("version", "unknown")).strip() or "unknown"
    label = str(payload.get("label", "bilby-result")).strip() or "bilby-result"
    parameters = frozendict({name: jnp.asarray(value) for name, value in columns.items()})
    import_id = canonical_fingerprint(
        {
            "kind": "imported-bilby-json-result",
            "resource": resource.manifest.manifest_id,
            "parameters": list(names),
            "sampler": sampler,
            "version": version,
            "label": label,
            "content": array_tree_fingerprint(
                {
                    "parameters": parameters,
                    "log_weights": log_weights,
                    "log_prior": log_prior,
                    "log_likelihood": log_likelihood,
                }
            )["sha256"],
        }
    )
    return ImportedBilbyResult(
        parameters,
        jnp.asarray(log_weights),
        None if log_prior is None else jnp.asarray(log_prior),
        None if log_likelihood is None else jnp.asarray(log_likelihood),
        jnp.asarray(0.0 if evidence_value is None else evidence_value),
        resource.manifest.manifest_id,
        names,
        sampler,
        version,
        label,
        evidence_kind,
        evidence_value is not None,
        import_id,
    )


__all__ = ["ImportedBilbyResult", "read_bilby_result_json"]
