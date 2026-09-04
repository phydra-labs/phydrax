#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import DifferentiationContract, ScientificArtifactEnvelope
from ._linear_theory import (
    CosmologyModelRequest,
    CosmologyModelResult,
    SubprocessCosmologyModelBackend,
)
from ._products import (
    cosmology_product_content_id,
    CosmologyProductProvenance,
    LinearTransferDescriptor,
    LinearTransferTable,
    MatterPowerDescriptor,
    MatterPowerTable,
    ThermodynamicsHistory,
)
from ._scales import CosmologyScaleContract


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class LinearTheoryPhysicsPolicy(StrictModule, NonTrainableState):
    initial_conditions: str = eqx.field(static=True)
    dark_energy_perturbations: str = eqx.field(static=True)
    recombination: str = eqx.field(static=True)
    reionization: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_conditions: str = "scalar-adiabatic",
        dark_energy_perturbations: str = "ppf",
        recombination: str = "backend-default-qualified",
        reionization: str = "tanh-optical-depth",
    ):
        values = tuple(
            str(value).strip()
            for value in (
                initial_conditions,
                dark_energy_perturbations,
                recombination,
                reionization,
            )
        )
        if any(not value for value in values):
            raise ValueError("Linear-theory physics-policy fields must be non-empty.")
        if values[0] != "scalar-adiabatic":
            raise ValueError(
                "First precision-backend scope supports scalar adiabatic ICs."
            )
        (
            self.initial_conditions,
            self.dark_energy_perturbations,
            self.recombination,
            self.reionization,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "linear-theory-physics-policy",
                "initial_conditions": values[0],
                "dark_energy_perturbations": values[1],
                "recombination": values[2],
                "reionization": values[3],
            }
        )


class LinearTheoryOutputPolicy(StrictModule, NonTrainableState):
    transfer_fields: tuple[str, ...] = eqx.field(static=True)
    gauge: str = eqx.field(static=True)
    power_field: str = eqx.field(static=True)
    include_thermodynamics: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer_fields: tuple[str, ...],
        /,
        *,
        gauge: str,
        power_field: str,
        include_thermodynamics: bool = False,
    ):
        fields = tuple(str(value).strip() for value in transfer_fields)
        gauge_ = str(gauge).strip()
        power_ = str(power_field).strip()
        if not fields or any(not value for value in fields) or not gauge_ or not power_:
            raise ValueError("Linear-theory output policy is invalid.")
        self.transfer_fields = fields
        self.gauge = gauge_
        self.power_field = power_
        self.include_thermodynamics = bool(include_thermodynamics)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "linear-theory-output-policy",
                "transfer_fields": list(fields),
                "gauge": gauge_,
                "power_field": power_,
                "include_thermodynamics": bool(include_thermodynamics),
            }
        )


class BackendBuildManifest(StrictModule, NonTrainableState):
    backend: str = eqx.field(static=True)
    release: str = eqx.field(static=True)
    application: str = eqx.field(static=True)
    arguments: tuple[str, ...] = eqx.field(static=True)
    binary_digest: str = eqx.field(static=True)
    build_options: tuple[str, ...] = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    build_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        backend: str,
        release: str,
        application: str,
        build_options: tuple[str, ...] = (),
        arguments: tuple[str, ...] = ("{request}", "{output}"),
        license_id: str,
    ):
        values = tuple(
            str(value).strip() for value in (backend, release, application, license_id)
        )
        options = tuple(str(value).strip() for value in build_options)
        arguments_ = tuple(str(value) for value in arguments)
        if (
            any(not value for value in values)
            or any(not value for value in options)
            or not any("{request}" in value for value in arguments_)
            or not any("{output}" in value for value in arguments_)
        ):
            raise ValueError("Backend build-manifest fields/arguments are invalid.")
        path = Path(values[2]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(
                "Backend build-manifest application must be an existing file."
            )
        digest = _sha256(path)
        self.backend, self.release, self.application, self.license_id = values
        self.binary_digest = digest
        self.build_options = options
        self.arguments = arguments_
        self.build_id = canonical_fingerprint(
            {
                "kind": "precision-backend-build",
                "backend": values[0],
                "release": values[1],
                "application": str(path),
                "binary_digest": digest,
                "build_options": list(options),
                "arguments": list(arguments_),
                "license_id": values[3],
            }
        )


class LinearTheoryResourcePolicy(StrictModule, NonTrainableState):
    timeout_seconds: float = eqx.field(static=True)
    backend_threads: int = eqx.field(static=True)
    output_byte_cap: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        timeout_seconds: float = 600.0,
        backend_threads: int = 1,
        output_byte_cap: int = 1_000_000_000,
    ):
        timeout = float(timeout_seconds)
        threads = int(backend_threads)
        cap = int(output_byte_cap)
        if not np.isfinite(timeout) or timeout <= 0.0 or threads <= 0 or cap <= 0:
            raise ValueError("Linear-theory resource policy is invalid.")
        self.timeout_seconds = timeout
        self.backend_threads = threads
        self.output_byte_cap = cap
        self.policy_id = canonical_fingerprint(
            {
                "kind": "linear-theory-resource-policy",
                "timeout_seconds": timeout,
                "backend_threads": threads,
                "output_byte_cap": cap,
            }
        )


class PrecisionLinearTheoryResult(StrictModule):
    products: CosmologyModelResult
    artifact: ScientificArtifactEnvelope
    cache_hit: bool = eqx.field(static=True)


def _provenance(
    request: CosmologyModelRequest,
    build: BackendBuildManifest,
    resources: LinearTheoryResourcePolicy,
) -> CosmologyProductProvenance:
    return CosmologyProductProvenance(
        producer=build.backend,
        producer_version=build.release,
        model_form_id=request.model_form_id,
        request_id=request.request_id,
        numerical_policy_id=resources.policy_id,
        physics_policy_id=f"{build.backend}-canonical-linear-theory",
        scale_id=request.scale.scale_id,
        source_kind="external",
        differentiation=DifferentiationContract.constant(),
    )


def _save_products(path: Path, products: CosmologyModelResult, /) -> None:
    thermo = products.thermodynamics
    product_scales = [products.transfer.scale, products.power.scale]
    if thermo is not None:
        product_scales.append(thermo.scale)
    if any(scale.scale_id != product_scales[0].scale_id for scale in product_scales[1:]):
        raise ValueError("Linear-theory products have inconsistent scale identities.")
    arrays = {
        "scale_json": np.asarray(json.dumps(product_scales[0].to_dict(), sort_keys=True)),
        "scale_factors": np.asarray(products.transfer.scale_factors),
        "wavenumbers": np.asarray(products.transfer.wavenumbers),
        "transfer_values": np.asarray(products.transfer.transfer_values),
        "power_values": np.asarray(products.power.power_values),
        "has_thermodynamics": np.asarray(thermo is not None, dtype=np.int8),
    }
    if thermo is not None:
        arrays.update(
            {
                "ionization_fraction": np.asarray(thermo.ionization_fraction),
                "baryon_temperature": np.asarray(thermo.baryon_temperature),
                "opacity_derivative": np.asarray(thermo.opacity_derivative),
                "visibility": np.asarray(thermo.visibility),
            }
        )
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, allow_pickle=False, **arrays)
    temporary.replace(path)


def _load_products(
    path: Path,
    request: CosmologyModelRequest,
    build: BackendBuildManifest,
    resources: LinearTheoryResourcePolicy,
) -> CosmologyModelResult:
    with np.load(path, allow_pickle=False) as arrays:
        scale_payload = json.loads(str(np.asarray(arrays["scale_json"]).item()))
        if not isinstance(scale_payload, dict):
            raise ValueError("Cached linear-theory scale metadata must be a mapping.")
        scale = CosmologyScaleContract.from_dict(scale_payload)
        if scale.scale_id != request.scale.scale_id:
            raise ValueError(
                "Cached linear-theory scale does not match the current request."
            )
        scales = jnp.asarray(arrays["scale_factors"])
        wavenumbers = jnp.asarray(arrays["wavenumbers"])
        transfer_values = jnp.asarray(arrays["transfer_values"])
        power_values = jnp.asarray(arrays["power_values"])
        has_thermodynamics = bool(np.asarray(arrays["has_thermodynamics"]).item())
        thermo_values = (
            tuple(
                jnp.asarray(arrays[name])
                for name in (
                    "ionization_fraction",
                    "baryon_temperature",
                    "opacity_derivative",
                    "visibility",
                )
            )
            if has_thermodynamics
            else None
        )
    provenance = _provenance(request, build, resources)
    transfer = LinearTransferTable(
        scales,
        wavenumbers,
        transfer_values,
        LinearTransferDescriptor(
            request.transfer_fields,
            gauge=request.gauge,
            normalization="relative-to-primordial-curvature",
        ),
        scale,
        provenance,
        request.realization,
    )
    power = MatterPowerTable(
        scales,
        wavenumbers,
        power_values,
        MatterPowerDescriptor(
            request.power_field,
            request.power_field,
            gauge=request.gauge,
            stage="linear",
        ),
        scale,
        provenance,
        request.realization,
    )
    thermodynamics = (
        ThermodynamicsHistory(
            scales,
            *thermo_values,
            scale,
            provenance,
            request.realization,
        )
        if thermo_values is not None
        else None
    )
    return CosmologyModelResult(
        transfer,
        power,
        thermodynamics,
        standard_output="cache-hit",
        standard_error="",
        return_code=0,
    )


def _run_backend(
    *,
    backend: str,
    request: CosmologyModelRequest,
    build: BackendBuildManifest,
    resources: LinearTheoryResourcePolicy,
    cache_directory: str,
) -> PrecisionLinearTheoryResult:
    cache_root = Path(cache_directory).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    key = canonical_fingerprint(
        {
            "kind": "precision-linear-theory-cache",
            "backend": backend,
            "request": request.request_id,
            "build": build.build_id,
            "resources": resources.policy_id,
        }
    )
    path = cache_root / f"{key}.npz"
    cache_hit = path.is_file()
    if cache_hit:
        products = _load_products(path, request, build, resources)
    else:
        runner = SubprocessCosmologyModelBackend(
            build.application,
            arguments=build.arguments,
            timeout_seconds=resources.timeout_seconds,
            backend_name=backend,
            backend_version=build.release,
            numerical_policy_id=resources.policy_id,
        )
        products = runner.run(request)
        _save_products(path, products)
        if path.stat().st_size > resources.output_byte_cap:
            path.unlink()
            raise RuntimeError("Precision backend artifact exceeds its byte cap.")
    content_digest = _sha256(path)
    product_ids = (
        cosmology_product_content_id(products.transfer),
        cosmology_product_content_id(products.power),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="precision-linear-theory-products",
        content_digest=content_digest,
        producer=backend,
        producer_version=build.release,
        build_id=build.build_id,
        license_id=build.license_id,
        resource_id=resources.policy_id,
        status="complete",
        parent_artifact_ids=product_ids,
    )
    return PrecisionLinearTheoryResult(products, artifact, cache_hit)


class ClassLinearTheoryBackend(StrictModule, NonTrainableState):
    build: BackendBuildManifest
    resources: LinearTheoryResourcePolicy
    cache_directory: str = eqx.field(static=True)

    def __init__(
        self,
        build: BackendBuildManifest,
        resources: LinearTheoryResourcePolicy,
        cache_directory: str,
        /,
    ):
        if build.backend.lower() != "class":
            raise ValueError("ClassLinearTheoryBackend requires a CLASS build manifest.")
        self.build = build
        self.resources = resources
        self.cache_directory = str(cache_directory)

    def run(self, request: CosmologyModelRequest, /) -> PrecisionLinearTheoryResult:
        return _run_backend(
            backend="class",
            request=request,
            build=self.build,
            resources=self.resources,
            cache_directory=self.cache_directory,
        )


class CambLinearTheoryBackend(StrictModule, NonTrainableState):
    build: BackendBuildManifest
    resources: LinearTheoryResourcePolicy
    cache_directory: str = eqx.field(static=True)

    def __init__(
        self,
        build: BackendBuildManifest,
        resources: LinearTheoryResourcePolicy,
        cache_directory: str,
        /,
    ):
        if build.backend.lower() != "camb":
            raise ValueError("CambLinearTheoryBackend requires a CAMB build manifest.")
        self.build = build
        self.resources = resources
        self.cache_directory = str(cache_directory)

    def run(self, request: CosmologyModelRequest, /) -> PrecisionLinearTheoryResult:
        return _run_backend(
            backend="camb",
            request=request,
            build=self.build,
            resources=self.resources,
            cache_directory=self.cache_directory,
        )


class PrecisionBackendOverlapEvidence(StrictModule):
    maximum_transfer_absolute_error: jnp.ndarray
    maximum_transfer_relative_error: jnp.ndarray
    maximum_power_relative_error: jnp.ndarray
    fields_match: jnp.ndarray
    grids_match: jnp.ndarray
    finite: jnp.ndarray
    successful: jnp.ndarray


def compare_precision_backends(
    first: PrecisionLinearTheoryResult,
    second: PrecisionLinearTheoryResult,
    /,
    *,
    absolute_floor: float = 1.0e-12,
) -> PrecisionBackendOverlapEvidence:
    left = first.products
    right = second.products
    fields_match = left.transfer.descriptor.fields == right.transfer.descriptor.fields
    grids_match = (
        left.transfer.scale_factors.shape == right.transfer.scale_factors.shape
        and left.transfer.wavenumbers.shape == right.transfer.wavenumbers.shape
        and bool(jnp.all(left.transfer.scale_factors == right.transfer.scale_factors))
        and bool(jnp.all(left.transfer.wavenumbers == right.transfer.wavenumbers))
    )
    if not fields_match or not grids_match:
        zero = jnp.asarray(0.0)
        return PrecisionBackendOverlapEvidence(
            zero,
            jnp.asarray(jnp.inf),
            jnp.asarray(jnp.inf),
            jnp.asarray(fields_match),
            jnp.asarray(grids_match),
            jnp.asarray(False),
            jnp.asarray(False),
        )
    transfer_difference = jnp.abs(
        left.transfer.transfer_values - right.transfer.transfer_values
    )
    transfer_scale = jnp.maximum(
        jnp.maximum(
            jnp.abs(left.transfer.transfer_values),
            jnp.abs(right.transfer.transfer_values),
        ),
        absolute_floor,
    )
    power_difference = jnp.abs(left.power.power_values - right.power.power_values)
    power_scale = jnp.maximum(
        jnp.maximum(jnp.abs(left.power.power_values), jnp.abs(right.power.power_values)),
        absolute_floor,
    )
    finite = jnp.all(jnp.isfinite(transfer_difference)) & jnp.all(
        jnp.isfinite(power_difference)
    )
    return PrecisionBackendOverlapEvidence(
        jnp.max(transfer_difference),
        jnp.max(transfer_difference / transfer_scale),
        jnp.max(power_difference / power_scale),
        jnp.asarray(True),
        jnp.asarray(True),
        finite,
        finite,
    )


__all__ = [
    "BackendBuildManifest",
    "CambLinearTheoryBackend",
    "ClassLinearTheoryBackend",
    "LinearTheoryResourcePolicy",
    "PrecisionBackendOverlapEvidence",
    "PrecisionLinearTheoryResult",
    "compare_precision_backends",
]
