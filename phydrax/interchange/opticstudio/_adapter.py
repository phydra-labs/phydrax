#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only OpticStudio interoperability through the optional ZOSPy package."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np

from ..._fingerprint import canonical_fingerprint, canonical_json
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ScientificArtifactEnvelope
from ...backends import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
)
from .._report import (
    AdapterCapability,
    AdapterError,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterRequirement,
    AdapterStatus,
)


_AnalysisName = Literal["cardinal-points", "surface-data", "system-data"]
if TYPE_CHECKING:
    from ...optics.geometric import SequentialOpticsPlan


_SettingValue = bool | int | float | str | None


OPTICSTUDIO_CAPABILITIES = BackendCapabilities(
    backend="opticstudio",
    problem_kinds=(
        "optics.analysis.cardinal-points",
        "optics.analysis.surface-data",
        "optics.analysis.system-data",
        "optics.sequential.export",
    ),
    execution="host",
    host_only=True,
    supports_matrix_free=False,
    supports_assembled=False,
    coordinate_dtypes=("float64",),
    supports_plan_prepare_solve_refresh=False,
    requires_explicit_release=True,
)


_ADAPTER_CAPABILITIES = (
    AdapterCapability(
        "opticstudio.analysis.cardinal-points",
        detail="ZOSPy Cardinal Points analysis with JSON-normalized output.",
    ),
    AdapterCapability(
        "opticstudio.analysis.surface-data",
        detail="ZOSPy Surface Data analysis with JSON-normalized output.",
    ),
    AdapterCapability(
        "opticstudio.analysis.system-data",
        detail="ZOSPy System Data analysis with JSON-normalized output.",
    ),
    AdapterCapability(
        "opticstudio.sequential.aperture.circular",
        detail="Active circular clear apertures represented by semi-diameter.",
    ),
    AdapterCapability(
        "opticstudio.sequential.frame.coaxial",
        detail="A common rigid pose with nondecreasing axial surface positions.",
    ),
    AdapterCapability(
        "opticstudio.sequential.interaction.transmit",
        detail="Transmissive sequential interfaces.",
    ),
    AdapterCapability(
        "opticstudio.sequential.medium.isotropic",
        detail="Explicit real scalar refractive indices lowered as model materials.",
    ),
    AdapterCapability(
        "opticstudio.sequential.surface.conic",
        detail="Rotational conics without polynomial asphere coefficients.",
    ),
    AdapterCapability(
        "opticstudio.sequential.surface.plane",
        detail="Planar rotational surfaces.",
    ),
    AdapterCapability(
        "opticstudio.sequential.surface.sphere",
        detail="Spherical rotational surfaces.",
    ),
)


class _OpticStudioUnsupportedFeatureError(AdapterError):
    report: AdapterReport
    features: tuple[str, ...]

    def __init__(self, report: AdapterReport, features: Sequence[str], /):
        self.report = report
        self.features = tuple(sorted(str(feature) for feature in features))
        super().__init__(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "OpticStudio sequential export rejected unsupported required semantics: "
            + ", ".join(self.features)
            + ".",
        )


class _OpticStudioBoundaryError(TypeError):
    pass


@final
class OpticStudioBackend(AbstractExternalBackend, NonTrainableState):
    """Immutable configuration for a fresh standalone OpticStudio process."""

    zosapi_nethelper: str | None = eqx.field(static=True)
    opticstudio_directory: str | None = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        zosapi_nethelper: str | None = None,
        opticstudio_directory: str | None = None,
        license_id: str = "LicenseRef-OpticStudio-Proprietary",
    ):
        nethelper = _optional_nonempty_text(zosapi_nethelper, "zosapi_nethelper")
        directory = _optional_nonempty_text(
            opticstudio_directory, "opticstudio_directory"
        )
        if nethelper is not None and directory is not None:
            raise ValueError(
                "Specify zosapi_nethelper or opticstudio_directory, not both."
            )
        license_id_ = str(license_id).strip()
        if not license_id_:
            raise ValueError("license_id must be non-empty.")
        self.zosapi_nethelper = nethelper
        self.opticstudio_directory = directory
        self.license_id = license_id_
        self.backend_id = canonical_fingerprint(
            {
                "kind": "opticstudio-standalone-backend",
                "zosapi_nethelper": nethelper,
                "opticstudio_directory": directory,
                "license_id": license_id_,
            }
        )

    @property
    def name(self) -> str:
        return "opticstudio"

    @property
    def capabilities(self) -> BackendCapabilities:
        return OPTICSTUDIO_CAPABILITIES

    def availability(self, /) -> BackendAvailability:
        return opticstudio_availability()

    def open_session(self) -> OpticStudioSession:
        """Launch a standalone session; the caller must close it or use ``with``."""

        # This is intentionally the only session-opening import boundary.
        zospy = importlib.import_module("zospy")
        keyword_arguments: dict[str, str] = {}
        if self.zosapi_nethelper is not None:
            keyword_arguments["zosapi_nethelper"] = self.zosapi_nethelper
        if self.opticstudio_directory is not None:
            keyword_arguments["opticstudio_directory"] = self.opticstudio_directory
        connection = zospy.ZOS(**keyword_arguments)
        try:
            system = connection.connect(mode="standalone")
            version = str(vars(zospy).get("__version__", "unknown")).strip() or "unknown"
            return OpticStudioSession._open(
                self,
                zospy=zospy,
                connection=connection,
                system=system,
                producer_version=version,
            )
        except BaseException as connection_error:
            try:
                connection.disconnect()
            except BaseException as cleanup_error:
                connection_error.add_note(
                    "OpticStudio cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise


@final
class OpticStudioSession:
    """Owned standalone process with idempotent deterministic cleanup.

    Vendor objects are held only in name-mangled slots. Public operations expose
    immutable Phydrax records rather than ZOSPy or ZOS-API handles.
    """

    __slots__ = (
        "__backend",
        "__closed",
        "__connection",
        "__export_artifact_id",
        "__producer_version",
        "__session_id",
        "__system",
        "__zospy",
    )

    def __init__(self, *args: Any, **kwargs: Any):
        del args, kwargs
        raise TypeError("OpticStudioSession values are created by OpticStudioBackend.")

    @classmethod
    def _open(
        cls,
        backend: OpticStudioBackend,
        /,
        *,
        zospy: Any,
        connection: Any,
        system: Any,
        producer_version: str,
    ) -> OpticStudioSession:
        value = object.__new__(cls)
        value.__backend = backend
        value.__closed = False
        value.__connection = connection
        value.__export_artifact_id = None
        value.__producer_version = producer_version
        value.__session_id = canonical_fingerprint(
            {
                "kind": "opticstudio-session",
                "backend": backend.backend_id,
                "zospy_version": producer_version,
                "mode": "standalone",
            }
        )
        value.__system = system
        value.__zospy = zospy
        return value

    @property
    def backend(self) -> OpticStudioBackend:
        return self.__backend

    @property
    def closed(self) -> bool:
        return self.__closed

    @property
    def producer_version(self) -> str:
        return self.__producer_version

    @property
    def session_id(self) -> str:
        return self.__session_id

    def __enter__(self) -> OpticStudioSession:
        self.__require_open()
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> bool:
        del exception_type, exception, traceback
        self.close()
        return False

    def close(self) -> None:
        """Disconnect exactly once and discard every vendor reference."""

        if self.__closed:
            return
        self.__closed = True
        connection = self.__connection
        self.__connection = None
        self.__system = None
        self.__zospy = None
        self.__export_artifact_id = None
        connection.disconnect()

    def __require_open(self) -> None:
        if self.__closed:
            raise RuntimeError("OpticStudio session is closed.")

    def _write_sequential_export(self, export: _SequentialExport, /) -> None:
        self.__require_open()
        self.__export_artifact_id = None
        _write_sequential_system(self.__zospy, self.__system, export)

    def _run_analysis(self, request: OpticStudioAnalysisRequest, /) -> str:
        self.__require_open()
        settings = dict(request.settings)
        if request.analysis == "cardinal-points":
            vendor_result = self.__zospy.analyses.reports.CardinalPoints(**settings).run(
                self.__system, oncomplete="Close"
            )
        elif request.analysis == "surface-data":
            vendor_result = self.__zospy.analyses.reports.SurfaceData(**settings).run(
                self.__system, oncomplete="Close"
            )
        else:
            vendor_result = self.__zospy.analyses.reports.SystemData().run(
                self.__system, oncomplete="Close"
            )
        return _canonical_result_json(vendor_result.to_json())

    def _record_export(self, report: AdapterReport) -> None:
        self.__require_open()
        self.__export_artifact_id = report.target_id

    def _export_parent_ids(self) -> tuple[str, ...]:
        self.__require_open()
        if self.__export_artifact_id is None:
            return ()
        return (self.__export_artifact_id,)


@final
class OpticStudioAnalysisRequest(StrictModule, NonTrainableState):
    """Immutable request for one supported ZOSPy analysis wrapper."""

    analysis: _AnalysisName = eqx.field(static=True)
    settings: tuple[tuple[str, _SettingValue], ...] = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        analysis: _AnalysisName | str,
        /,
        *,
        settings: Mapping[str, object] | None = None,
    ):
        settings_input: Mapping[str, object] = {} if settings is None else settings
        analysis_ = str(analysis).strip().lower().replace("_", "-")
        if analysis_ not in ("cardinal-points", "surface-data", "system-data"):
            raise ValueError("Unsupported OpticStudio analysis request.")
        settings_ = _normalize_analysis_settings(analysis_, settings_input)
        self.analysis = analysis_  # type: ignore[assignment]
        self.settings = settings_
        self.request_id = canonical_fingerprint(
            {
                "kind": "opticstudio-analysis-request",
                "analysis": analysis_,
                "settings": dict(settings_),
            }
        )


@final
class OpticStudioRunResult(StrictModule, NonTrainableState):
    """Immutable JSON-normalized result and its auditable artifact evidence."""

    request: OpticStudioAnalysisRequest = eqx.field(static=True)
    payload_json: str = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        request: OpticStudioAnalysisRequest,
        payload_json: str,
        report: AdapterReport,
        artifact: ScientificArtifactEnvelope,
        /,
    ):
        if not isinstance(request, OpticStudioAnalysisRequest):
            raise TypeError("request must be an OpticStudioAnalysisRequest.")
        if not isinstance(report, AdapterReport):
            raise TypeError("report must be an AdapterReport.")
        if not isinstance(artifact, ScientificArtifactEnvelope):
            raise TypeError("artifact must be a ScientificArtifactEnvelope.")
        payload_json_ = _canonical_result_json(payload_json)
        if hashlib.sha256(payload_json_.encode("utf-8")).hexdigest() != (
            artifact.content_digest
        ):
            raise ValueError("Artifact digest does not identify the normalized result.")
        self.request = request
        self.payload_json = payload_json_
        self.report = report
        self.artifact = artifact
        self.result_id = canonical_fingerprint(
            {
                "kind": "opticstudio-run-result",
                "request": request.request_id,
                "report": report.report_id,
                "artifact": artifact.artifact_id,
            }
        )


@dataclass(frozen=True, slots=True)
class _SequentialSurface:
    radius: float
    conic: float
    thickness: float
    semi_diameter: float | None
    refractive_index_after: float


@dataclass(frozen=True, slots=True)
class _SequentialExport:
    source_id: str
    refractive_index_before: float
    surfaces: tuple[_SequentialSurface, ...]


@dataclass(frozen=True, slots=True)
class _SequentialPreflight:
    export: _SequentialExport | None
    report: AdapterReport
    unsupported_features: tuple[str, ...]


def opticstudio_availability() -> BackendAvailability:
    """Probe optional-package and host support without launching OpticStudio."""

    requirement = "install ZOSPy and a licensed Windows OpticStudio installation"
    # This is intentionally the only non-session import boundary.
    try:
        zospy = importlib.import_module("zospy")
    except (ImportError, OSError):
        return BackendAvailability(
            capabilities=OPTICSTUDIO_CAPABILITIES,
            available=False,
            requirement=requirement,
            reason="optional dependency 'zospy' could not be imported",
        )
    version = str(vars(zospy).get("__version__", "unknown")).strip() or "unknown"
    if sys.platform != "win32":
        return BackendAvailability(
            capabilities=OPTICSTUDIO_CAPABILITIES,
            available=False,
            requirement=requirement,
            reason=f"platform {sys.platform!r} is unsupported; expected 'win32'",
            versions=(("zospy", version),),
        )
    return BackendAvailability(
        capabilities=OPTICSTUDIO_CAPABILITIES,
        available=True,
        requirement=requirement,
        reason="ZOSPy imported on a supported host",
        versions=(("zospy", version),),
    )


def export_sequential_to_opticstudio(
    plan: SequentialOpticsPlan,
    session: OpticStudioSession,
    /,
    *,
    length_unit_in_metres: float,
) -> AdapterReport:
    """Replace the session system with a strict supported sequential-plan export."""

    if not isinstance(session, OpticStudioSession):
        raise TypeError("session must be an OpticStudioSession.")
    if session.closed:
        raise RuntimeError("OpticStudio session is closed.")
    _reject_traced_values(plan, "plan")
    _reject_traced_values(length_unit_in_metres, "length_unit_in_metres")
    length_scale = float(length_unit_in_metres)
    if not math.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("length_unit_in_metres must be finite and positive.")
    preflight = _preflight_sequential_export(plan, length_scale)
    if preflight.export is None:
        raise _OpticStudioUnsupportedFeatureError(
            preflight.report, preflight.unsupported_features
        )
    session._write_sequential_export(preflight.export)
    session._record_export(preflight.report)
    return preflight.report


def run_opticstudio_analysis(
    session: OpticStudioSession, request: OpticStudioAnalysisRequest, /
) -> OpticStudioRunResult:
    """Run one analysis and detach its result into immutable canonical JSON."""

    if not isinstance(session, OpticStudioSession):
        raise TypeError("session must be an OpticStudioSession.")
    if not isinstance(request, OpticStudioAnalysisRequest):
        raise TypeError("request must be an OpticStudioAnalysisRequest.")
    payload_json = session._run_analysis(request)
    content_digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    target_id = canonical_fingerprint(
        {
            "kind": "opticstudio-analysis-json",
            "request": request.request_id,
            "content_digest": content_digest,
        }
    )
    capability = _analysis_capability(request.analysis)
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "opticstudio-analysis-result",
        "application/json",
        source_id=session.session_id,
        target_id=target_id,
        stage="opticstudio-analysis",
        source_profile=AdapterFormatProfile(
            "opticstudio-analysis-result",
            qualifiers={"analysis": request.analysis},
        ),
        target_profile=AdapterFormatProfile(
            "application/json",
            qualifiers={"normalization": "phydrax-canonical-json"},
        ),
        preserved_fields=("data", "header", "messages", "metadata", "settings"),
        requirements=(
            AdapterRequirement(
                capability.semantic_id,
                rationale="The requested analysis must have a supported ZOSPy wrapper.",
            ),
        ),
        capabilities=(capability,),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind=f"opticstudio-analysis/{request.analysis}",
        content_digest=content_digest,
        producer="ZOSPy/OpticStudio",
        producer_version=session.producer_version,
        build_id=session.backend.backend_id,
        license_id=session.backend.license_id,
        resource_id=request.request_id,
        status="complete",
        parent_artifact_ids=session._export_parent_ids(),
    )
    return OpticStudioRunResult(request, payload_json, report, artifact)


def _preflight_sequential_export(
    plan: SequentialOpticsPlan, length_scale: float, /
) -> _SequentialPreflight:
    from ...optics.geometric import SequentialOpticsPlan

    if not isinstance(plan, SequentialOpticsPlan):
        raise TypeError("plan must be a SequentialOpticsPlan.")
    source_id = plan.plan_id
    surface_kinds = tuple(str(value) for value in plan.surface_kinds)
    interactions = tuple(str(value) for value in plan.interactions)
    requirements = [
        AdapterRequirement(
            "opticstudio.sequential.medium.isotropic",
            rationale="SequentialOpticsPlan stores scalar isotropic refractive indices.",
        ),
        AdapterRequirement(
            "opticstudio.sequential.frame.coaxial",
            rationale="Standard sequential rows require one coaxial surface sequence.",
        ),
    ]
    for surface_kind in sorted(set(surface_kinds)):
        requirements.append(
            AdapterRequirement(
                f"opticstudio.sequential.surface.{surface_kind}",
                rationale=f"The plan contains {surface_kind} surfaces.",
            )
        )
    for interaction in sorted(set(interactions)):
        requirements.append(
            AdapterRequirement(
                f"opticstudio.sequential.interaction.{interaction}",
                rationale=f"The plan contains {interaction} interactions.",
            )
        )
    aperture_active = np.asarray(plan.aperture_active, dtype=bool)
    if bool(np.any(aperture_active)):
        requirements.append(
            AdapterRequirement(
                "opticstudio.sequential.aperture.circular",
                rationale="The plan contains active circular clear apertures.",
            )
        )

    losses: list[AdapterLoss] = []
    unsupported: list[str] = []
    for index, surface_kind in enumerate(surface_kinds):
        if surface_kind not in ("plane", "sphere", "conic"):
            feature = f"surfaces[{index}].surface_kind={surface_kind}"
            unsupported.append(feature)
            losses.append(
                AdapterLoss(
                    f"surfaces[{index}].surface_kind",
                    "export",
                    "unsupported",
                    "Only plane, sphere, and conic sequential surfaces are supported.",
                    changes_interpretation=True,
                )
            )
    for index, interaction in enumerate(interactions):
        if interaction != "transmit":
            feature = f"surfaces[{index}].interaction={interaction}"
            unsupported.append(feature)
            losses.append(
                AdapterLoss(
                    f"surfaces[{index}].interaction",
                    "export",
                    "unsupported",
                    "The OpticStudio subset supports transmissive interactions only.",
                    changes_interpretation=True,
                )
            )

    axial_positions, frame_losses, frame_features = _coaxial_positions(
        plan.frames, length_scale
    )
    losses.extend(frame_losses)
    unsupported.extend(frame_features)
    target_id = canonical_fingerprint(
        {
            "kind": "opticstudio-sequential-export",
            "source": source_id,
            "length_unit_in_metres": length_scale,
            "profile": "plane-sphere-conic-isotropic-circular-aperture",
        }
    )
    source_profile = AdapterFormatProfile(
        "phydrax-sequential-optics-plan",
        qualifiers={
            "source_length_unit_in_metres": format(length_scale, ".17g"),
        },
    )
    target_profile = AdapterFormatProfile(
        "opticstudio-sequential-system",
        qualifiers={
            "lens_unit": "metre",
            "material_model": "isotropic-index",
            "surface_subset": "plane-sphere-conic",
        },
    )
    if unsupported:
        report = AdapterReport(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            source_profile.format,
            target_profile.format,
            source_id=source_id,
            target_id=target_id,
            stage="opticstudio-sequential-export",
            source_profile=source_profile,
            target_profile=target_profile,
            coordinate_mapping=(
                "Common Phydrax rigid pose maps to the OpticStudio positive z axis.",
            ),
            preserved_fields=(),
            assumptions=(
                "All plan lengths use the explicitly supplied source length unit.",
            ),
            losses=losses,
            requirements=requirements,
            capabilities=_ADAPTER_CAPABILITIES,
        )
        return _SequentialPreflight(None, report, tuple(sorted(unsupported)))

    curvatures = np.asarray(plan.curvatures, dtype=float)
    conics = np.asarray(plan.conic_constants, dtype=float)
    semi_diameters = np.asarray(plan.clear_semi_diameters, dtype=float)
    refractive_indices = np.asarray(plan.refractive_indices, dtype=float)
    surfaces = []
    for index, surface_kind in enumerate(surface_kinds):
        curvature = float(curvatures[index])
        radius = 0.0 if curvature == 0.0 else length_scale / curvature
        thickness = (
            0.0
            if index + 1 == len(surface_kinds)
            else float(axial_positions[index + 1] - axial_positions[index])
        )
        semi_diameter = (
            float(semi_diameters[index]) * length_scale
            if bool(aperture_active[index])
            else None
        )
        surfaces.append(
            _SequentialSurface(
                radius,
                float(conics[index]),
                thickness,
                semi_diameter,
                float(refractive_indices[index + 1]),
            )
        )
    export = _SequentialExport(
        source_id,
        float(refractive_indices[0]),
        tuple(surfaces),
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        source_profile.format,
        target_profile.format,
        source_id=source_id,
        target_id=target_id,
        stage="opticstudio-sequential-export",
        source_profile=source_profile,
        target_profile=target_profile,
        coordinate_mapping=(
            "Common Phydrax rigid pose maps to the OpticStudio positive z axis.",
        ),
        preserved_fields=(
            "surface_kind",
            "curvature",
            "conic_constant",
            "axial_spacing",
            "refractive_index",
            "circular_clear_aperture",
        ),
        assumptions=(
            "All plan lengths use the explicitly supplied source length unit.",
            "The final physical surface and OpticStudio image surface are coincident.",
        ),
        requirements=requirements,
        capabilities=_ADAPTER_CAPABILITIES,
    )
    return _SequentialPreflight(export, report, ())


def _coaxial_positions(
    frames: Sequence[Any], length_scale: float, /
) -> tuple[np.ndarray, tuple[AdapterLoss, ...], tuple[str, ...]]:
    rotations = tuple(np.asarray(frame.rotation, dtype=float) for frame in frames)
    translations = tuple(np.asarray(frame.translation, dtype=float) for frame in frames)
    losses: list[AdapterLoss] = []
    features: list[str] = []
    if not rotations:
        return np.empty((0,), dtype=float), (), ()
    reference_rotation = rotations[0]
    reference_translation = translations[0]
    axis = reference_rotation[:, -1]
    axial_positions = []
    for index, (rotation, translation) in enumerate(
        zip(rotations, translations, strict=True)
    ):
        if not np.allclose(rotation, reference_rotation, rtol=0.0, atol=1.0e-10):
            features.append(f"frames[{index}].rotation=noncoaxial")
            losses.append(
                AdapterLoss(
                    f"frames[{index}].rotation",
                    "export",
                    "unsupported",
                    "Tilted sequential surfaces require coordinate-break semantics.",
                    changes_interpretation=True,
                )
            )
        offset = translation - reference_translation
        axial = float(np.dot(offset, axis))
        transverse = offset - axial * axis
        if not np.allclose(transverse, 0.0, rtol=0.0, atol=1.0e-10):
            features.append(f"frames[{index}].translation=decentered")
            losses.append(
                AdapterLoss(
                    f"frames[{index}].translation",
                    "export",
                    "unsupported",
                    "Decentered sequential surfaces require coordinate-break semantics.",
                    changes_interpretation=True,
                )
            )
        axial_positions.append(axial * length_scale)
    positions = np.asarray(axial_positions, dtype=float)
    for index, distance in enumerate(np.diff(positions)):
        if distance < 0.0:
            features.append(f"frames[{index + 1}].translation=reverse-ordered")
            losses.append(
                AdapterLoss(
                    f"frames[{index + 1}].translation",
                    "export",
                    "unsupported",
                    "Transmissive sequential surfaces must be ordered along positive z.",
                    changes_interpretation=True,
                )
            )
    return positions, tuple(losses), tuple(features)


def _write_sequential_system(
    zospy: Any, system: Any, export: _SequentialExport, /
) -> None:
    system.new(saveifneeded=False)
    system.make_sequential()
    system.SystemData.Units.LensUnits = zospy.constants.SystemData.ZemaxSystemUnits.Meters
    editor = system.LDE
    object_surface = editor.GetSurfaceAt(0)
    zospy.solvers.material_model(
        object_surface.MaterialCell,
        refractive_index=export.refractive_index_before,
    )
    for index, description in enumerate(export.surfaces, start=1):
        surface = editor.InsertNewSurfaceAt(index)
        surface.Comment = f"Phydrax sequential surface {index - 1}"
        surface.Radius = description.radius
        surface.Conic = description.conic
        surface.Thickness = description.thickness
        if description.semi_diameter is not None:
            surface.SemiDiameter = description.semi_diameter
        zospy.solvers.material_model(
            surface.MaterialCell,
            refractive_index=description.refractive_index_after,
        )


def _analysis_capability(analysis: _AnalysisName, /) -> AdapterCapability:
    semantic_id = f"opticstudio.analysis.{analysis}"
    return next(
        capability
        for capability in _ADAPTER_CAPABILITIES
        if capability.semantic_id == semantic_id
    )


def _normalize_analysis_settings(
    analysis: str, settings: Mapping[str, object], /
) -> tuple[tuple[str, _SettingValue], ...]:
    if not isinstance(settings, Mapping):
        raise TypeError("OpticStudio analysis settings must be a mapping.")
    allowed = {
        "cardinal-points": frozenset(
            ("orientation", "surface_1", "surface_2", "wavelength")
        ),
        "surface-data": frozenset(("surface",)),
        "system-data": frozenset(),
    }[analysis]
    normalized: list[tuple[str, _SettingValue]] = []
    for raw_key, raw_value in settings.items():
        if not isinstance(raw_key, str):
            raise TypeError("OpticStudio analysis setting names must be strings.")
        key = raw_key.strip().lower()
        if key not in allowed:
            raise ValueError(f"Unsupported {analysis} setting {key!r}.")
        _reject_traced_values(raw_value, f"settings[{key!r}]")
        value = _normalize_setting_value(raw_value, key)
        normalized.append((key, value))
    result = tuple(sorted(normalized))
    values = dict(result)
    if analysis == "cardinal-points":
        if "orientation" in values and values["orientation"] not in ("X-Z", "Y-Z"):
            raise ValueError("Cardinal-points orientation must be 'X-Z' or 'Y-Z'.")
        for key in ("surface_1", "wavelength"):
            if key in values and (
                not isinstance(values[key], int)
                or isinstance(values[key], bool)
                or values[key] < 1
            ):
                raise ValueError(f"Cardinal-points {key} must be a positive integer.")
        if "surface_2" in values:
            second = values["surface_2"]
            if second != "Image" and (
                not isinstance(second, int) or isinstance(second, bool) or second < 1
            ):
                raise ValueError(
                    "Cardinal-points surface_2 must be a positive integer or 'Image'."
                )
        first = values.get("surface_1", 1)
        second = values.get("surface_2", "Image")
        if isinstance(first, int) and isinstance(second, int) and first >= second:
            raise ValueError("Cardinal-points surface_1 must precede numeric surface_2.")
    elif analysis == "surface-data":
        surface = values.get("surface")
        if not isinstance(surface, int) or isinstance(surface, bool) or surface < 0:
            raise ValueError("Surface-data surface must be a nonnegative integer.")
    return result


def _normalize_setting_value(value: object, name: str, /) -> _SettingValue:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        normalized: _SettingValue = value
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"OpticStudio setting {name!r} must be finite.")
        normalized = value
    else:
        raise TypeError(
            f"OpticStudio setting {name!r} must be a JSON scalar, not "
            f"{type(value).__name__}."
        )
    if isinstance(normalized, str):
        normalized = normalized.strip()
        if not normalized:
            raise ValueError(f"OpticStudio setting {name!r} must be non-empty.")
    return normalized


def _canonical_result_json(payload: str, /) -> str:
    if not isinstance(payload, str):
        raise TypeError("ZOSPy analysis results must serialize to JSON text.")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("ZOSPy returned malformed analysis JSON.") from error
    return canonical_json(value)


def _reject_traced_values(value: object, owner: str, /) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        if isinstance(leaf, jax.core.Tracer):
            raise _OpticStudioBoundaryError(
                f"{owner} contains a traced value at the host-only, "
                "non-differentiable OpticStudio boundary."
            )


def _optional_nonempty_text(value: str | None, name: str, /) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty when provided.")
    return normalized


__all__ = [
    "OPTICSTUDIO_CAPABILITIES",
    "OpticStudioAnalysisRequest",
    "OpticStudioBackend",
    "OpticStudioRunResult",
    "OpticStudioSession",
    "export_sequential_to_opticstudio",
    "opticstudio_availability",
    "run_opticstudio_analysis",
]
