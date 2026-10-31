#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit, optional OCCT interoperability for native B-Rep sources.

OCP is imported only when one of these adapter operations is called. Importing
this module does not import OCP and never supplies an external tessellation as
native source geometry. In-memory OCCT shapes are serialized to the standard
BRep text carrier and decoded by the native codec, preserving the native source
curves, p-curves, incidence, and assembly uses. Triangle realization is a
separate optional native ``BRepTessellationPolicy`` request.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Protocol, Self, Sequence, TYPE_CHECKING, TypeVar

from .._host_io import open_regular_file
from .._physical import SpatialCoordinateContract
from .._publication import PublicationMode, publish_bytes


if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Shape

    from ..geometry.brep._constructors import BRepTessellationPolicy
    from ._cad import CadImportResult


class _IsSameEntity(Protocol):
    def IsSame(self, other: Self, /) -> bool: ...


_T = TypeVar("_T", bound=_IsSameEntity)


_OCCT_SYMBOL_MODULE = {
    "Bnd_Box": "Bnd",
    "BRep_Tool": "BRep",
    "BRepAdaptor_Curve": "BRepAdaptor",
    "BRepAdaptor_Surface": "BRepAdaptor",
    "BRepBndLib": "BRepBndLib",
    "BRepBuilderAPI_Copy": "BRepBuilderAPI",
    "BRepBuilderAPI_MakeVertex": "BRepBuilderAPI",
    "BRepClass_FaceClassifier": "BRepClass",
    "BRepClass3d_SolidClassifier": "BRepClass3d",
    "BRepExtrema_DistShapeShape": "BRepExtrema",
    "BRepTools": "BRepTools",
    "BRepTools_WireExplorer": "BRepTools",
    "GeomAPI_ProjectPointOnCurve": "GeomAPI",
    "GeomAPI_ProjectPointOnSurf": "GeomAPI",
    "GeomLProp_SLProps": "GeomLProp",
    "gp_Pnt": "gp",
    "gp_Pnt2d": "gp",
    "gp_Vec": "gp",
    "ShapeAnalysis_Surface": "ShapeAnalysis",
    "TopAbs_EDGE": "TopAbs",
    "TopAbs_FACE": "TopAbs",
    "TopAbs_IN": "TopAbs",
    "TopAbs_ON": "TopAbs",
    "TopAbs_REVERSED": "TopAbs",
    "TopAbs_SOLID": "TopAbs",
    "TopAbs_VERTEX": "TopAbs",
    "TopAbs_WIRE": "TopAbs",
    "TopExp": "TopExp",
    "TopExp_Explorer": "TopExp",
    "TopLoc_Location": "TopLoc",
    "TopoDS": "TopoDS",
    "TopoDS_Iterator": "TopoDS",
    "TopoDS_Shape": "TopoDS",
    "TopTools_FormatVersion_VERSION_1": "TopTools",
}


@dataclass(frozen=True, slots=True)
class _OCCTModules:
    Bnd: ModuleType
    BRep: ModuleType
    BRepAdaptor: ModuleType
    BRepBndLib: ModuleType
    BRepBuilderAPI: ModuleType
    BRepClass: ModuleType
    BRepClass3d: ModuleType
    BRepExtrema: ModuleType
    BRepTools: ModuleType
    GeomAPI: ModuleType
    GeomLProp: ModuleType
    gp: ModuleType
    IFSelect: ModuleType
    IGESControl: ModuleType
    ShapeAnalysis: ModuleType
    STEPControl: ModuleType
    TopAbs: ModuleType
    TopExp: ModuleType
    TopLoc: ModuleType
    TopoDS: ModuleType
    TopTools: ModuleType

    def symbol(self, name: str, /) -> Any:
        module_name = _OCCT_SYMBOL_MODULE.get(name)
        if module_name is None:
            raise AttributeError(f"OCCT symbol {name!r} is not in the adapter contract.")
        return getattr(getattr(self, module_name), name)


@lru_cache(maxsize=1)
def _occt_modules() -> _OCCTModules:
    """Load OCCT modules only on an explicit adapter operation."""
    try:
        return _OCCTModules(
            Bnd=importlib.import_module("OCP.Bnd"),
            BRep=importlib.import_module("OCP.BRep"),
            BRepAdaptor=importlib.import_module("OCP.BRepAdaptor"),
            BRepBndLib=importlib.import_module("OCP.BRepBndLib"),
            BRepBuilderAPI=importlib.import_module("OCP.BRepBuilderAPI"),
            BRepClass=importlib.import_module("OCP.BRepClass"),
            BRepClass3d=importlib.import_module("OCP.BRepClass3d"),
            BRepExtrema=importlib.import_module("OCP.BRepExtrema"),
            BRepTools=importlib.import_module("OCP.BRepTools"),
            GeomAPI=importlib.import_module("OCP.GeomAPI"),
            GeomLProp=importlib.import_module("OCP.GeomLProp"),
            gp=importlib.import_module("OCP.gp"),
            IFSelect=importlib.import_module("OCP.IFSelect"),
            IGESControl=importlib.import_module("OCP.IGESControl"),
            ShapeAnalysis=importlib.import_module("OCP.ShapeAnalysis"),
            STEPControl=importlib.import_module("OCP.STEPControl"),
            TopAbs=importlib.import_module("OCP.TopAbs"),
            TopExp=importlib.import_module("OCP.TopExp"),
            TopLoc=importlib.import_module("OCP.TopLoc"),
            TopoDS=importlib.import_module("OCP.TopoDS"),
            TopTools=importlib.import_module("OCP.TopTools"),
        )
    except ImportError as error:
        raise ImportError(
            "Explicit OCCT interoperability requires the optional OCP package."
        ) from error


def _explore(
    shape: TopoDS_Shape,
    kind: Any,
    caster: Callable[[Any], _T],
    occt: _OCCTModules,
    /,
) -> list[_T]:
    explorer = occt.TopExp.TopExp_Explorer(shape, kind)
    entities: list[_T] = []
    while explorer.More():
        entities.append(caster(explorer.Current()))
        explorer.Next()
    return entities


def explore_unique(
    shape: TopoDS_Shape,
    kind: Any,
    caster: Callable[[Any], _T],
    /,
) -> list[_T]:
    """Explore first-seen OCCT entities, deduplicating by ``IsSame`` identity."""
    occt = _occt_modules()
    entities: list[_T] = []
    for candidate in _explore(shape, kind, caster, occt):
        if not any(entity.IsSame(candidate) for entity in entities):
            entities.append(candidate)
    return entities


def shape_index(entities: Sequence[_IsSameEntity], candidate: _IsSameEntity, /) -> int:
    """Return the first-seen source index of one ``IsSame`` OCCT entity."""
    for index, entity in enumerate(entities):
        if entity.IsSame(candidate):
            return index
    raise RuntimeError("OCCT returned an entity absent from the global topology map.")


def _file_digest(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _canonical_brep_text(shape: TopoDS_Shape, occt: _OCCTModules, /) -> bytes:
    """Serialize source geometry without mutable flags or cached mesh data."""
    if shape.IsNull():
        raise ValueError("Cannot serialize a null OCCT shape.")
    copy = occt.BRepBuilderAPI.BRepBuilderAPI_Copy(shape, True, False).Shape()
    pending = [copy]
    while pending:
        current = pending.pop()
        current.Modified(False)
        current.Checked(False)
        children = occt.TopoDS.TopoDS_Iterator(current, False, False)
        while children.More():
            pending.append(children.Value())
            children.Next()
    stream = io.BytesIO()
    written = occt.BRepTools.BRepTools.Write_s(
        copy,
        stream,
        False,
        False,
        occt.TopTools.TopTools_FormatVersion_VERSION_1,
    )
    if written is False:
        raise ValueError("OCCT could not serialize the source shape as BRep text.")
    return stream.getvalue()


def shape_digest(shape: TopoDS_Shape, /) -> str:
    """Digest exact OCCT geometry/topology, independent of cached triangulation."""
    return hashlib.sha256(_canonical_brep_text(shape, _occt_modules())).hexdigest()


def _read_occt_shape(
    path: str | os.PathLike[str], occt: _OCCTModules, /
) -> tuple[TopoDS_Shape, str, str]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest_before = _file_digest(source)
    suffix = source.suffix.lower()
    if suffix in {".step", ".stp"}:
        reader = occt.STEPControl.STEPControl_Reader()
        if reader.ReadFile(str(source)) != occt.IFSelect.IFSelect_RetDone:
            raise ValueError(f"OCCT failed to read STEP file {source}.")
        reader.TransferRoots()
        shape = reader.OneShape()
        source_format = "step"
    elif suffix in {".iges", ".igs"}:
        reader = occt.IGESControl.IGESControl_Reader()
        if reader.ReadFile(str(source)) != occt.IFSelect.IFSelect_RetDone:
            raise ValueError(f"OCCT failed to read IGES file {source}.")
        reader.TransferRoots()
        shape = reader.OneShape()
        source_format = "iges"
    elif suffix in {".brep", ".brp"}:
        shape = occt.TopoDS.TopoDS_Shape()
        if not occt.BRepTools.BRepTools.Read_s(
            shape, str(source), occt.BRep.BRep_Builder()
        ):
            raise ValueError(f"OCCT failed to read BRep file {source}.")
        source_format = "brep"
    else:
        raise ValueError("OCCT shape reading supports STEP, IGES, and BRep text files.")
    if shape.IsNull():
        raise ValueError(f"CAD file {source} contains no transferable shape.")
    if _file_digest(source) != digest_before:
        raise OSError(f"CAD file changed while OCCT was reading it: {source}")
    return shape, source_format, digest_before


def read_occt_shape(path: str | os.PathLike[str], /) -> tuple[TopoDS_Shape, str, str]:
    """Explicitly read a raw OCCT shape for optional comparison/interoperability."""
    return _read_occt_shape(path, _occt_modules())


def import_occt_shape(
    shape: TopoDS_Shape,
    *,
    coordinate_contract: SpatialCoordinateContract,
    source_id: str = "occt-shape",
    source_digest: str | None = None,
    source_format: str = "occt-shape",
    tessellation: BRepTessellationPolicy | None = None,
    relative_geometric_tolerance: float = 1.0e-8,
) -> CadImportResult:
    """Convert an explicit OCCT shape through the native exact BRep-text codec.

    ``coordinate_contract`` declares the length unit of the shape's numeric
    coordinates. OCCT is only a serialization bridge here: source carriers,
    p-curves, topology and assembly uses are reconstructed by the native
    decoder. No OCCT mesh is adopted as source authority. The codec result is
    returned whole, so its import losses (dropped derived records, synthesized
    boundaries, chart-closure repairs, source tolerances) reach the caller with
    the model they describe.
    """
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
    identifier = str(source_id).strip()
    if not identifier:
        raise ValueError("source_id must be nonempty.")
    if not isinstance(source_format, str) or not source_format.strip():
        raise ValueError("source_format must be nonempty.")
    payload = _canonical_brep_text(shape, _occt_modules())
    from ._cad import CadImportPolicy
    from ._cad_brep_text import decode_brep_text_resource
    from ._cad_io import CadResource

    # The declared identity is bound when the model is published, so the
    # tessellation's source-fidelity proof is derived once, on that model.
    return decode_brep_text_resource(
        CadResource(payload, hashlib.sha256(payload).hexdigest(), identifier),
        CadImportPolicy(
            coordinate_contract,
            tessellation=tessellation,
            relative_geometric_tolerance=relative_geometric_tolerance,
        ),
        source_length_unit=coordinate_contract.length_unit,
        source_format=source_format,
        source_digest=source_digest,
    )


def persist_occt_shape(
    shape: TopoDS_Shape,
    destination: str | Path,
    /,
    *,
    coordinate_contract: SpatialCoordinateContract,
    overwrite: bool = False,
    tessellation: BRepTessellationPolicy | None = None,
) -> CadImportResult:
    """Publish a validated BRep-text carrier, then import its native source.

    Returns the same codec result as :func:`import_occt_shape`, bound to the
    published path and digest.
    """
    if not isinstance(coordinate_contract, SpatialCoordinateContract):
        raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
    if shape.IsNull():
        raise ValueError("Cannot persist a null OCCT shape.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be boolean.")
    target = Path(destination).expanduser().resolve()
    if target.suffix.lower() not in {".brep", ".brp"}:
        raise ValueError("A persisted OCCT shape requires a .brep or .brp destination.")
    occt = _occt_modules()
    payload = _canonical_brep_text(shape, occt)
    digest = hashlib.sha256(payload).hexdigest()

    def validate_staged(path: os.PathLike[str], /) -> None:
        # The staged carrier is named relative to the held publication directory,
        # so it is read through that descriptor rather than by a cwd-relative name.
        staged = occt.TopoDS.TopoDS_Shape()
        with open_regular_file(path) as stream:
            occt.BRepTools.BRepTools.Read_s(staged, stream, occt.BRep.BRep_Builder())
        if staged.IsNull():
            raise ValueError("OCCT could not validate staged native BRep text.")

    mode: PublicationMode = "atomic_replace" if overwrite else "exclusive"
    publish_bytes(
        target,
        payload,
        maximum_bytes=len(payload),
        mode=mode,
        validator=validate_staged,
    )
    return import_occt_shape(
        shape,
        coordinate_contract=coordinate_contract,
        source_id=os.fspath(target),
        source_digest=digest,
        source_format="brep",
        tessellation=tessellation,
    )


__all__ = [
    "explore_unique",
    "import_occt_shape",
    "persist_occt_shape",
    "read_occt_shape",
    "shape_digest",
    "shape_index",
]
