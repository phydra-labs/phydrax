"""Provider input and optional-availability contracts without native provider installs."""

import sys

import pytest

from phydrax.geometry.surface import SurfaceBooleanOperation
from phydrax.meshing import MeshingFailure, MeshingFailureCategory
from phydrax.meshing.providers import (
    GmshProvider,
    ManifoldProvider,
    TiogaProvider,
    VoroCrustOptions,
    VoroCrustProvider,
)
from tests._support.optional_surfaces import boolean_cube


def test_gmsh_constructor_refuses_untyped_options() -> None:
    with pytest.raises(TypeError, match="options"):
        GmshProvider("gmsh")  # ty: ignore[invalid-argument-type]


def test_tioga_constructor_refuses_untyped_options_before_worker_start() -> None:
    with pytest.raises(TypeError, match="options"):
        TiogaProvider("tioga")  # ty: ignore[invalid-argument-type]


def test_manifold_missing_dependency_retains_provider_unavailable_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_provider(name: str) -> None:
        return None

    monkeypatch.setattr(
        "phydrax.meshing.providers._manifold.find_spec",
        missing_provider,
    )
    with pytest.raises(MeshingFailure) as failure:
        ManifoldProvider().execute(
            (boolean_cube((0, 0, 0)), boolean_cube((0.5, 0, 0))),
            SurfaceBooleanOperation.UNION,
        )
    assert failure.value.category is MeshingFailureCategory.PROVIDER_UNAVAILABLE


def test_vorocrust_refuses_untyped_surface_before_worker_start() -> None:
    with VoroCrustProvider(
        sys.executable, "unused-vorocrust-boundary-worker"
    ) as provider:
        with pytest.raises(TypeError, match="surface"):
            provider.execute("surface", VoroCrustOptions(1.0))  # ty: ignore[invalid-argument-type]
