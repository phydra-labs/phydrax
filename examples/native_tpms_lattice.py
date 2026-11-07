#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified TPMS lattices: a periodic Schwarz P cell and a clipped diamond specimen.

1. The Schwarz P nodal table ``cos 2πx + cos 2πy + cos 2πz`` has certified
   critical levels; the level ``c = 1`` meets a critical value and is refused
   with its separation evidence, while ``c = 0`` is accepted.
2. The network ``{F <= 0}`` is discovered on the quotient of the unit cubic
   cell (three-torus) and reported with its quotient topology (genus 3) and
   certified volume-fraction enclosure. The half-lattice shift proves
   ``F(u + 1/2) = -F(u)``, so the exact fraction is 1/2.
3. The quotient region is filled with a periodic tetrahedral mesh whose paired
   cell faces are exact lattice translates, and its effective conductivity
   (solid conductivity 1, void pores) is homogenized with Hill–Mandel evidence.
4. A Schwarz D ("diamond") network clipped by a cylinder becomes a certified
   solid with crease curves and trimmed caps.
5. Both solids are exported as 3MF under a closed, oriented 2-manifold
   certificate.

Run from the repository root::

    python examples/native_tpms_lattice.py [output-directory]
"""

import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

import phydrax as phx
from phydrax.applications.solid_mechanics import PeriodicCellProblem
from phydrax.discretization import PeriodicCell
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    discover_periodic_trigonometric_surface,
)
from phydrax.geometry.surface import SurfaceExportPolicy
from phydrax.units import MILLIMETER


G = phx.geometry
M = phx.meshing


def critical_levels() -> dict[str, object]:
    table = G.TrigonometricTable.family("schwarz_p")
    try:
        G.TrigonometricLevelSet(table, PeriodicCell(np.eye(3)), level=1.0)
    except G.TrigonometricSingularLevelError as refusal:
        outcome = refusal.separation.outcome
    else:
        raise RuntimeError("Schwarz P at c = 1 must be refused as singular.")
    half = Fraction(1, 2)
    sign = table.symmetry_sign(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (half, half, half))
    return {"refused_level": 1.0, "refusal_outcome": outcome, "half_shift_sign": sign}


def periodic_cell(directory: Path) -> dict[str, object]:
    source = G.TrigonometricLevelSet(
        G.TrigonometricTable.family("schwarz_p"), PeriodicCell(np.eye(3))
    )
    surface = discover_periodic_trigonometric_surface(
        source,
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.2, root_tolerance=1.0e-8
        ),
        source_id="schwarz-p-network",
    )
    if surface.topology is None:
        raise RuntimeError("Quotient discovery left unresolved boxes.")
    metrics = surface.lattice_metrics()
    volume = M.mesh_periodic_trigonometric_region(surface)
    conduction = PeriodicCellProblem(volume.mesh, physics="conductivity").solve(1.0)
    phi = float(conduction.volume_fraction)
    exported = M.export_lattice_solid(
        directory / "schwarz-p-cell.3mf", volume, SurfaceExportPolicy(MILLIMETER)
    )
    return {
        "genus": list(surface.topology.component_genera),
        "certified_solid_fraction": [
            metrics.solid_fraction_lower,
            metrics.solid_fraction_upper,
        ],
        "measured_specific_surface": metrics.measured_specific_surface,
        "tetrahedra": sum(block.cell_count for block in volume.mesh.blocks),
        "quotient_region_components": volume.quotient_region_components,
        "mesh_volume_fraction": phi,
        "effective_conductivity": np.asarray(conduction.effective_tensor).tolist(),
        "hill_mandel_accepted": bool(conduction.accepted),
        "cubic_symmetry_defect": float(conduction.cubic_symmetry_defect),
        "hashin_shtrikman_upper": 2.0 * phi / (3.0 - phi),
        "export": {
            "path": exported.path.name,
            "certificate_id": exported.certificate.certificate_id,
            "lossy": exported.lossy,
        },
    }


def diamond_specimen(directory: Path) -> dict[str, object]:
    network = G.TrigonometricLevelSet(
        G.TrigonometricTable.family("schwarz_d"), PeriodicCell(np.eye(3))
    )
    envelope = G.Cylinder((0.13, 0.21, 0.07), (0.0, 0.0, 0.4), 0.25)
    surface = M.construct_lattice_specimen_surface(
        G.Intersection(network, envelope),
        policy=AdaptiveImplicitSurfacePolicy(
            relative_cell_diameter=0.125, root_tolerance=1.0e-9
        ),
        source_id="diamond-cylinder",
    )
    volume = M.mesh_lattice_specimen(surface)
    if not volume.certified_solid:
        raise RuntimeError("The specimen is not a certified solid.")
    exported = M.export_lattice_solid(
        directory / "diamond-cylinder.3mf", volume, SurfaceExportPolicy(MILLIMETER)
    )
    return {
        "faces": dict(
            zip(
                surface.facet_names,
                np.bincount(
                    np.asarray(surface.triangle_facets),
                    minlength=len(surface.facet_names),
                ).tolist(),
                strict=True,
            )
        ),
        "crease_segments": int(surface.crease_segments.shape[0]),
        "fidelity": {
            "implicit_face_distance": surface.implicit_face_distance,
            "cap_deviation": surface.cap_deviation,
            "crease_root_distance": surface.crease_root_distance,
        },
        "measured_volume": volume.measure,
        "measured_porosity": volume.porosity,
        "export": {
            "path": exported.path.name,
            "certificate_id": exported.certificate.certificate_id,
            "enclosed_volume": exported.enclosed_volume,
        },
    }


def main() -> None:
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
        directory.mkdir(parents=True, exist_ok=True)
        report = {
            "critical_levels": critical_levels(),
            "periodic_cell": periodic_cell(directory),
            "diamond_specimen": diamond_specimen(directory),
        }
    else:
        with tempfile.TemporaryDirectory(prefix="phydrax-tpms-example-") as temporary:
            report = {
                "critical_levels": critical_levels(),
                "periodic_cell": periodic_cell(Path(temporary)),
                "diamond_specimen": diamond_specimen(Path(temporary)),
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
