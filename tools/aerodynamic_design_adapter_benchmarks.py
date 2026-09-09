#!/usr/bin/env python
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Real-engine aerodynamic qualification; missing engines are errors, not skips.

Run from the repository with PYTHONPATH=. and an explicit pinned executable.
DAFoam needs its sourced OpenFOAM/PETSc environment and Python 3 environment
containing DAFoam 4.0.3, OpenMDAO, MPhys, mpi4py, petsc4py and IDWarp. The benchmark
creates its own NACA0012 finite-volume mesh and case from public equations; no
third-party tutorial source or mesh is copied, and no engine is installed here.
The deliberately modest mesh is a runtime/derivative qualification, not a
mesh-independent aerodynamic accuracy claim.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from phydrax.interchange.dafoam import pin_dafoam_runtime, run_dafoam
from phydrax.interchange.energy_runtime import pin_energy_executable
from phydrax.interchange.xfoil import run_xfoil_polar, XFOILOperatingPoint


GEOMETRY_SOURCE = "https://www.pdas.com/naca456thick4.html; analytic NACA0012, finite trailing-edge thickness"


def naca0012_coordinates(half_panels: int = 80) -> tuple[tuple[float, float], ...]:
    """Public NACA four-digit thickness equation; clockwise is not imposed."""
    if not 10 <= half_panels <= 119:
        raise ValueError("half_panels must lie in [10, 119].")
    x = [0.5 * (1 - math.cos(math.pi * i / half_panels)) for i in range(half_panels + 1)]
    y = [
        0.12
        * (
            1.4845 * math.sqrt(v)
            + v * (-0.630 + v * (-1.758 + v * (1.4215 - 0.5075 * v)))
        )
        for v in x
    ]
    return tuple(zip(reversed(x), reversed(y))) + tuple(
        (x[i], -y[i]) for i in range(1, len(x))
    )


def _foam(name: str, body: str, cls: str = "dictionary") -> bytes:
    return (
        f"FoamFile\n{{ version 2.0; format ascii; class {cls}; object {name}; }}\n"
        + body
        + "\n"
    ).encode("ascii")


def _mesh_files(surface, radial_cells=48, radius=20.0, span=0.1):
    """Explicit outward-oriented, one-cell-span annular hexahedral OpenFOAM mesh."""
    surface = np.asarray(surface, dtype=float)
    n = len(surface)
    center = np.array([0.5, 0.0])
    directions = surface - center
    distances = np.sqrt(np.sum(directions * directions, axis=1))
    outer = center + radius * directions / distances[:, None]
    # Geometric radial spacing resolves near-wall variation without zero-volume
    # cells at either the leading edge or the finite-thickness trailing edge.
    eta = np.expm1(np.linspace(0, 8, radial_cells + 1)) / np.expm1(8)
    rings = (1 - eta[:, None, None]) * surface + eta[:, None, None] * outer
    xyz = np.empty((2, radial_cells + 1, n, 3))
    xyz[..., :2] = rings
    xyz[0, ..., 2] = 0
    xyz[1, ..., 2] = span
    points = xyz.reshape(-1, 3)

    def node(z, r, i):
        return (z * (radial_cells + 1) + r) * n + i % n

    faces = {}
    for r in range(radial_cells):
        for i in range(n):
            # (r, theta, z) is right-handed for this counterclockwise contour.
            v = (
                node(0, r, i),
                node(0, r + 1, i),
                node(0, r + 1, i + 1),
                node(0, r, i + 1),
                node(1, r, i),
                node(1, r + 1, i),
                node(1, r + 1, i + 1),
                node(1, r, i + 1),
            )
            cell = r * n + i
            local = (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            )
            labels = (
                "symmetry1",
                "symmetry2",
                None,
                "inout" if r == radial_cells - 1 else None,
                None,
                "wing" if r == 0 else None,
            )
            for indices, patch in zip(local, labels):
                face = tuple(v[j] for j in indices)
                key = tuple(sorted(face))
                if key in faces:
                    faces[key][2] = cell
                else:
                    faces[key] = [face, cell, None, patch]
    internal = [f for f in faces.values() if f[2] is not None]
    internal.sort(key=lambda f: (f[1], f[2]))
    ordered = list(internal)
    boundary = []
    for patch in ("wing", "inout", "symmetry1", "symmetry2"):
        group = [f for f in faces.values() if f[2] is None and f[3] == patch]
        kind = (
            "wall"
            if patch == "wing"
            else "symmetry"
            if patch.startswith("symmetry")
            else "patch"
        )
        boundary.append(
            f"{patch}\n{{ type {kind}; nFaces {len(group)}; startFace {len(ordered)}; }}"
        )
        ordered.extend(group)
    if len(ordered) != len(faces):
        raise ValueError("Mesh topology has an unassigned boundary face.")
    lists = {
        "points": (
            "vectorField",
            ["(" + " ".join(f"{v:.17g}" for v in p) + ")" for p in points],
        ),
        "faces": ("faceList", ["4(" + " ".join(map(str, f[0])) + ")" for f in ordered]),
        "owner": ("labelList", [str(f[1]) for f in ordered]),
        "neighbour": ("labelList", [str(f[2]) for f in internal]),
        "boundary": ("polyBoundaryMesh", boundary),
    }
    return {
        "constant/polyMesh/" + name: _foam(
            name, f"{len(rows)}\n(\n" + "\n".join(rows) + "\n)", cls
        )
        for name, (cls, rows) in lists.items()
    }


def naca0012_dafoam_case(half_panels: int = 64, radial_cells: int = 48):
    """Generated steady incompressible SA case, U=10 m/s, chord=1 m, Re=1e6."""
    if not 16 <= radial_cells <= 256:
        raise ValueError("radial_cells must lie in [16, 256].")
    files = _mesh_files(naca0012_coordinates(half_panels), radial_cells)
    dictionaries = {
        "system/controlDict": "startFrom startTime; startTime 0; stopAt endTime; endTime 3000; deltaT 1;\n"
        "writeControl timeStep; writeInterval 3000; purgeWrite 0; writeFormat ascii; writePrecision 16;\n"
        "writeCompression off; timeFormat general; timePrecision 16; runTimeModifiable false;\n",
        "system/fvSchemes": "ddtSchemes { default steadyState; }\n"
        "gradSchemes { default Gauss linear; }\n"
        "divSchemes { default none; div(phi,U) bounded Gauss linearUpwindV grad(U);\n"
        "div(phi,nuTilda) bounded Gauss upwind; div((nuEff*dev2(T(grad(U))))) Gauss linear;\n"
        "div(pc) bounded Gauss upwind; }\n"
        "laplacianSchemes { default Gauss linear corrected; }\n"
        "interpolationSchemes { default linear; } snGradSchemes { default corrected; }\n"
        "wallDist { method meshWave; }\n",
        "system/fvSolution": "SIMPLE { consistent false; nNonOrthogonalCorrectors 1; }\n"
        "solvers { p { solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.05; }\n"
        '"(U|nuTilda)" { solver smoothSolver; smoother GaussSeidel; tolerance 1e-10; relTol 0.05; nSweeps 2; } }\n'
        "relaxationFactors { fields { p 0.3; } equations { U 0.7; nuTilda 0.7; } }\n",
        "constant/transportProperties": "transportModel Newtonian; nu [0 2 -1 0 0 0 0] 1e-5;\n",
        "constant/turbulenceProperties": (
            "simulationType RAS; RAS { RASModel SpalartAllmaras; "
            "turbulence on; printCoeffs off; }\n"
        ),
    }
    files.update(
        {path: _foam(Path(path).name, body) for path, body in dictionaries.items()}
    )
    fields = {
        "U": (
            "volVectorField",
            "0 1 -1 0 0 0 0",
            "(10 0 0)",
            "type fixedValue; value uniform (0 0 0);",
            "type inletOutlet; inletValue uniform (10 0 0); value uniform (10 0 0);",
        ),
        "p": (
            "volScalarField",
            "0 2 -2 0 0 0 0",
            "0",
            "type zeroGradient;",
            "type fixedValue; value uniform 0;",
        ),
        "nuTilda": (
            "volScalarField",
            "0 2 -1 0 0 0 0",
            "3e-5",
            "type fixedValue; value uniform 0;",
            "type inletOutlet; inletValue uniform 3e-5; value uniform 3e-5;",
        ),
        "nut": (
            "volScalarField",
            "0 2 -1 0 0 0 0",
            "3e-5",
            "type nutUSpaldingWallFunction; value uniform 3e-5;",
            "type calculated; value uniform 3e-5;",
        ),
    }
    for name, (cls, dimensions, initial, wall, farfield) in fields.items():
        files["0/" + name] = _foam(
            name,
            f"dimensions [{dimensions}]; internalField uniform {initial};\n"
            f"boundaryField {{ wing {{ {wall} }} inout {{ {farfield} }}\n"
            "symmetry1 { type symmetry; } symmetry2 { type symmetry; } }",
            cls,
        )
    options = {
        "solverName": "DASimpleFoam",
        "designSurfaces": ["wing"],
        "primalMinResTol": 1e-8,
        "primalMinResTolDiff": 10.0,
        "checkMeshThreshold": {
            "maxAspectRatio": 1000.0,
            "maxNonOrth": 85.0,
            "maxSkewness": 4.0,
            "maxIncorrectlyOrientedFaces": 0,
        },
        "primalBC": {"useWallFunction": True},
        "normalizeStates": {"U": 10.0, "p": 50.0, "nuTilda": 3e-4, "phi": 1.0},
        "function": {
            name: {
                "type": "force",
                "source": "patchToFace",
                "patches": ["wing"],
                "directionMode": direction,
                "patchVelocityInputName": "patchV",
                "scale": 0.2,
            }
            for name, direction in (("CD", "parallelToFlow"), ("CL", "normalToFlow"))
        },
        "inputInfo": {
            "patchV": {
                "type": "patchVelocity",
                "patches": ["inout"],
                "flowAxis": "x",
                "normalAxis": "y",
                "components": ["solver", "function"],
            }
        },
        "adjEqnOption": {
            "gmresRelTol": 1e-7,
            "gmresAbsTol": 1e-12,
            "pcFillLevel": 1,
            "jacMatReOrdering": "rcm",
        },
    }
    return files, options, {"patchV": [10.0, 2.0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="engine", required=True)
    xfoil = sub.add_parser("xfoil")
    xfoil.add_argument("--executable", required=True)
    xfoil.add_argument("--version", default="6.99", choices=("6.9", "6.99"))
    xfoil.add_argument("--timeout", type=float, default=120)
    foam = sub.add_parser("dafoam")
    foam.add_argument("--python", required=True)
    foam.add_argument("--python-version", required=True)
    foam.add_argument("--package-root", required=True)
    foam.add_argument("--dependency-root", action="append", default=[])
    foam.add_argument("--timeout", type=float, default=1800)
    foam.add_argument("--half-panels", type=int, default=64)
    foam.add_argument("--radial-cells", type=int, default=48)
    foam.add_argument("--fd-step", type=float, default=1e-3)
    args = parser.parse_args()
    if args.engine == "xfoil":
        executable = pin_energy_executable(
            args.executable,
            version=args.version,
            license_id="GPL-2.0-or-later",
            source_url="https://web.mit.edu/drela/Public/web/xfoil/",
        )
        result = run_xfoil_polar(
            executable,
            naca0012_coordinates(),
            tuple(
                XFOILOperatingPoint(alpha, 1e6, 0.0, 9.0) for alpha in (-4.0, 0.0, 4.0)
            ),
            timeout_per_point=args.timeout,
        )
        print(
            json.dumps(
                {
                    "engine": "XFOIL",
                    "geometry_source": GEOMETRY_SOURCE,
                    "converged": result.converged,
                    "artifact_id": result.artifact.artifact_id,
                    "points": [
                        {
                            "operating_point": asdict(p.operating_point),
                            "coefficients": asdict(p.coefficients)
                            if p.coefficients is not None
                            else None,
                            "seconds": p.run.elapsed_seconds,
                            "converged": p.converged,
                            "failure": p.failure_reason,
                            "artifact_id": p.artifact.artifact_id,
                        }
                        for p in result.points
                    ],
                },
                indent=2,
            )
        )
        if not result.converged:
            raise SystemExit(1)
    else:
        if not math.isfinite(args.fd_step) or args.fd_step <= 0:
            parser.error("--fd-step must be positive and finite")
        python = pin_energy_executable(
            args.python, version=args.python_version, license_id="PSF-2.0"
        )
        runtime = pin_dafoam_runtime(
            python,
            args.package_root,
            license_id="GPL-3.0-or-later",
            dependency_roots=args.dependency_root,
        )
        files, options, design = naca0012_dafoam_case(args.half_panels, args.radial_cells)
        result = run_dafoam(
            runtime,
            case_files=files,
            options=options,
            design=design,
            geometry_source=GEOMETRY_SOURCE,
            total_adjoint=True,
            timeout=args.timeout,
        )
        print(
            json.dumps(
                {
                    "engine": "DAFoam",
                    "accepted": result.accepted,
                    "failure": result.failure_reason,
                    "seconds": result.run.elapsed_seconds,
                    "artifact_id": result.artifact.artifact_id,
                    "functions": dict(result.functions),
                    "totals": [asdict(t) for t in result.total_derivatives],
                    "adjoints": [asdict(a) for a in result.adjoints],
                },
                indent=2,
            )
        )
        result.require_acceptance()
        perturbed = []
        for sign in (-1, 1):
            values = {
                "patchV": [design["patchV"][0], design["patchV"][1] + sign * args.fd_step]
            }
            run = run_dafoam(
                runtime,
                case_files=files,
                options=options,
                design=values,
                geometry_source=GEOMETRY_SOURCE,
                timeout=args.timeout,
            ).require_acceptance()
            perturbed.append(dict(run.functions))
        errors = {}
        for total in result.total_derivatives:
            fd = (perturbed[1][total.function] - perturbed[0][total.function]) / (
                2 * args.fd_step
            )
            adjoint = total.values[1]
            errors[total.function] = {
                "adjoint_alpha": adjoint,
                "central_fd_alpha": fd,
                "absolute_error": abs(fd - adjoint),
            }
        print(json.dumps({"same_case_central_difference": errors}, indent=2))
        if any(
            v["absolute_error"] > 1e-4 + 0.03 * abs(v["central_fd_alpha"])
            for v in errors.values()
        ):
            raise SystemExit(
                "DAFoam total adjoint disagrees with independent central finite differences."
            )


if __name__ == "__main__":
    main()
