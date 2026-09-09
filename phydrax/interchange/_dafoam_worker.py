#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Private standalone DAFoam 4.0.3 worker; needs no PHYDRAX installation.

Only this worker produces aerodynamic-result.json. Native engine convergence
and OpenMDAO's reverse total derivatives are the source of the reported data.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_runtime(runtime):
    pins = dict(runtime["implementation_files"])
    for path, expected in pins.items():
        if _file_hash(path) != expected:
            raise ValueError("DAFoam runtime implementation changed: " + path)
    return _fingerprint(pins)


def main():
    request = json.loads(Path("aerodynamic-request.json").read_text())
    runtime = request["runtime"]
    runtime_sha = _verify_runtime(runtime)
    for path, expected in request["case_sha256"].items():
        if _file_hash(path) != expected:
            raise ValueError("Staged DAFoam case identity mismatch: " + path)

    import dafoam
    import numpy as np
    import openmdao.api as om
    from dafoam import PYDAFOAM
    from dafoam.mphys.mphys_dafoam import DAFoamFunctions, DAFoamSolver
    from mpi4py import MPI
    from petsc4py import PETSc

    if Path(dafoam.__file__).resolve().parent != Path(runtime["package_root"]):
        raise ValueError("Imported DAFoam package is not the pinned package root.")
    if MPI.COMM_WORLD.size != 1:
        raise ValueError(
            "The serialized aerodynamic profile is serial, not MPI-distributed."
        )

    def array_hash(array):
        value = np.ascontiguousarray(array, dtype=np.float64)
        return hashlib.sha256(value.tobytes()).hexdigest()

    solver = PYDAFOAM(options=request["options"], comm=MPI.COMM_WORLD)
    if solver.version != runtime["version"]:
        raise ValueError("The imported DAFoam engine has a different version.")
    design = {
        name: np.asarray(values, dtype=float)
        for name, values in request["design"].items()
    }
    for name, values in design.items():
        info = request["options"]["inputInfo"][name]
        if values.size != solver.solver.getInputSize(name, info["type"]):
            raise ValueError("Wrong native DAFoam input-vector length: " + name)
    solver.set_solver_input(design)

    def mesh_hash():
        coords = np.empty(solver.xv0.size)
        solver.solver.getOFMeshPoints(coords)
        if not np.isfinite(coords).all():
            raise ValueError("DAFoam realized mesh has nonfinite coordinates.")
        return array_hash(coords)

    mesh_sha = mesh_hash()
    design_sha = _fingerprint(request["design"])
    result = {
        "request_id": request["request_id"],
        "runtime_sha256": runtime_sha,
        "functions": {},
        "total_derivatives": [],
        "adjoints": [],
        "mesh_accepted": bool(solver.solver.checkMesh() == 1),
        "state_accepted": False,
        "state_sha256": "",
        "mesh_sha256": mesh_sha,
        "design_sha256": design_sha,
        "failure_reason": "",
        "convergence_options": {
            name: solver.getOption(name)
            for name in (
                "primalMinResTol",
                "primalMinResTolDiff",
                "checkMeshThreshold",
                "adjEqnOption",
            )
        },
        "engine_version": solver.version,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "openmdao", "mphys", "mpi4py", "petsc4py")
        },
        "petsc_version": PETSc.Sys.getVersion(),
        "mpi_library": MPI.Get_library_version(),
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "WM_PROJECT_VERSION",
                "WM_PROJECT_DIR",
                "FOAM_LIBBIN",
                "FOAM_USER_LIBBIN",
                "DAFOAM_ROOT_PATH",
                "PETSC_DIR",
                "PETSC_ARCH",
                "LD_LIBRARY_PATH",
                "DYLD_LIBRARY_PATH",
                "PYTHONPATH",
                "OMP_NUM_THREADS",
            )
        },
    }
    if not result["mesh_accepted"]:
        result["failure_reason"] = (
            "Native DAFoam checkMesh rejected the declared design mesh."
        )
    else:
        problem = om.Problem(reports=False, comm=MPI.COMM_WORLD)
        dvs = om.IndepVarComp()
        for name, values in design.items():
            dvs.add_output(name, val=values)
        problem.model.add_subsystem("design", dvs, promotes=["*"])
        state_component = DAFoamSolver(solver=solver)
        problem.model.add_subsystem("flow", state_component, promotes=["*"])
        problem.model.add_subsystem(
            "responses", DAFoamFunctions(solver=solver), promotes=["*"]
        )
        problem.setup(mode="rev")
        try:
            problem.run_model()
        except om.AnalysisError as error:
            # Preserve actual rejected native solve evidence, not penalty values.
            result["failure_reason"] = str(error)
        else:
            states = solver.getStates()
            result["state_accepted"] = bool(
                solver.primalFail == 0 and np.isfinite(states).all()
            )
            result["state_sha256"] = array_hash(states)
            result["mesh_sha256"] = mesh_hash()
            if not result["state_accepted"]:
                result["failure_reason"] = (
                    "Native primalFail or nonfinite state rejected the primal."
                )
            else:
                functions = {
                    name: float(problem.get_val(name).item())
                    for name in request["options"]["function"]
                }
                if not all(np.isfinite(value) for value in functions.values()):
                    result["state_accepted"] = False
                    result["failure_reason"] = (
                        "Native function extraction produced nonfinite values."
                    )
                else:
                    result["functions"] = functions
                    if request["total_adjoint"]:
                        for name in functions:
                            failure = ""
                            totals = None
                            try:
                                # One output at a time binds each convergence record to its
                                # exact adjoint. There is no intervening nonlinear solve.
                                totals = problem.compute_totals(
                                    of=[name], wrt=list(design), return_format="flat_dict"
                                )
                            except om.AnalysisError as error:
                                failure = str(error)
                            unchanged = (
                                array_hash(solver.getStates()) == result["state_sha256"]
                                and mesh_hash() == result["mesh_sha256"]
                                and all(
                                    np.array_equal(problem.get_val(key), values)
                                    for key, values in design.items()
                                )
                            )
                            if not unchanged:
                                failure = (
                                    failure
                                    or "State, mesh or design changed during total-adjoint evaluation."
                                )
                            ksp = solver.ksp
                            reason = (
                                int(ksp.getConvergedReason()) if ksp is not None else 0
                            )
                            residual = (
                                float(ksp.getResidualNorm()) if ksp is not None else None
                            )
                            iterations = (
                                int(ksp.getIterationNumber()) if ksp is not None else 0
                            )
                            if (
                                reason <= 0
                                or residual is None
                                or not np.isfinite(residual)
                            ):
                                failure = (
                                    failure
                                    or "PETSc did not report a converged finite adjoint solve."
                                )
                            if totals is None:
                                failure = (
                                    failure or "Native total derivative was not returned."
                                )
                            elif not all(np.isfinite(v).all() for v in totals.values()):
                                failure = failure or "Nonfinite native total derivative."
                            adjoint = solver.vec2Array(state_component.psi)
                            if not np.isfinite(adjoint).all():
                                failure = failure or "Nonfinite native adjoint vector."
                            result["adjoints"].append(
                                {
                                    "function": name,
                                    "accepted": not failure,
                                    "state_sha256": result["state_sha256"],
                                    "mesh_sha256": result["mesh_sha256"],
                                    "design_sha256": design_sha,
                                    "adjoint_sha256": array_hash(adjoint),
                                    "linear_iterations": iterations,
                                    "linear_residual_norm": residual
                                    if residual is not None and np.isfinite(residual)
                                    else None,
                                    "petsc_converged_reason": reason,
                                    "failure_reason": failure,
                                }
                            )
                            if failure:
                                result["failure_reason"] = failure
                                result["total_derivatives"] = []
                                break
                            for variable in design:
                                result["total_derivatives"].append(
                                    {
                                        "function": name,
                                        "design_variable": variable,
                                        "values": totals[name, variable]
                                        .reshape(-1)
                                        .tolist(),
                                    }
                                )
        problem.cleanup()
    _verify_runtime(runtime)
    Path("aerodynamic-result.json").write_text(
        json.dumps(result, allow_nan=False, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
