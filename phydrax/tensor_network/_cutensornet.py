# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Shape-only cuTensorNet planning, isolated from device-local execution."""

from __future__ import annotations

import ctypes
import os
from multiprocessing import get_context
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

from ._contraction_provider import ContractionPathPlanner


if TYPE_CHECKING:
    from ._topology import ContractionStructure


def _planner_worker(
    connection: Connection, recipe: str, request: ContractionPathPlanner, count: int
) -> None:
    # cutensornetCreate otherwise associates the current GPU. The spawned worker
    # has no inherited CUDA state; hide devices before loading its selected DSO.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        from .._cuda_native import cuda_component_function, load_cuda_component

        identity = load_cuda_component("cutensornet")
        function = cuda_component_function("cutensornet", "phydrax_cutensornet_plan")
        capacity = max(1, 2 * (count - 1))
        pairs = (ctypes.c_int32 * capacity)()
        pair_count = ctypes.c_int64()
        from importlib.metadata import version

        error = ctypes.create_string_buffer(4096)
        status = function(
            recipe.encode("ascii"),
            request.maximum_workspace_bytes,
            request.maximum_search_iterations,
            request.seed,
            pairs,
            capacity,
            ctypes.byref(pair_count),
            error,
            len(error),
        )
        if status != 0:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        if pair_count.value != count - 1:
            raise ValueError("cuTensorNet planner returned an incomplete binary path.")
        path = tuple(
            (pairs[2 * index], pairs[2 * index + 1]) for index in range(pair_count.value)
        )
        connection.send(
            (
                path,
                (
                    identity.version,
                    identity.build_hash,
                    version("cutensor-cu13"),
                    version("cutensornet-cu13"),
                ),
                None,
            )
        )
    except BaseException as error:
        connection.send((None, None, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def plan_cutensornet_path(
    structure: ContractionStructure, dtype: str, request: ContractionPathPlanner, /
) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    ids = {label: index for index, label in enumerate(structure.labels)}
    tokens = [
        {"float32": 0, "float64": 1, "complex64": 2, "complex128": 3}[dtype],
        len(structure.operands),
    ]
    for operand in structure.operands:
        unique = tuple(dict.fromkeys(leg.label for leg in operand.legs))
        tokens.extend(
            (
                len(unique),
                *(structure.dimension(mode) for mode in unique),
                *(ids[mode] for mode in unique),
            )
        )
    tokens.extend(
        (
            len(structure.outputs),
            *(structure.dimension(mode) for mode in structure.outputs),
            *(ids[mode] for mode in structure.outputs),
            0,
        )
    )
    recipe = " ".join(str(value) for value in tokens)
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(
        target=_planner_worker, args=(sender, recipe, request, len(structure.operands))
    )
    worker.start()
    sender.close()
    try:
        # The deadline includes interpreter/import/library initialization, not
        # just graph search. No unbounded readiness handshake precedes it.
        if not receiver.poll(request.maximum_planning_seconds):
            raise TimeoutError(
                "cuTensorNet path planning exceeded maximum_planning_seconds."
            )
        path, build, error = receiver.recv()
        if error is not None:
            raise RuntimeError(f"cuTensorNet path planning failed: {error}")
        if path is None or build is None:
            raise RuntimeError("cuTensorNet planner returned no path/provenance.")
        return path, build
    finally:
        receiver.close()
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=1.0)
        if worker.is_alive():
            worker.kill()
            worker.join()
