"""Python-call versus C-only cost of the planar Ruppert rows, interleaved.

Run from the repository root with PHYDRAX_MESHCORE_LIBRARY set to the library
under test (and PYTHONPATH=.): ``python native/meshcore/tools/planar_probe.py [calls]``.
Reports median/min microseconds and mean minor page faults per call for the
public ``constrained_delaunay_2d`` and for the bare C entry plus free.
"""

import ctypes
import os
import resource
import statistics
import sys
import time

import numpy as np

from phydrax import _meshcore


def _rows() -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    square_segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    inputs = os.path.join("benchmarks", "native_meshing_inputs")
    ring = np.load(os.path.join(inputs, "q2ring_pts.npy"), allow_pickle=False)
    ring_segments = np.load(os.path.join(inputs, "q2ring_seg.npy"), allow_pickle=False)
    return [
        ("ruppert square 1e3", square, square_segments, 1.5 / 1000),
        ("ruppert ring-slit 1e4", ring, ring_segments, 1.5 / 10000),
        ("ruppert square 1e4", square, square_segments, 1.5 / 10000),
    ]


def main() -> None:
    calls = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    library = ctypes.CDLL(os.environ["PHYDRAX_MESHCORE_LIBRARY"])
    pointer = ctypes.c_void_p
    entry = library.phx_mc_constrained_delaunay_2d
    entry.argtypes = [
        ctypes.c_int64, pointer, ctypes.c_int64, pointer, ctypes.c_int64, pointer,
        ctypes.c_int32, ctypes.c_double, ctypes.c_double, ctypes.POINTER(pointer),
    ]
    entry.restype = ctypes.c_int32
    library.phx_mc_mesh_free.argtypes = [pointer]

    def faults() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_minflt

    for name, points, segments, area in _rows():
        samples: dict[str, list[float]] = {"python": [], "c": []}
        fault_counts: dict[str, list[int]] = {"python": [], "c": []}

        def python_call() -> None:
            _meshcore.constrained_delaunay_2d(
                points, segments, keep_convex_hull=True, min_angle=20.0, max_area=area
            )

        def c_call() -> None:
            handle = pointer()
            status = entry(
                points.shape[0], points.ctypes.data, segments.shape[0], segments.ctypes.data,
                0, None, 1, 20.0, area, ctypes.byref(handle),
            )
            if status != 0:
                raise RuntimeError(f"status {status}")
            library.phx_mc_mesh_free(handle)

        for index in range(calls + 5):
            for kind in (("python", "c") if index % 2 == 0 else ("c", "python")):
                call = python_call if kind == "python" else c_call
                before = faults()
                start = time.perf_counter()
                call()
                elapsed = time.perf_counter() - start
                if index >= 5:
                    samples[kind].append(elapsed * 1e6)
                    fault_counts[kind].append(faults() - before)
        python = statistics.median(samples["python"])
        c = statistics.median(samples["c"])
        print(
            f"{name}: python {python:.1f}/{min(samples['python']):.1f} us "
            f"(faults {statistics.mean(fault_counts['python']):.1f}), "
            f"C {c:.1f}/{min(samples['c']):.1f} us "
            f"(faults {statistics.mean(fault_counts['c']):.1f}), "
            f"python - C median {python - c:.1f} us"
        )


if __name__ == "__main__":
    main()
