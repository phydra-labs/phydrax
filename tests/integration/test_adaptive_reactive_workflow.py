#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import contextlib
import io
import runpy


def test_growth_amr_superquadric_wall_and_monolithic_workflows_compose():
    paths = (
        "examples/growing_reactive_particle_pool.py",
        "examples/adaptive_catalyst_pellet.py",
        "examples/superquadric_triangle_wall.py",
        "examples/monolithic_reactive_cfd_dem.py",
    )
    outputs = []
    for path in paths:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            namespace = runpy.run_path(path)
        outputs.append(namespace["result"])
    assert all(bool(value.successful) for value in outputs)
