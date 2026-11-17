import gc
from threading import Event, Thread

import numpy as np
import pytest

from phydrax._collection_scope import collection_scope
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellBlock, CellMesh
from phydrax.meshing import certify_cell_mesh


def test_scope_restores_thresholds_on_success_and_exception() -> None:
    previous = gc.get_threshold()
    try:
        gc.set_threshold(701, 13, 17)
        with collection_scope():
            young, middle, old = gc.get_threshold()
            assert young >= 50_000
            assert (middle, old) == (13, 17)
        assert gc.get_threshold() == (701, 13, 17)
        with pytest.raises(RuntimeError, match="failure"):
            with collection_scope():
                raise RuntimeError("failure")
        assert gc.get_threshold() == (701, 13, 17)
    finally:
        gc.set_threshold(*previous)


def test_nested_scopes_preserve_higher_caller_threshold() -> None:
    previous = gc.get_threshold()
    try:
        gc.set_threshold(100_000, 12, 15)
        with collection_scope():
            outer = gc.get_threshold()
            assert outer == (100_000, 12, 15)
            with collection_scope():
                assert gc.get_threshold() == outer
            assert gc.get_threshold() == outer
        assert gc.get_threshold() == (100_000, 12, 15)
    finally:
        gc.set_threshold(*previous)


def test_disabled_collection_is_untouched() -> None:
    enabled = gc.isenabled()
    previous = gc.get_threshold()
    gc.disable()
    try:
        with collection_scope():
            assert not gc.isenabled()
            assert gc.get_threshold() == previous
        assert not gc.isenabled()
        assert gc.get_threshold() == previous
    finally:
        if enabled:
            gc.enable()


def test_overlapping_scopes_restore_after_last_exit() -> None:
    previous = gc.get_threshold()
    entered = Event()
    release = Event()
    try:
        gc.set_threshold(703, 13, 17)
        with collection_scope():

            def run() -> None:
                with collection_scope():
                    entered.set()
                    if not release.wait(10):
                        raise RuntimeError("scope release timed out")

            worker = Thread(target=run)
            worker.start()
            assert entered.wait(10)
        assert gc.get_threshold() == (50_000, 13, 17)
        release.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert gc.get_threshold() == (703, 13, 17)
    finally:
        release.set()
        gc.set_threshold(*previous)


def test_certification_identity_is_unchanged_when_collection_is_disabled() -> None:
    mesh = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        (CellBlock("triangle", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)),),
    )
    contract = SpatialCoordinateContract.si()
    scoped = certify_cell_mesh(mesh, contract)
    enabled = gc.isenabled()
    gc.disable()
    try:
        unscoped = certify_cell_mesh(mesh, contract)
    finally:
        if enabled:
            gc.enable()
    assert scoped.result_id == unscoped.result_id
    assert scoped.audit.report_id == unscoped.audit.report_id
