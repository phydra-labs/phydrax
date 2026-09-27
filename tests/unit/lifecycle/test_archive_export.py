#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path

import numpy as np
import pytest

from phydrax._identity import NumericRevision, SemanticProvenance
from phydrax.lifecycle._archive import create, export, LifecycleArchive
from phydrax.lifecycle._models import RevisionLineage


def _archive(tmp_path: Path, arrays: dict[str, np.ndarray]) -> LifecycleArchive:
    revision = RevisionLineage(
        NumericRevision(SemanticProvenance({"kind": "lifecycle-export-test"}), arrays)
    )
    return create(tmp_path / "source.zip", manifest=revision, arrays=arrays)


def test_npz_export_round_trips_every_selected_payload(tmp_path: Path) -> None:
    arrays = {"u": np.asarray((1.0, 2.0)), "v": np.asarray((3.0,))}
    output = export(_archive(tmp_path, arrays), tmp_path / "out.npz", format="npz")
    with np.load(output) as exported:
        assert sorted(exported.files) == ["metadata", "u", "v"]
        np.testing.assert_array_equal(exported["u"], arrays["u"])
        np.testing.assert_array_equal(exported["v"], arrays["v"])


def test_npz_export_never_silently_drops_a_payload_named_allow_pickle(
    tmp_path: Path,
) -> None:
    arrays = {"allow_pickle": np.asarray((1.0, 2.0)), "u": np.asarray((3.0,))}
    archive = _archive(tmp_path, arrays)
    with pytest.raises(TypeError, match="allow_pickle"):
        export(archive, tmp_path / "out.npz", format="npz")
    assert not (tmp_path / "out.npz").exists()
