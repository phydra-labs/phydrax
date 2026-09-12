import json

import numpy as np
import pytest

import phydrax as phx


def _limits():
    return phx.interchange.ResourceLimits(
        max_bytes=100_000,
        max_depth=16,
        max_nodes=1_000,
        max_attributes=100,
        max_losses=100,
    )


def test_bilby_json_import_is_numeric_bounded_and_event_ready(tmp_path):
    payload = {
        "label": "event",
        "sampler": "dynesty",
        "version": "2.8.0",
        "use_ratio": True,
        "log_evidence": 3.25,
        "search_parameter_keys": ["mass", "phase"],
        "posterior": {
            "__dataframe__": True,
            "content": {
                "mass": [20.0, 21.0, 22.0],
                "phase": [0.1, 0.2, 0.3],
                "weights": [2.0, 3.0, 5.0],
                "log_prior": [-2.0, -2.0, -2.0],
                "log_likelihood": [1.0, 2.0, 3.0],
            },
        },
        "priors": {"__prior_dict__": True, "content": {}},
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    imported = phx.interchange.read_bilby_result_json(
        path,
        trusted_root=tmp_path,
        limits=_limits(),
    )
    event = imported.to_event_posterior(
        event_id="event",
        parameterization_id="mass-phase",
        likelihood_id="likelihood",
        provider_id="provider",
    )

    assert imported.parameter_names == ("mass", "phase")
    assert imported.evidence_kind == "noise-relative"
    np.testing.assert_allclose(imported.log_evidence, 3.25)
    np.testing.assert_allclose(np.sum(np.exp(imported.log_weights)), 1.0)
    assert event.posterior.samples["mass"].shape == (3,)


def test_bilby_json_import_rejects_executable_posterior_markers(tmp_path):
    payload = {
        "search_parameter_keys": ["x"],
        "posterior": {
            "__dataframe__": True,
            "content": {"x": [1.0], "__function__": ["unsafe"]},
        },
    }
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="executable object marker"):
        phx.interchange.read_bilby_result_json(
            path,
            trusted_root=tmp_path,
            limits=_limits(),
        )


@pytest.mark.parametrize(
    ("name", "value", "error"),
    (
        ("search_parameter_keys", ["x", "x"], ValueError),
        ("use_ratio", "true", TypeError),
    ),
)
def test_bilby_json_import_rejects_ambiguous_metadata(tmp_path, name, value, error):
    payload = {
        "search_parameter_keys": ["x"],
        "posterior": {"x": [1.0]},
        name: value,
    }
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(error):
        phx.interchange.read_bilby_result_json(
            path,
            trusted_root=tmp_path,
            limits=_limits(),
        )


def test_bilby_json_import_rejects_duplicate_object_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"search_parameter_keys":["x"],"search_parameter_keys":["x"],'
        '"posterior":{"x":[1.0]}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        phx.interchange.read_bilby_result_json(
            path,
            trusted_root=tmp_path,
            limits=_limits(),
        )
