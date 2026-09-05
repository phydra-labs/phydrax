#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import importlib.util
import uuid

import pytest

from phydrax.interchange.helics import HelicsChannel, HelicsValueSession


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("helics") is None,
    reason="requires the optional real HELICS3 runtime",
)


def test_real_two_federate_typed_delivery_and_time_grants():
    key = "phydrax-" + uuid.uuid4().hex
    with HelicsValueSession(
        key + "-source",
        license_id="LicenseRef-Caller-Qualification",
        federate_count=2,
        publications=(
            HelicsChannel(key + "/power", "double", "W"),
            HelicsChannel(key + "/enabled", "boolean"),
        ),
    ) as source:
        with HelicsValueSession(
            key + "-sink",
            license_id="LicenseRef-Caller-Qualification",
            broker=source.broker_address,
            subscriptions=(
                HelicsChannel("power", "double", "W", target=key + "/power"),
                HelicsChannel("enabled", "boolean", target=key + "/enabled"),
            ),
        ) as sink:
            source.enter_execution_async()
            sink.enter_execution_async()
            source.complete_execution()
            sink.complete_execution()
            before = sink.read_values()
            assert all(not sample.has_value and sample.value is None for sample in before)
            source.publish({key + "/power": 125.0, key + "/enabled": True})
            source.request_time_async(1)
            sink.request_time_async(1)
            source_grant = source.complete_time()
            sink_grant = sink.complete_time()
            assert 0 <= sink_grant.granted_time <= sink_grant.requested_time
            assert 0 <= source_grant.granted_time <= source_grant.requested_time
            values = {sample.channel: sample for sample in sink.read_values()}
            assert values["power"].has_value and values["power"].value == 125.0
            assert values["enabled"].value is True
            assert values["power"].last_update_time <= values["power"].granted_time
            assert sink.artifact.status == "complete"
        assert sink.closed
    assert source.closed


def test_real_helics_does_not_coerce_types_or_advance_backwards():
    key = "phydrax-" + uuid.uuid4().hex
    with HelicsValueSession(
        key,
        license_id="LicenseRef-Caller-Qualification",
        publications=(HelicsChannel(key + "/switch", "boolean"),),
    ) as session:
        session.enter_execution()
        with pytest.raises(TypeError):
            session.publish({key + "/switch": 1})
        with pytest.raises(ValueError):
            session.advance(-1)
        session.publish({key + "/switch": True})
        grant = session.advance(1)
        assert grant.granted_time == 1 and not grant.terminated
    with pytest.raises(RuntimeError):
        session.publish({key + "/switch": False})
