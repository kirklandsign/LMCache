# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the Blackhole connector and its adapter.

The Blackhole connector is a no-op sink used in tests and benchmarks: every
read returns "miss", every write is dropped on the floor. These tests pin
that behavior so the connector keeps working as a stable mocking target.
"""

# Standard
from unittest.mock import MagicMock
import asyncio

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.connector.blackhole_adapter import (
    BlackholeConnectorAdapter,
)
from lmcache.v1.storage_backend.connector.blackhole_connector import (
    BlackholeConnector,
)


def _make_key(chunk_hash: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.float16,
    )


class TestBlackholeConnector:
    def test_exists_is_always_false(self):
        conn = BlackholeConnector()
        key = _make_key()
        assert asyncio.run(conn.exists(key)) is False
        assert conn.exists_sync(key) is False

    def test_get_returns_none(self):
        conn = BlackholeConnector()
        assert asyncio.run(conn.get(_make_key())) is None

    def test_put_is_a_noop(self):
        conn = BlackholeConnector()
        # Put returns nothing and must not touch the memory object.
        memory_obj = MagicMock(name="memory_obj")
        assert asyncio.run(conn.put(_make_key(), memory_obj)) is None
        memory_obj.assert_not_called()

    def test_close_is_safe_to_call(self):
        conn = BlackholeConnector()
        # close() just logs and returns; no exception should escape.
        assert asyncio.run(conn.close()) is None


class TestBlackholeConnectorAdapter:
    def test_schema_matches_blackhole_prefix(self):
        adapter = BlackholeConnectorAdapter()
        assert adapter.schema == "blackhole://"
        assert adapter.can_parse("blackhole://anything") is True
        assert adapter.can_parse("redis://foo") is False

    def test_create_connector_returns_blackhole(self):
        adapter = BlackholeConnectorAdapter()
        # The adapter ignores the context entirely, so a plain MagicMock
        # is enough to drive create_connector.
        context = MagicMock(url="blackhole://x")
        conn = adapter.create_connector(context)
        assert isinstance(conn, BlackholeConnector)


@pytest.mark.parametrize("chunk_hash", [0, 1, 42, 2**62])
def test_exists_remains_false_for_arbitrary_keys(chunk_hash: int):
    conn = BlackholeConnector()
    assert conn.exists_sync(_make_key(chunk_hash)) is False
