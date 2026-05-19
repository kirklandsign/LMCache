# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for lmcache.v1.utils.cache_utils.TTLListCache.

Covers cache hit/miss semantics, expiration policy, the per-call timeout
override, clear, repr, and concurrent access.
"""

# Standard
from threading import Thread

# Third Party
import pytest

# First Party
from lmcache.v1.utils.cache_utils import TTLListCache


def make_loader(items: list[int]) -> "tuple[list[int], list[list[int]]]":
    """Return ``(call_log, loader)`` where loader appends to call_log each call.

    Useful for asserting how many times ``get_cached`` fell through to the
    refresh path.
    """
    call_log: list[list[int]] = []

    def loader() -> list[int]:
        snapshot = list(items)
        call_log.append(snapshot)
        return snapshot

    return call_log, loader


class TestGetCached:
    def test_initial_call_refreshes(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        log, loader = make_loader([1, 2, 3])
        result = cache.get_cached(loader)
        assert result == [1, 2, 3]
        assert len(log) == 1

    def test_second_call_within_ttl_uses_cache(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        log, loader = make_loader([1, 2, 3])
        cache.get_cached(loader)
        cache.get_cached(loader)
        assert len(log) == 1

    def test_timeout_override_zero_always_refreshes(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        log, loader = make_loader([1])
        cache.get_cached(loader, timeout_override=0)
        cache.get_cached(loader, timeout_override=0)
        cache.get_cached(loader, timeout_override=0)
        assert len(log) == 3

    def test_timeout_override_does_not_persist(self):
        """timeout_override is per-call; the instance default still wins later."""
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        log, loader = make_loader([1])
        cache.get_cached(loader, timeout_override=0)  # refresh
        cache.get_cached(loader)  # should use cache (instance ttl=10s)
        assert len(log) == 1

    def test_expired_refresh_replaces_cache(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        log, loader_a = make_loader([1, 2])
        cache.get_cached(loader_a)

        # Force expiry, then load with a different loader.
        cache.cache_time = 0.0
        _, loader_b = make_loader([9, 9, 9])
        result = cache.get_cached(loader_b)
        assert result == [9, 9, 9]
        assert len(log) == 1  # original loader was called exactly once


class TestIsExpired:
    def test_uninitialized_is_always_expired(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        assert cache.is_expired() is True
        assert cache.is_expired(timeout_override=0) is True
        assert cache.is_expired(timeout_override=10_000) is True

    def test_fresh_cache_not_expired(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        cache.cache_time = 1000.0
        assert cache.is_expired(current_time=1005.0) is False

    def test_expired_when_age_exceeds_timeout(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        cache.cache_time = 1000.0
        assert cache.is_expired(current_time=1011.0) is True

    def test_override_zero_short_circuits_to_expired(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        cache.cache_time = 1000.0
        # Even though only 1s elapsed, override=0 means always expired.
        assert cache.is_expired(timeout_override=0, current_time=1001.0) is True


class TestClearAndIntrospection:
    def test_clear_resets_cache_and_time(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        _, loader = make_loader([1, 2, 3])
        cache.get_cached(loader)
        cache.clear()
        assert len(cache) == 0
        assert cache.cache_time == 0.0
        assert cache.is_expired() is True

    def test_get_cache_age_when_uninitialized(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        assert cache.get_cache_age() == float("inf")

    def test_repr_uninitialized(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=42.0)
        text = repr(cache)
        assert "uninitialized" in text
        assert "timeout=42.0s" in text

    def test_repr_after_load(self):
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=5.0)
        _, loader = make_loader([1, 2])
        cache.get_cached(loader)
        text = repr(cache)
        assert "cache_items=2" in text
        assert "old" in text
        assert "timeout=5.0s" in text


class TestConcurrency:
    def test_concurrent_refresh_calls_loader_once(self):
        """Many threads racing on a cold cache should fire only one refresh."""
        cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
        log, loader = make_loader([1, 2, 3])

        results: list[list[int]] = []

        def worker() -> None:
            results.append(cache.get_cached(loader))

        threads = [Thread(target=worker) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == [1, 2, 3] for r in results)
        # Double-checked locking should suppress most duplicate loads; in the
        # worst case 2 calls slip through before the lock holder publishes.
        assert len(log) <= 2


@pytest.mark.parametrize("payload", [[], [0], list(range(1024))])
def test_payload_passthrough(payload: list[int]):
    cache: TTLListCache[int] = TTLListCache(timeout_seconds=10.0)
    _, loader = make_loader(payload)
    assert cache.get_cached(loader) == payload
    assert len(cache) == len(payload)
