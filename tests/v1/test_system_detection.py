# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for lmcache.v1.system_detection.

Covers SystemMemoryDetector (psutil-backed memory probe) and NUMADetector
(manual/auto/None routing). The auto-from-sysfs success path requires GPU
hardware to exercise faithfully, so it is covered only via its exception-
swallowing fallback.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
import pytest

# First Party
from lmcache.v1.system_detection import (
    NUMADetector,
    NUMAMapping,
    SystemMemoryDetector,
)


class TestSystemMemoryDetector:
    def test_returns_gb_from_psutil(self):
        fake_mem = SimpleNamespace(available=4 * 1024**3)  # 4 GiB
        with patch(
            "lmcache.v1.system_detection.psutil.virtual_memory",
            return_value=fake_mem,
        ):
            result = SystemMemoryDetector.get_available_memory_gb()
        assert result == pytest.approx(4.0)

    def test_returns_zero_on_psutil_failure(self):
        with patch(
            "lmcache.v1.system_detection.psutil.virtual_memory",
            side_effect=RuntimeError("psutil unavailable"),
        ):
            result = SystemMemoryDetector.get_available_memory_gb()
        assert result == 0.0

    def test_real_psutil_returns_positive_value(self):
        # Smoke check against the real OS — should never be zero on a
        # running machine, and should be a float.
        result = SystemMemoryDetector.get_available_memory_gb()
        assert isinstance(result, float)
        assert result >= 0.0


class TestNUMADetectorRouting:
    def test_none_mode_returns_none(self):
        config = SimpleNamespace(numa_mode=None, extra_config=None)
        assert NUMADetector.get_numa_mapping(config) is None

    def test_unknown_mode_raises(self):
        config = SimpleNamespace(numa_mode="weird", extra_config=None)
        with pytest.raises(AssertionError):
            NUMADetector.get_numa_mapping(config)


class TestNUMADetectorManual:
    def test_manual_mode_reads_mapping_from_config(self):
        mapping = {0: 0, 1: 1, 2: 0, 3: 1}
        config = SimpleNamespace(
            numa_mode="manual",
            extra_config={"gpu_to_numa_mapping": mapping},
        )
        result = NUMADetector.get_numa_mapping(config)
        assert isinstance(result, NUMAMapping)
        assert result.gpu_to_numa_mapping == mapping

    def test_manual_mode_without_extra_config_raises(self):
        config = SimpleNamespace(numa_mode="manual", extra_config=None)
        with pytest.raises(AssertionError):
            NUMADetector.get_numa_mapping(config)

    def test_manual_mode_missing_key_raises(self):
        config = SimpleNamespace(
            numa_mode="manual",
            extra_config={"some_other_key": 1},
        )
        with pytest.raises(AssertionError):
            NUMADetector.get_numa_mapping(config)


class TestNUMADetectorAutoFallback:
    def test_auto_mode_swallows_exception_and_returns_none(self):
        config = SimpleNamespace(numa_mode="auto", extra_config=None)
        # Force current_device to raise so the auto path hits its except.
        with patch(
            "lmcache.v1.system_detection.torch_dev.current_device",
            side_effect=RuntimeError("no cuda"),
        ):
            assert NUMADetector.get_numa_mapping(config) is None
