# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the small public surface of lmcache.v1.compute.blend.

Covers:
* ``LMCBlendCommonMetadata`` / ``LMCBlendMetadata`` dataclass behavior,
  including ``clean()`` resetting all tensor fields.
* ``LMCBlenderBuilder.get_or_create`` / ``get`` registry semantics, with
  ``LMCBlender`` and ``VLLMModelTracker.get_model`` mocked out so the test
  does not need a real vLLM model.
"""

# Standard
from unittest.mock import MagicMock, patch
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.compute.blend.metadata import (
    LMCBlendCommonMetadata,
    LMCBlendMetadata,
)
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder


class TestLMCBlendCommonMetadata:
    def test_required_fields_only(self):
        meta = LMCBlendCommonMetadata(check_layers=[0, 4, 8])
        assert meta.check_layers == [0, 4, 8]
        assert meta.recomp_ratios is None
        assert meta.thresholds is None

    def test_full_construction(self):
        meta = LMCBlendCommonMetadata(
            check_layers=[1, 2],
            recomp_ratios=[0.1, 0.2],
            thresholds=[0.5, 0.6],
        )
        assert meta.recomp_ratios == [0.1, 0.2]
        assert meta.thresholds == [0.5, 0.6]


class TestLMCBlendMetadata:
    def test_defaults_to_none(self):
        meta = LMCBlendMetadata()
        assert meta.imp_indices is None
        assert meta.attn_mask is None
        assert meta.positions is None

    def test_clean_resets_all_fields(self):
        meta = LMCBlendMetadata(
            imp_indices=torch.tensor([1, 2, 3]),
            attn_mask=torch.ones(2, 2),
            positions=torch.arange(8),
        )
        meta.clean()
        assert meta.imp_indices is None
        assert meta.attn_mask is None
        assert meta.positions is None


def _fresh_id(prefix: str) -> str:
    """Generate a per-test instance_id so the process-wide builder registry
    can't leak entries between tests.
    """
    return f"{prefix}-{uuid.uuid4().hex}"


class TestLMCBlenderBuilder:
    def test_get_unknown_instance_raises(self):
        with pytest.raises(ValueError, match="not found"):
            LMCBlenderBuilder.get(_fresh_id("missing"))

    def test_get_or_create_constructs_once_and_caches(self):
        fake_blender = MagicMock(name="LMCBlender-instance")
        fake_vllm_model = MagicMock(name="vllm-model")
        instance_id = _fresh_id("once")

        with (
            patch(
                "lmcache.v1.compute.blend.utils.LMCBlender",
                return_value=fake_blender,
            ) as blender_cls,
            patch(
                "lmcache.v1.compute.blend.utils.VLLMModelTracker.get_model",
                return_value=fake_vllm_model,
            ) as get_model,
        ):
            engine = MagicMock(name="cache-engine")
            connector = MagicMock(name="gpu-connector")
            config = MagicMock(name="config")

            first = LMCBlenderBuilder.get_or_create(
                instance_id, engine, connector, config
            )
            second = LMCBlenderBuilder.get_or_create(
                instance_id, engine, connector, config
            )

            assert first is fake_blender
            assert second is fake_blender
            # Constructor and tracker lookup only fire on the first call.
            assert blender_cls.call_count == 1
            assert get_model.call_count == 1

    def test_get_after_create_returns_same_instance(self):
        fake_blender = MagicMock(name="LMCBlender-instance")
        instance_id = _fresh_id("get")
        with (
            patch(
                "lmcache.v1.compute.blend.utils.LMCBlender",
                return_value=fake_blender,
            ),
            patch(
                "lmcache.v1.compute.blend.utils.VLLMModelTracker.get_model",
                return_value=MagicMock(),
            ),
        ):
            LMCBlenderBuilder.get_or_create(
                instance_id, MagicMock(), MagicMock(), MagicMock()
            )
        assert LMCBlenderBuilder.get(instance_id) is fake_blender

    def test_distinct_instance_ids_get_distinct_blenders(self):
        # New LMCBlender per call so each instance_id sees a distinct object.
        instances = [MagicMock(name=f"blender-{i}") for i in range(2)]

        def make_blender(*args, **kwargs):
            return instances.pop(0)

        id_a, id_b = _fresh_id("a"), _fresh_id("b")
        with (
            patch(
                "lmcache.v1.compute.blend.utils.LMCBlender",
                side_effect=make_blender,
            ),
            patch(
                "lmcache.v1.compute.blend.utils.VLLMModelTracker.get_model",
                return_value=MagicMock(),
            ),
        ):
            a = LMCBlenderBuilder.get_or_create(
                id_a, MagicMock(), MagicMock(), MagicMock()
            )
            b = LMCBlenderBuilder.get_or_create(
                id_b, MagicMock(), MagicMock(), MagicMock()
            )
        assert a is not b
