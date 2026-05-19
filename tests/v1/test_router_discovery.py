# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for lmcache.v1.utils.router_discovery.discover_api_routers.

Builds throwaway packages on disk, imports them via importlib, and asserts
that only modules matching the suffix and not in the exclude set contribute
their ``router`` attribute to the discovered list.
"""

# Standard
from pathlib import Path
import sys
import textwrap

# Third Party
from fastapi import APIRouter
import pytest

# First Party
from lmcache.v1.utils.router_discovery import discover_api_routers


def _write_module(pkg_dir: Path, name: str, body: str) -> None:
    (pkg_dir / f"{name}.py").write_text(textwrap.dedent(body))


@pytest.fixture
def fake_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """Create a unique fake package on disk and add its parent to sys.path."""
    pkg_name = f"_lmc_router_disc_test_{tmp_path.name}"
    pkg_dir = tmp_path / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")

    monkeypatch.syspath_prepend(str(tmp_path))
    yield pkg_dir, pkg_name

    # Drop any cached imports so the next test starts fresh.
    for mod in list(sys.modules):
        if mod == pkg_name or mod.startswith(pkg_name + "."):
            del sys.modules[mod]


class TestDiscoverApiRouters:
    def test_picks_up_matching_suffix(self, fake_pkg):
        pkg_dir, pkg_name = fake_pkg
        _write_module(
            pkg_dir,
            "alpha_api",
            """
            from fastapi import APIRouter
            router = APIRouter()
            """,
        )
        _write_module(
            pkg_dir,
            "beta_api",
            """
            from fastapi import APIRouter
            router = APIRouter()
            """,
        )
        # This one does not end in _api → should be ignored.
        _write_module(
            pkg_dir,
            "gamma_helper",
            """
            from fastapi import APIRouter
            router = APIRouter()
            """,
        )

        routers = discover_api_routers(pkg_dir, pkg_name)
        assert len(routers) == 2
        for r in routers:
            assert isinstance(r, APIRouter)

    def test_custom_suffix(self, fake_pkg):
        pkg_dir, pkg_name = fake_pkg
        _write_module(
            pkg_dir,
            "alpha_routes",
            """
            from fastapi import APIRouter
            router = APIRouter()
            """,
        )
        _write_module(
            pkg_dir,
            "beta_api",
            """
            from fastapi import APIRouter
            router = APIRouter()
            """,
        )

        routers = discover_api_routers(pkg_dir, pkg_name, suffix="_routes")
        assert len(routers) == 1

    def test_exclude_skips_named_modules(self, fake_pkg):
        pkg_dir, pkg_name = fake_pkg
        _write_module(
            pkg_dir,
            "alpha_api",
            """
            from fastapi import APIRouter
            router = APIRouter()
            """,
        )
        _write_module(
            pkg_dir,
            "skip_me_api",
            """
            from fastapi import APIRouter
            router = APIRouter()
            """,
        )

        routers = discover_api_routers(pkg_dir, pkg_name, exclude={"skip_me_api"})
        assert len(routers) == 1

    def test_module_without_router_attribute_is_ignored(self, fake_pkg):
        pkg_dir, pkg_name = fake_pkg
        _write_module(
            pkg_dir,
            "noattr_api",
            """
            # no router defined
            value = 42
            """,
        )
        _write_module(
            pkg_dir,
            "good_api",
            """
            from fastapi import APIRouter
            router = APIRouter()
            """,
        )

        routers = discover_api_routers(pkg_dir, pkg_name)
        assert len(routers) == 1

    def test_module_with_non_apirouter_attribute_is_ignored(self, fake_pkg):
        pkg_dir, pkg_name = fake_pkg
        _write_module(
            pkg_dir,
            "bogus_api",
            """
            router = "not an APIRouter instance"
            """,
        )

        routers = discover_api_routers(pkg_dir, pkg_name)
        assert routers == []

    def test_empty_dir_returns_empty_list(self, fake_pkg):
        pkg_dir, pkg_name = fake_pkg
        assert discover_api_routers(pkg_dir, pkg_name) == []
