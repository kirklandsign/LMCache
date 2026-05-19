# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ``lmcache`` CLI entry point in lmcache.cli.main.

Covers argument dispatch (subcommand wiring), the no-command help branch,
KeyboardInterrupt -> exit 130, and uncaught-exception -> exit 1.
"""

# Standard
from unittest.mock import MagicMock, patch
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.main import main


class _FakeCommand(BaseCommand):
    """Test double that records which arg.func payload was invoked."""

    def __init__(self, name: str = "fake", on_execute=None):
        self._name = name
        self.called_with: argparse.Namespace | None = None
        self._on_execute = on_execute

    def name(self) -> str:
        return self._name

    def help(self) -> str:
        return "fake command for testing"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--flag", action="store_true")

    def execute(self, args: argparse.Namespace) -> None:
        self.called_with = args
        if self._on_execute is not None:
            self._on_execute()


def _run_with(commands: list[BaseCommand], argv: list[str]) -> None:
    """Invoke ``main()`` with the given ALL_COMMANDS and argv."""
    with (
        patch("lmcache.cli.main.ALL_COMMANDS", commands),
        patch("sys.argv", ["lmcache", *argv]),
    ):
        main()


class TestDispatch:
    def test_subcommand_dispatches_to_execute(self):
        cmd = _FakeCommand("fake")
        _run_with([cmd], ["fake", "--flag"])
        assert cmd.called_with is not None
        assert cmd.called_with.flag is True

    def test_multiple_commands_only_target_one_runs(self):
        a = _FakeCommand("a")
        b = _FakeCommand("b")
        _run_with([a, b], ["b"])
        assert a.called_with is None
        assert b.called_with is not None


class TestNoCommand:
    def test_no_subcommand_prints_help_and_exits_one(self, capsys):
        cmd = _FakeCommand("fake")
        with pytest.raises(SystemExit) as exc_info:
            _run_with([cmd], [])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # argparse prints the help text on stdout.
        assert "usage:" in captured.out.lower()
        # The execute path of the registered command must not have run.
        assert cmd.called_with is None


class TestErrorHandling:
    def test_keyboard_interrupt_exits_130(self):
        cmd = _FakeCommand(
            "boom", on_execute=lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        )
        with pytest.raises(SystemExit) as exc_info:
            _run_with([cmd], ["boom"])
        assert exc_info.value.code == 130

    def test_uncaught_exception_exits_one(self):
        cmd = _FakeCommand(
            "boom", on_execute=lambda: (_ for _ in ()).throw(RuntimeError("nope"))
        )
        with pytest.raises(SystemExit) as exc_info:
            _run_with([cmd], ["boom"])
        assert exc_info.value.code == 1

    def test_systemexit_from_command_passes_through_with_its_code(self):
        # Commands may call sys.exit() themselves; the wrapper's
        # ``except Exception`` branch must not swallow SystemExit, so the
        # original exit code reaches the OS.
        def raise_exit() -> None:
            raise SystemExit(42)

        cmd = _FakeCommand("boom", on_execute=raise_exit)
        with pytest.raises(SystemExit) as exc_info:
            _run_with([cmd], ["boom"])
        assert exc_info.value.code == 42


class TestRegistration:
    def test_register_calls_each_command(self):
        cmd_a = MagicMock(spec=BaseCommand)
        cmd_a.name.return_value = "a"
        cmd_a.help.return_value = "alpha"
        cmd_b = MagicMock(spec=BaseCommand)
        cmd_b.name.return_value = "b"
        cmd_b.help.return_value = "beta"

        # When neither subcommand is provided, main() exits with code 1
        # without invoking either command's execute() — but both must have
        # been registered on the parser.
        with (
            patch("lmcache.cli.main.ALL_COMMANDS", [cmd_a, cmd_b]),
            patch("sys.argv", ["lmcache"]),
        ):
            with pytest.raises(SystemExit):
                main()

        cmd_a.register.assert_called_once()
        cmd_b.register.assert_called_once()
