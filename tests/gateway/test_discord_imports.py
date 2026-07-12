"""Import-safety tests for the Discord gateway adapter."""

import builtins
import importlib
import sys


class TestDiscordImportSafety:
    def test_module_imports_even_when_discord_dependency_is_missing(self):
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "discord" or name.startswith("discord."):
                raise ImportError("discord unavailable for test")
            return original_import(name, globals, locals, fromlist, level)

        # This test deliberately re-imports the adapter with discord.py
        # simulated-missing, which rebinds ``adapter.discord = None``. That
        # poisoned module must NOT survive into other tests: a later test that
        # does ``monkeypatch.setattr("plugins.platforms.discord.adapter."
        # "discord.DMChannel", ...)`` would blow up with ``'NoneType' object
        # has no attribute 'DMChannel'``. ``monkeypatch.delitem(...,
        # raising=False)`` does not protect us — when the adapter module is
        # *absent* here (order-dependent under pytest-randomly) it records
        # nothing to restore, so the reimport leaks the ``discord=None`` module.
        # Snapshot the exact sys.modules entries and restore them ourselves
        # (pop when originally absent, reinstate when present). We also manage
        # ``builtins.__import__`` by hand rather than via ``monkeypatch`` so the
        # restore order is guaranteed and we don't disturb the shared
        # (autouse-fixture) monkeypatch instance. See HUM-2223 / HUM-2208.
        poisonable = (
            "plugins.platforms.discord.adapter",
            "plugins.platforms.discord",
        )
        saved = {name: sys.modules.get(name) for name in poisonable}

        builtins.__import__ = fake_import
        try:
            for name in poisonable:
                sys.modules.pop(name, None)

            module = importlib.import_module("plugins.platforms.discord.adapter")

            assert module.DISCORD_AVAILABLE is False
            assert module.discord is None
        finally:
            builtins.__import__ = original_import
            for name, original in saved.items():
                if original is not None:
                    sys.modules[name] = original
                else:
                    # Was absent before this test — drop the poisoned
                    # (discord=None) module so the next importer rebuilds it
                    # cleanly against the real/faked discord library.
                    sys.modules.pop(name, None)

            # ``importlib.import_module`` rebinds the child attribute on each
            # parent package along the chain to the freshly imported (poisoned,
            # discord=None) module objects — i.e. ``plugins.platforms.discord``
            # becomes a new package whose ``.adapter`` is the discord=None
            # module. pytest's ``monkeypatch.setattr`` string-target resolver
            # walks *package attributes* via ``getattr`` (see
            # ``_pytest.monkeypatch.resolve``), NOT ``sys.modules`` — so
            # restoring ``sys.modules`` alone leaves the poisoned objects
            # reachable through ``plugins.platforms.discord[.adapter]`` and a
            # later ``monkeypatch.setattr("plugins.platforms.discord.adapter."
            # "discord.X")`` resolves ``discord`` to ``None``. Repoint each
            # parent's attribute to the restored module, parent-first.
            for name in ("plugins.platforms.discord",
                         "plugins.platforms.discord.adapter"):
                parent_name, _, attr = name.rpartition(".")
                parent_mod = sys.modules.get(parent_name)
                restored = sys.modules.get(name)
                if parent_mod is None:
                    continue
                if restored is not None:
                    setattr(parent_mod, attr, restored)
                elif hasattr(parent_mod, attr):
                    delattr(parent_mod, attr)
