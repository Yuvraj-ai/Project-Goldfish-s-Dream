from __future__ import annotations

import logging

import pytest


TEST_PLUGIN_CODE = '''\
from open_deep_research.api.plugins.base import SourcePlugin, NormalizedResult, ContentResult

class TestPlugin(SourcePlugin):
    name = "test_plugin"

    async def search(self, query, max_results=10):
        return []

    async def fetch(self, url):
        return ContentResult(url=url, content="test")
'''

NO_PLUGIN_CODE = '''\
X = 42

def helper():
    return X
'''


def test_loader_empty_init():
    from open_deep_research.api.plugins.loader import PluginLoader

    loader = PluginLoader()
    assert loader._plugins == {}
    assert loader.list() == []


def test_load_from_nonexistent_directory(tmp_path, caplog):
    from open_deep_research.api.plugins.loader import PluginLoader

    caplog.set_level(logging.WARNING)
    loader = PluginLoader()
    bad_dir = str(tmp_path / "nonexistent")
    plugins = loader.load_from_directory(bad_dir)
    assert plugins == []
    assert "does not exist" in caplog.text


def test_load_from_directory_with_valid_plugin(tmp_path):
    from open_deep_research.api.plugins.loader import PluginLoader

    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(TEST_PLUGIN_CODE)

    loader = PluginLoader()
    plugins = loader.load_from_directory(str(tmp_path))
    assert len(plugins) == 1
    assert plugins[0].name == "test_plugin"


def test_load_from_directory_no_source_plugin(tmp_path):
    from open_deep_research.api.plugins.loader import PluginLoader

    plugin_file = tmp_path / "helper.py"
    plugin_file.write_text(NO_PLUGIN_CODE)

    loader = PluginLoader()
    plugins = loader.load_from_directory(str(tmp_path))
    assert plugins == []


def test_load_from_directory_skips_underscore_files(tmp_path):
    from open_deep_research.api.plugins.loader import PluginLoader

    plugin_file = tmp_path / "_private.py"
    plugin_file.write_text(TEST_PLUGIN_CODE)

    loader = PluginLoader()
    plugins = loader.load_from_directory(str(tmp_path))
    assert plugins == []


def test_get_returns_none_for_unknown():
    from open_deep_research.api.plugins.loader import PluginLoader

    loader = PluginLoader()
    assert loader.get("nonexistent") is None


def test_get_returns_plugin_for_registered(tmp_path):
    from open_deep_research.api.plugins.loader import PluginLoader

    plugin_file = tmp_path / "my_plugin.py"
    plugin_file.write_text(TEST_PLUGIN_CODE)

    loader = PluginLoader()
    loader.load_from_directory(str(tmp_path))
    plugin = loader.get("test_plugin")
    assert plugin is not None
    assert plugin.name == "test_plugin"


def test_list_returns_all_loaded_plugins(tmp_path):
    from open_deep_research.api.plugins.loader import PluginLoader

    TWO_PLUGIN_CODE = '''\
from open_deep_research.api.plugins.base import SourcePlugin, NormalizedResult, ContentResult

class PluginA(SourcePlugin):
    name = "plugin_a"

    async def search(self, query, max_results=10):
        return []

    async def fetch(self, url):
        return ContentResult(url=url, content="a")

class PluginB(SourcePlugin):
    name = "plugin_b"

    async def search(self, query, max_results=10):
        return []

    async def fetch(self, url):
        return ContentResult(url=url, content="b")
'''
    (tmp_path / "a.py").write_text(TWO_PLUGIN_CODE)

    loader = PluginLoader()
    loader.load_from_directory(str(tmp_path))
    names = [p.name for p in loader.list()]
    assert "plugin_a" in names
    assert "plugin_b" in names
    assert len(names) == 2


def test_load_from_directory_handles_import_error(tmp_path, caplog):
    from open_deep_research.api.plugins.loader import PluginLoader

    broken_file = tmp_path / "broken.py"
    broken_file.write_text("import nonexistent_module_xyz\n")

    caplog.set_level(logging.ERROR)
    loader = PluginLoader()
    plugins = loader.load_from_directory(str(tmp_path))
    assert plugins == []
    assert "Failed to load plugin" in caplog.text


def test_load_from_directory_skips_non_py_files(tmp_path):
    from open_deep_research.api.plugins.loader import PluginLoader

    (tmp_path / "data.txt").write_text("not python")
    (tmp_path / "notes.md").write_text("# readme")

    loader = PluginLoader()
    plugins = loader.load_from_directory(str(tmp_path))
    assert plugins == []
