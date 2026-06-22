from __future__ import annotations

import importlib.util
import logging
import os

from open_deep_research.api.plugins.base import SourcePlugin

logger = logging.getLogger(__name__)


class PluginLoader:
    def __init__(self) -> None:
        self._plugins: dict[str, SourcePlugin] = {}

    def load_from_directory(self, directory: str) -> list[SourcePlugin]:
        plugins: list[SourcePlugin] = []
        if not os.path.isdir(directory):
            logger.warning("Plugin directory %s does not exist", directory)
            return plugins

        for filename in os.listdir(directory):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            filepath = os.path.join(directory, filename)
            try:
                spec = importlib.util.spec_from_file_location(
                    f"plugin_{filename[:-3]}", filepath,
                )
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type) and issubclass(attr, SourcePlugin) and attr is not SourcePlugin:
                        instance = attr()
                        self._plugins[instance.name] = instance
                        plugins.append(instance)
            except Exception:
                logger.exception("Failed to load plugin %s", filename)

        return plugins

    def get(self, name: str) -> SourcePlugin | None:
        return self._plugins.get(name)

    def register(self, name: str, plugin: SourcePlugin) -> None:
        self._plugins[name] = plugin

    def list(self) -> list[SourcePlugin]:
        return list(self._plugins.values())
