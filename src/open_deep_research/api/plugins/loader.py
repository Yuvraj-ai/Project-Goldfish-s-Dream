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
        logger.info("Loading plugins from directory %s", directory)
        plugins: list[SourcePlugin] = []
        if not os.path.isdir(directory):
            logger.warning("Plugin directory %s does not exist", directory)
            return plugins

        for filename in os.listdir(directory):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            filepath = os.path.join(directory, filename)
            logger.debug("Discovered plugin candidate %s", filepath)
            try:
                spec = importlib.util.spec_from_file_location(
                    f"plugin_{filename[:-3]}", filepath,
                )
                if spec is None or spec.loader is None:
                    logger.warning("Skipping plugin %s: no import spec/loader", filename)
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if isinstance(attr, type) and issubclass(attr, SourcePlugin) and attr is not SourcePlugin:
                        instance = attr()
                        self._plugins[instance.name] = instance
                        plugins.append(instance)
                        logger.info("Loaded plugin %s from %s", instance.name, filename)
            except Exception:
                logger.exception("Failed to load plugin %s", filename)

        logger.info("Loaded %d plugin(s) from directory %s", len(plugins), directory)
        return plugins

    def get(self, name: str) -> SourcePlugin | None:
        plugin = self._plugins.get(name)
        if plugin is None:
            logger.warning("Plugin %s not found in registry", name)
        return plugin

    def register(self, name: str, plugin: SourcePlugin) -> None:
        logger.info("Registering plugin %s", name)
        self._plugins[name] = plugin

    def list(self) -> list[SourcePlugin]:
        logger.debug("Listing registered plugins: %d total", len(self._plugins))
        return list(self._plugins.values())
