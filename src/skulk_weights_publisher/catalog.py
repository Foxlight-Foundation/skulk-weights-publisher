"""Preferred American-English aliases for catalog helpers."""

from __future__ import annotations

from skulk_weights_publisher.catalogue import (
    CatalogueSource as CatalogSource,
)
from skulk_weights_publisher.catalogue import (
    CatalogueView as CatalogView,
)
from skulk_weights_publisher.catalogue import (
    filter_catalogue_entries as filter_catalog_entries,
)
from skulk_weights_publisher.catalogue import (
    find_catalogue_entry as find_catalog_entry,
)
from skulk_weights_publisher.catalogue import (
    load_catalogue_view as load_catalog_view,
)
from skulk_weights_publisher.catalogue import (
    write_default_config,
)

__all__ = [
    "CatalogSource",
    "CatalogView",
    "filter_catalog_entries",
    "find_catalog_entry",
    "load_catalog_view",
    "write_default_config",
]
