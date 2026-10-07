"""Connector for Europeana, the EU aggregator of library and archive holdings.

Europeana holds *metadata*, not text: each record points at a scan on the
contributing institution's own site. So this is a discovery connector with a
routed fetch: ``discover()`` finds open-licence Latin text items, and
``fetch()`` follows each record's links through ``ingest.copies`` to a
connector that can read the scan (an MDZ page, an archive.org item, or a IIIF
manifest). When a record only points at a viewer page we cannot read, it
raises ``NoReadableCopy`` listing the links -- it does not scrape viewers.

Needs an API key for real use (free, https://pro.europeana.eu/pages/get-api);
set ``EUROPEANA_API_KEY``. Without one it falls back to the shared
``api2demo`` key, which is heavily rate-limited and meant for trials only.

Identifier:  ``/<provider>/<record>`` as Europeana prints it, e.g.
             ``/1613/item_CM5FSZPYSNIPQ5ZKXCYI537CQXKBTOCX``
discover():  a free-text query; results are Latin (``proxy_dc_language:la``),
             type TEXT, open reusability.

Usage:
    python scripts/ingest.py europeana "usuris" --discover --limit 20
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, Iterator, List

import requests

from .base import Connector, RawWork
from .copies import fetch_first_readable
from .iiif_meta import century_of
from .treatises import _stage_for

_SEARCH = "https://api.europeana.eu/record/v2/search.json"
_RECORD = "https://api.europeana.eu/record/v2{id}.json"
_DEMO_KEY = "api2demo"


def _first(value: Any) -> Any:
    if isinstance(value, dict):                  # language map {"def": [...]}
        value = next(iter(value.values()), None)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _as_list(value: Any) -> List[str]:
    if isinstance(value, dict):
        value = [v for vs in value.values() for v in (vs if isinstance(vs, list) else [vs])]
    if isinstance(value, str):
        return [value]
    return [str(v) for v in (value or []) if v]


class EuropeanaConnector(Connector):
    name = "europeana"

    def __init__(self, api_key: str = "", timeout: float = 60.0):
        self.key = api_key or os.environ.get("EUROPEANA_API_KEY") or _DEMO_KEY
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "LatinRAG-Research/1.0 (scholarly research)"
        if self.key == _DEMO_KEY:
            print("  europeana: using the shared demo key (set EUROPEANA_API_KEY)",
                  file=sys.stderr)

    def _get(self, url: str, **params) -> Dict[str, Any]:
        resp = self.session.get(url, params={"wskey": self.key, **params},
                                timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") is False:
            raise ValueError(f"Europeana error: {data.get('error') or data}")
        return data

    # ---- search -----------------------------------------------------------
    def catalog(self, query: str, limit: int = 50) -> Iterator[Dict[str, Any]]:
        got, cursor = 0, "*"
        while got < limit:
            data = self._get(_SEARCH, query=query, rows=min(100, limit - got),
                             cursor=cursor, reusability="open",
                             qf=["proxy_dc_language:la", "TYPE:TEXT"])
            items = data.get("items", [])
            for it in items:
                got += 1
                yield {
                    "id": it["id"], "title": _first(it.get("title")),
                    "author": ", ".join(_as_list(it.get("dcCreator"))) or None,
                    "year": _first(it.get("year")),
                    "provider": _first(it.get("dataProvider")),
                    "rights": _first(it.get("rights")),
                    "shown_by": _as_list(it.get("edmIsShownBy")),
                    "shown_at": _as_list(it.get("edmIsShownAt")),
                }
                if got >= limit:
                    return
            cursor = data.get("nextCursor")
            if not items or not cursor:
                return

    def discover(self, query: str, limit: int = 50) -> List[str]:
        return [r["id"] for r in self.catalog(query, limit)]

    # ---- fetch ------------------------------------------------------------
    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        base, _, opts = identifier.partition("#")
        rid = "/" + base.strip().replace("europeana:", "").strip("/")
        if not re.match(r"^/\d+/\w+$", rid):
            raise ValueError(f"expected a Europeana id like /1613/item_ABC, got {identifier!r}")
        obj = self._get(_RECORD.format(id=rid))["object"]
        proxy = next((p for p in obj.get("proxies", []) if not p.get("europeanaProxy")),
                     (obj.get("proxies") or [{}])[0])
        agg = (obj.get("aggregations") or [{}])[0]

        urls: List[str] = []
        for w in agg.get("webResources", []):
            urls += _as_list(w.get("dctermsIsReferencedBy"))        # IIIF manifests
        urls += _as_list(agg.get("edmIsShownAt")) + _as_list(agg.get("edmIsShownBy"))
        urls += [w.get("about", "") for w in agg.get("webResources", [])]
        urls = list(dict.fromkeys(u for u in urls if u))

        created = " ".join(_as_list(proxy.get("dctermsCreated")) + _as_list(proxy.get("year")))
        century = century_of(created)
        cat_meta = {
            "title": _first(proxy.get("dcTitle")) or rid,
            "author": ", ".join(_as_list(proxy.get("dcCreator"))) or None,
            "century": century,
            "language_stage": _stage_for(((century or 0) * 100 - 50) or None),
            "license": _first(agg.get("edmRights")),
        }
        cat_meta = {k: v for k, v in cat_meta.items() if v}
        cat_meta.update(meta_overrides)
        meta, parts = fetch_first_readable(urls, options=opts, **cat_meta)
        meta["source"] = (f"Europeana {rid} "
                          f"({_first(agg.get('edmDataProvider')) or 'provider unknown'}); "
                          f"{meta.get('source', '')}").strip("; ")
        return meta, parts
