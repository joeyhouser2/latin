"""Connector for Google Books (catalogue + the Internet Archive's copy).

Google's own full-text endpoints are not scriptable, but Google-digitized
public-domain volumes were mirrored to the Internet Archive as ``bub_gb_<id>``
items with OCR text, which ``treatises`` already reads (the repo's Salmasius and
several economic treatises came that way). So this connector's job is the part
Google *does* offer: search the catalogue, and map a Books id to its archive.org
copy.

The Books API works without a key only until a shared daily quota is spent
(it routinely is). Set ``GOOGLE_BOOKS_API_KEY`` (free, Google Cloud console)
for dependable use. ``fetch()`` itself needs no key -- it only talks to
archive.org -- so an id found by any means can be ingested.

Identifier:  a Books id (``dSihnyKx6hgC``) or a books.google.* URL.
discover():  a free-text query, Latin, full-view (public domain) volumes only.
             Returned ids are *candidates*: not every volume has an archive.org
             mirror; ``fetch()`` says so plainly when one does not.

Usage:
    python scripts/ingest.py googlebooks dSihnyKx6hgC
    python scripts/ingest.py googlebooks "de usuris" --discover --limit 20
"""
from __future__ import annotations

import os
import re
from typing import List

import requests

from .base import Connector, RawWork

_API = "https://www.googleapis.com/books/v1/volumes"
_ID = re.compile(r"(?:[?&]id=|^)([\w-]{12})(?:$|[&#])")


class GoogleBooksConnector(Connector):
    name = "googlebooks"

    def __init__(self, api_key: str = "", timeout: float = 60.0):
        self.key = api_key or os.environ.get("GOOGLE_BOOKS_API_KEY", "")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "LatinRAG-Research/1.0 (scholarly research)"

    def discover(self, query: str, limit: int = 40) -> List[str]:
        ids: List[str] = []
        start = 0
        while len(ids) < limit:
            params = {"q": query, "langRestrict": "la", "filter": "full",
                      "printType": "books", "maxResults": min(40, limit - len(ids)),
                      "startIndex": start}
            if self.key:
                params["key"] = self.key
            resp = self.session.get(_API, params=params, timeout=self.timeout)
            if resp.status_code == 429:
                raise RuntimeError(
                    "Google Books API quota exhausted. Set GOOGLE_BOOKS_API_KEY "
                    "(free) to get your own quota.")
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                break
            ids += [it["id"] for it in items]
            start += len(items)
        return list(dict.fromkeys(ids))[:limit]

    def fetch(self, identifier: str, **meta_overrides) -> RawWork:
        from .registry import get_connector
        m = _ID.search(identifier.strip().replace("googlebooks:", ""))
        if not m:
            raise ValueError(f"no Google Books id in {identifier!r}")
        gid = m.group(1)
        ia_id = f"bub_gb_{gid}"
        try:
            meta, parts = get_connector("treatises").fetch(f"ia:{ia_id}", **meta_overrides)
        except ValueError as e:
            raise ValueError(
                f"Google Books {gid}: archive.org copy {ia_id} unusable ({e}). "
                "Google's own text is not scriptable; download the PDF by hand "
                "and ingest it with `ocrimages`/`file`.") from e
        meta["source"] = f"Google Books {gid} via Internet Archive ({ia_id})"
        return meta, parts
