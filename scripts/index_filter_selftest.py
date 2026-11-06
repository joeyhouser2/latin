"""Self-test: short segments stay in the store but get no search vector.

Uses a temporary database and index and a fake embedder, so it touches no real data.

Run:  python scripts/index_filter_selftest.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss
import numpy as np

from core.embedder import EMBEDDING_DIM
from core.normalize import is_worth_indexing, select_for_index
from pipeline import Library

LONG = ["Ave maris stella Dei mater alma.", "Qui tollis peccata mundi."]
SHORT = ["15.", "Cod.", "Vatican.", "Jerusalem, Jerusalem."]


class FakeEmbedder:
    def embed(self, texts, **_):
        rng = np.random.default_rng(0)
        return rng.random((len(texts), EMBEDDING_DIM), dtype="float32")


def indexed_ids(lib):
    return set(faiss.vector_to_array(lib.vectors.index.id_map).astype("int64").tolist())


def main():
    assert all(is_worth_indexing(t) for t in LONG)
    assert not any(is_worth_indexing(t) for t in SHORT)
    assert not is_worth_indexing("") and not is_worth_indexing("a b c")
    assert is_worth_indexing("καθ ὃν βιοὺς τίς")

    with tempfile.TemporaryDirectory() as tmp:
        lib = Library(db_path=os.path.join(tmp, "c.db"), index_path=os.path.join(tmp, "i.faiss"),
                      embedder=FakeEmbedder())
        doc = lib.add_document("\n".join(LONG + SHORT), "Test", verse=True)
        segs = list(doc.iter_segments())
        assert len(segs) == len(LONG + SHORT), "all segments must reach the store"
        want = {s.id for s in segs if s.latin_text in LONG}
        assert len(want) == len(LONG)
        assert indexed_ids(lib) == want, "ingest should index only long segments"

        assert lib.reindex() == len(LONG)
        assert indexed_ids(lib) == want, "reindex should index only long segments"
        assert len(lib.store.iter_all_segments()) == len(LONG + SHORT), "reindex must not drop text"

        # A document with only short segments still gets one vector: the
        # segment with the most letters. One with no letters gets none.
        tiny = lib.add_document("12.\nIn Christo\nCod.", "Tiny", verse=True)
        tsegs = {s.latin_text: s.id for s in tiny.iter_segments()}
        bare = lib.add_document("1.\n2.", "Numbers only", verse=True)
        expect = want | {tsegs["In Christo"]}
        assert indexed_ids(lib) == expect, "short-only doc should get its best segment"
        assert lib.reindex() == len(expect)
        assert indexed_ids(lib) == expect, "reindex should keep the fallback segment"
        assert not ({s.id for s in bare.iter_segments()} & indexed_ids(lib))
        lib.store.close()

    picked = select_for_index(["Cod.", "Vatican.", "x"], str, lambda t: 1)
    assert picked == ["Vatican."], picked
    assert select_for_index(["a b c d e", "Cod."], str, lambda t: 1) == ["a b c d e"]
    print("index filter selftest: OK")


if __name__ == "__main__":
    main()
