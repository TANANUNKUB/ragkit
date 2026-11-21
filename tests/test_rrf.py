from ragkit.models import Chunk, ScoredHit
from ragkit.retrievers.aggregate import aggregate_max_per_doc
from ragkit.retrievers.rrf import rrf_fuse, rrf_fuse_hits


def test_rrf_known_scores():
    # item a ranks 1 in list1 and 2 in list2; b ranks 2 and 1; c only in list1 at 3.
    k = 60
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "a"]], k=k))
    score_a = 1 / (k + 1) + 1 / (k + 2)
    score_b = 1 / (k + 2) + 1 / (k + 1)
    score_c = 1 / (k + 3)
    assert abs(fused["a"] - score_a) < 1e-12
    assert abs(fused["b"] - score_b) < 1e-12
    assert abs(fused["c"] - score_c) < 1e-12
    # a and b tied on this example
    assert abs(fused["a"] - fused["b"]) < 1e-12
    assert fused["a"] > fused["c"]


def test_rrf_missing_from_one_list_and_ordering():
    fused = rrf_fuse([["x"], ["y", "x"]], k=60)
    ids = [i for i, _ in fused]
    assert set(ids) == {"x", "y"}
    # x is rank 1 + rank 2, y is only rank 1 → x wins
    assert ids[0] == "x"


def test_rrf_rejects_non_positive_k():
    try:
        rrf_fuse([["a"]], k=0)
        assert False
    except ValueError:
        pass


def test_rrf_fuse_hits_preserves_chunk_and_source():
    def hit(cid: str, doc: str, score: float) -> ScoredHit:
        return ScoredHit(chunk=Chunk(id=cid, doc_id=doc, text=cid), score=score, source="bm25")

    out = rrf_fuse_hits(
        [[hit("c1", "d1", 1.0), hit("c2", "d2", 0.5)], [hit("c2", "d2", 0.9)]],
        k=60,
    )
    assert all(h.source == "rrf" for h in out)
    assert {h.chunk.id for h in out} == {"c1", "c2"}


def test_aggregate_max_per_doc():
    def hit(cid: str, doc: str, score: float) -> ScoredHit:
        return ScoredHit(chunk=Chunk(id=cid, doc_id=doc, text=cid), score=score, source="bm25")

    hits = [hit("a#0", "a", 0.4), hit("a#1", "a", 0.9), hit("b#0", "b", 0.7)]
    out = aggregate_max_per_doc(hits)
    assert [h.chunk.doc_id for h in out] == ["a", "b"]
    assert out[0].chunk.id == "a#1"
    assert out[0].score == 0.9
