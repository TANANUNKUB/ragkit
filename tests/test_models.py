from ragkit.models import Answer, Chunk, Document, EvalExample, Query, ScoredHit


def test_defaults():
    doc = Document(id="d", text="hello")
    assert doc.metadata == {}
    chunk = Chunk(id="d#0", doc_id="d", text="hello")
    assert chunk.metadata == {}
    q = Query(text="q")
    assert q.filters == {}
    hit = ScoredHit(chunk=chunk, score=0.5, source="bm25")
    ans = Answer(answer="ok", citations=["d"], found=True, hits=[hit])
    assert ans.found is True
    ev = EvalExample(id="e", query="q")
    assert ev.gold_doc_ids == []
    assert ev.gold_answer is None
