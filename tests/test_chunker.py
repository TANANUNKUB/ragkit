from ragkit.chunkers.fixed import FixedSizeChunker
from ragkit.chunkers.splitting import iter_sections, split_sentences
from ragkit.chunkers.structure import StructureChunker
from ragkit.chunkers.tokens import ApproxTokenCounter
from ragkit.parsers.base import ParsedDocument


def test_split_sentences_uses_punctuation_not_spaces():
    # English
    sents = split_sentences("Hello world. How are you? Fine.")
    assert sents[0].startswith("Hello")
    assert any("How are you" in s for s in sents)
    # Thai has no spaces between words — must stay one sentence without punctuation.
    thai = "เอกสารนี้กำหนดสิทธิ์การเข้าถึงคลังเอกสารขององค์กร"
    assert split_sentences(thai) == [thai]


def test_iter_sections_markdown_headings():
    text = "# Title\n\nIntro para.\n\n## Method\n\nDetails here.\n\n## Results\n\nNumbers."
    sections = iter_sections(text)
    paths = [p for p, _ in sections]
    assert any(p.endswith("Method") or "Method" in p for p in paths)
    assert any("Results" in p for p in paths)


def test_structure_chunker_respects_token_budget_and_overlap():
    counter = ApproxTokenCounter(chars_per_token=4.0)
    # Build a long section that must be split.
    sentences = [f"This is sentence number {i} with some extra words." for i in range(40)]
    text = "# Section\n\n" + " ".join(sentences)
    chunker = StructureChunker(max_tokens=40, overlap=8, token_counter=counter)
    chunks = chunker.chunk(ParsedDocument(id="d1", text=text, title="Paper"))
    assert len(chunks) >= 2
    for ch in chunks:
        # Prefix (title > section) counts too, allow a small slack for packing.
        assert counter.count(ch.text) <= 40 + 5
        assert ch.doc_id == "d1"
        assert ch.id.startswith("d1#")
        assert "section" in ch.metadata
        assert "chunk_idx" in ch.metadata
    # Overlap: the start of chunk i+1 should share a suffix of chunk i (body-ish).
    if len(chunks) >= 2:
        # At least some character overlap between consecutive chunk bodies.
        a, b = chunks[0].text, chunks[1].text
        # Drop heading prefixes for a looser check.
        def body(t: str) -> str:
            return t.split("\n\n", 1)[-1]

        assert any(body(a)[i : i + 20] in body(b) for i in range(0, max(1, len(body(a)) - 20), 8)) or body(a)[-16:] in body(b)


def test_structure_chunker_does_not_split_short_sentence():
    text = "# H\n\nA short standalone sentence that fits."
    chunks = StructureChunker(max_tokens=200, overlap=10).chunk(
        ParsedDocument(id="d", text=text, title="T")
    )
    assert len(chunks) == 1
    assert "short standalone sentence" in chunks[0].text


def test_structure_chunker_keeps_thai_block_together_when_under_budget():
    thai = "พระราชบัญญัติธุรกิจสถาบันการเงินกำหนดหลักเกณฑ์การกำกับดูแลอย่างรัดกุม"
    chunks = StructureChunker(max_tokens=200, overlap=10).chunk(
        ParsedDocument(id="law", text=thai, title="มาตรา 1", metadata={"tenant_id": "public"})
    )
    assert len(chunks) == 1
    assert thai in chunks[0].text
    # Did not invent spaces or split into characters.
    assert chunks[0].text.count(thai) == 1


def test_fixed_size_fallback_windows():
    counter = ApproxTokenCounter(chars_per_token=4.0)
    text = "abcd" * 80  # 320 chars ≈ 80 tokens
    chunker = FixedSizeChunker(max_tokens=20, overlap=5, token_counter=counter)
    chunks = chunker.chunk(ParsedDocument(id="x", text=text))
    assert len(chunks) >= 3
    for ch in chunks:
        assert counter.count(ch.text) <= 20 + 1


def test_table_is_not_shredded_without_header_repeat_when_small():
    table = "# Tables\n\n| Year | Rev |\n| --- | --- |\n| 2020 | 10 |\n| 2021 | 12 |"
    chunks = StructureChunker(max_tokens=400, overlap=10).chunk(
        ParsedDocument(id="fin", text=table, title="10-K")
    )
    assert len(chunks) == 1
    assert "| Year | Rev |" in chunks[0].text
