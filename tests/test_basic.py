from src.ingest import clean_text, fixed_spans, sentence_spans, strip_boilerplate
from src.utils import first_hit_rank, is_hit, rrf_fuse, tokenize


def test_tokenize_keeps_devanagari_words_whole():
    assert tokenize("किसान सम्मान निधि Yojana") == ["किसान", "सम्मान", "निधि", "yojana"]


def test_clean_text_dehyphenates_and_joins_lines():
    assert clean_text("exam-\nple text\nnext line\n\nNew para") == "example text next line\n\nNew para"


def test_fixed_spans_cover_text_with_overlap():
    text = "word " * 1000
    spans = fixed_spans(text, size=400, overlap=80)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(e - s <= 400 for s, e in spans)
    assert all(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))  # overlap


def test_sentence_spans_split_on_danda():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है। Third one."
    assert len(sentence_spans(text)) == 3


def test_strip_boilerplate_removes_repeated_header_and_page_numbers():
    pages = [f"Ministry of X\nBody text {i}\n{i}" for i in range(1, 8)]
    out = strip_boilerplate(pages)
    assert all("Ministry of X" not in p for p in out)
    assert out[2].strip() == "Body text 3"


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf_fuse([[1, 2], [1, 3]])
    assert fused[0][0] == 1


def test_is_hit_and_rank():
    chunk = {"doc": "a.pdf", "page_start": 3, "page_end": 4}
    assert is_hit(chunk, "a.pdf", [4])
    assert not is_hit(chunk, "a.pdf", [5])
    assert not is_hit(chunk, "b.pdf", [3])
    other = {"doc": "a.pdf", "page_start": 9, "page_end": 9}
    assert first_hit_rank([other, chunk], "a.pdf", [3]) == 2
    assert first_hit_rank([other], "a.pdf", [3]) is None
