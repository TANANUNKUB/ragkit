from ragkit.eval.metrics import mrr, ndcg_at_k, percentile, recall_at_k


def test_recall_perfect_and_partial_and_zero():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["a", "b"], k=2) == 1.0
    assert recall_at_k(ranked, ["a", "z"], k=2) == 0.5
    assert recall_at_k(ranked, ["z"], k=2) == 0.0


def test_recall_empty_gold_is_zero():
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_recall_rejects_non_positive_k():
    try:
        recall_at_k(["a"], ["a"], k=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_mrr_first_and_later_and_miss():
    assert mrr(["g", "x"], ["g"]) == 1.0
    assert mrr(["x", "g"], ["g"]) == 0.5
    assert mrr(["x", "y"], ["g"]) == 0.0
    assert mrr(["x"], []) == 0.0


def test_ndcg_known_values():
    import math

    # Gold {a,b}; ranking a,c,b at k=3 → rel = [1, 0, 1]
    value = ndcg_at_k(["a", "c", "b", "d"], ["a", "b"], k=3)
    dcg = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert abs(value - dcg / idcg) < 1e-9


def test_ndcg_perfect_is_one():
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], k=2) == 1.0


def test_ndcg_empty_gold():
    assert ndcg_at_k(["a"], [], k=5) == 0.0


def test_percentile_p50_p95():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 5.0
    assert percentile(xs, 50) == 3.0
    assert percentile([], 50) == 0.0
    # 95th of 0..99 is interpolated near 94.05
    values = [float(i) for i in range(100)]
    p95 = percentile(values, 95)
    assert 94.0 <= p95 <= 95.0
