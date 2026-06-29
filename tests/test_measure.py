from benchmark import measure


def test_timeit_callable_runs_iters_and_reports_stats():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    stats = measure.timeit_callable(fn, warmup=2, iters=10)
    assert calls["n"] == 12  # warmup + iters
    assert stats["n_iters"] == 10
    assert stats["mean_ms"] >= 0.0
    assert stats["std_ms"] >= 0.0
    assert "median_ms" in stats


def test_peak_rss_mb_is_positive():
    assert measure.peak_rss_mb() > 0.0


def test_time_first_call_runs_once():
    calls = {"n": 0}
    measure.time_first_call_ms(lambda: calls.__setitem__("n", calls["n"] + 1))
    assert calls["n"] == 1
