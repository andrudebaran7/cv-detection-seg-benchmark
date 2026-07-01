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


def test_percentile_linear_interpolation():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert measure._percentile(data, 50) == 55.0
    assert measure._percentile(data, 90) == 91.0
    assert round(measure._percentile(data, 99), 4) == 99.1


def test_latency_stats_reports_mean_std_percentiles():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    s = measure.latency_stats(data)
    assert s["n"] == 10
    assert s["mean_ms"] == 55.0
    assert s["p50_ms"] == 55.0
    assert s["p90_ms"] == 91.0
    assert s["std_ms"] >= 0.0


def test_latency_stats_single_sample():
    s = measure.latency_stats([42.0])
    assert s["n"] == 1
    assert s["mean_ms"] == 42.0
    assert s["p50_ms"] == 42.0 and s["p90_ms"] == 42.0 and s["p99_ms"] == 42.0
    assert s["std_ms"] == 0.0


def test_time_per_image_one_sample_per_image_after_warmup():
    calls = {"n": 0}

    def predict_one(img):
        calls["n"] += 1

    images = ["a", "b", "c"]
    samples = measure.time_per_image(predict_one, images, warmup=2)
    assert len(samples) == 3
    assert calls["n"] == 2 + 3  # warmup on images[0] + one per image
    assert all(ms >= 0.0 for ms in samples)
