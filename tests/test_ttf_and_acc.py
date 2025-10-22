from seiskit import acc, optools, ttf


def test_ttf_empty():
    freqs, amps = ttf.ttf_from_acc([], 0.01)
    assert freqs == [] and amps == []


def test_ttf_simple_sine():
    # create a simple sine wave at 1 Hz sampled at 20 Hz
    import math

    fs = 20.0
    dt = 1.0 / fs
    n = 40
    acc_ts = [math.sin(2 * math.pi * 1.0 * (i * dt)) for i in range(n)]
    freqs, amps = ttf.ttf_from_acc(acc_ts, dt)
    # expect non-empty output and a peak near 1 Hz
    assert len(freqs) > 0
    # find index of maximum amplitude
    imax = max(range(len(amps)), key=lambda i: amps[i])
    assert abs(freqs[imax] - 1.0) < 0.5


def test_acc2fas_calls_ttf():
    data = [0.0, 1.0, 0.0, -1.0]
    freqs1, amps1 = acc.acc2fas(data, 0.1)
    freqs2, amps2 = ttf.ttf_from_acc(data, 0.1)
    assert freqs1 == freqs2 and amps1 == amps2


def test_read_time_series(tmp_path):
    p = tmp_path / "vals.txt"
    p.write_text("# comment\n0.1\n0.2\n\n0.3\n")
    vals = optools.read_time_series(str(p))
    assert vals == [0.1, 0.2, 0.3]
