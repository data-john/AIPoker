import json

import report
import spike_b_extrapolate as ex
from common import gate


def test_power_law_fit_recovers_exponent_and_defaults_to_linear():
    a, b, note = ex.fit_power_law([(10_000_000, 3.0), (30_000_000, 7.0)])
    assert 0.74 < b < 0.80 and "fitted" in note
    _, b1, note1 = ex.fit_power_law([(10_000_000, 3.0)])
    assert b1 == 1.0 and "linear" in note1


def test_report_marks_mock_and_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    (tmp_path / "spike_a_engine.json").write_text(
        json.dumps(
            {
                "spike": "spike_a_engine",
                "system": {"logical_cpus": 4},
                "gates": [gate("G0.1", True, "ok")],
            }
        )
    )
    (tmp_path / "spike_f_local_decision.json").write_text(
        json.dumps(
            {
                "spike": "spike_f_local_decision",
                "is_mock": True,
                "gates": [gate("F", None, "mock run")],
            }
        )
    )
    (tmp_path / "spike_d_gpu.json").write_text(
        json.dumps({"spike": "spike_d_gpu", "gates": [gate("G0.3", None, "torch missing")]})
    )
    text = report.build()
    assert "| G0.1 | PASS |" in text
    assert "MOCK (not a model result)" in text
    assert "| G0.3 | PENDING |" in text
    assert "Not yet run: G0.2" in text
