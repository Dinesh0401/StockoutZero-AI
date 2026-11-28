import json

from research.calibration import build_calibration_report, save_calibration_report


def test_calibration_report_is_reproducible_and_has_synthetic_warning():
    first = build_calibration_report((30,), (7, 19))
    second = build_calibration_report((30,), (7, 19))
    assert first == second
    assert first[0].seed_count == 2
    assert "Synthetic" in first[0].warning


def test_calibration_report_saves_machine_readable_json(tmp_path):
    destination = tmp_path / "calibration.json"
    reports = save_calibration_report(
        str(destination),
        history_lengths=(30,),
        seeds=(7,),
    )
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved[0]["history_length"] == 30
    assert saved[0]["seed_count"] == reports[0].seed_count
