"""Regression tests for deletion of call-scoped runtime artifacts."""

from src.callintel.api.routes_calls import cleanup_runtime_artifacts
from src.callintel.core.config import settings


def test_call_deletion_removes_only_call_scoped_artifacts(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    reports = tmp_path / "reports"
    cache = tmp_path / "data" / "cache"
    uploads.mkdir()
    reports.mkdir()
    cache.mkdir(parents=True)

    monkeypatch.setattr(settings, "upload_dir", uploads)
    monkeypatch.setattr(settings, "reports_dir", reports)
    monkeypatch.setattr(settings, "base_dir", tmp_path)

    target_upload = uploads / "CALL_ABC123_customer.wav"
    other_upload = uploads / "CALL_DEF456_other.wav"
    target_report = reports / "call_report_CALL_ABC123.json"
    shared_report = reports / "benchmark_summary.html"
    target_cache = cache / "CALL_ABC123_embeddings.bin"
    for path in (target_upload, other_upload, target_report, shared_report, target_cache):
        path.write_text("fixture", encoding="utf-8")

    deleted = cleanup_runtime_artifacts("CALL_ABC123")

    assert deleted == 3
    assert not target_upload.exists()
    assert not target_report.exists()
    assert not target_cache.exists()
    assert other_upload.exists()
    assert shared_report.exists()


def test_history_cleanup_removes_staged_uploads_but_preserves_shared_reports(
    tmp_path, monkeypatch
):
    uploads = tmp_path / "uploads"
    reports = tmp_path / "reports"
    uploads.mkdir()
    reports.mkdir()

    monkeypatch.setattr(settings, "upload_dir", uploads)
    monkeypatch.setattr(settings, "reports_dir", reports)
    monkeypatch.setattr(settings, "base_dir", tmp_path)

    (uploads / "CALL_ABC123_customer.wav").write_text("audio", encoding="utf-8")
    (uploads / "orphaned_partial_upload.tmp").write_text("partial", encoding="utf-8")
    shared_report = reports / "benchmark_summary.html"
    shared_report.write_text("keep", encoding="utf-8")

    deleted = cleanup_runtime_artifacts()

    assert deleted == 2
    assert list(uploads.iterdir()) == []
    assert shared_report.exists()
