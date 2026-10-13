"""
Integration test for automated deterministic end-to-end pipeline verification.
"""

import pytest
from pathlib import Path
from scripts.verify_pipeline import verify_end_to_end_pipeline


def test_pipeline_verification_suite():
    results = verify_end_to_end_pipeline()
    assert results["overall_status"] == "PASSED"
    assert results["calls_evaluated"] >= 2
    assert results["aggregate_audit"]["overall_grounding_pass_rate"] > 0.90
    assert results["total_execution_seconds"] < 60.0

    # Ensure artifacts were generated
    reports_dir = Path("reports")
    assert (reports_dir / "pipeline_verification.json").exists()
    assert (reports_dir / "PIPELINE_VERIFICATION_REPORT.md").exists()
