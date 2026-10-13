"""
Production Smoke Test & Deployment Readiness Validator.
Verifies operational endpoints, health probes, latency headers, and telemetry.
Can run against an in-process TestClient or a deployed HTTP target (e.g., http://localhost:8000).

Usage:
  python scripts/production_smoke_test.py [--target http://localhost:8000]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_smoke_tests(base_url: str | None = None) -> bool:
    print("=" * 70)
    print("CALLINTEL AI: PRODUCTION DEPLOYMENT SMOKE TEST")
    print("=" * 70)

    if base_url:
        print(f"[*] Target Environment: External HTTP Host -> {base_url}")
        import requests

        class ClientWrapper:
            def get(self, path, **kwargs):
                return requests.get(f"{base_url}{path}", **kwargs)

            def post(self, path, **kwargs):
                return requests.post(f"{base_url}{path}", **kwargs)

            def options(self, path, **kwargs):
                return requests.options(f"{base_url}{path}", **kwargs)

        client = ClientWrapper()
    else:
        print("[*] Target Environment: In-Process FastAPI TestClient")
        from fastapi.testclient import TestClient
        from src.callintel.api.app import app

        client = TestClient(app)

    passed = 0
    failed = 0

    def check(name: str, condition: bool, details: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {name} {details}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {details}")

    # 1. Liveness Probe
    try:
        r = client.get("/health/live")
        check(
            "Liveness Probe (/health/live)",
            r.status_code == 200 and r.json().get("status") == "ALIVE",
            f"(HTTP {r.status_code}, uptime={r.json().get('uptime_seconds')}s)",
        )
    except Exception as exc:
        check("Liveness Probe (/health/live)", False, f"Exception: {exc}")

    # 2. Readiness Probe
    try:
        r = client.get("/health/ready")
        checks = r.json().get("checks", {})
        db_ok = checks.get("database", {}).get("status") == "HEALTHY"
        models_ok = all(v == "LOADABLE" for v in checks.get("models", {}).values())
        check(
            "Readiness Probe (/health/ready)",
            r.status_code == 200 and db_ok and models_ok,
            f"(HTTP {r.status_code}, db={db_ok}, models_ready={models_ok})",
        )
    except Exception as exc:
        check("Readiness Probe (/health/ready)", False, f"Exception: {exc}")

    # 3. Telemetry & Metrics Probe (JSON)
    try:
        r = client.get("/metrics")
        check(
            "Telemetry Metrics JSON (/metrics)",
            r.status_code == 200 and "traffic" in r.json(),
            f"(HTTP {r.status_code}, total_requests={r.json().get('traffic', {}).get('total_requests')})",
        )
    except Exception as exc:
        check("Telemetry Metrics JSON (/metrics)", False, f"Exception: {exc}")

    # 4. Telemetry Metrics (Prometheus Text Exposition)
    try:
        r = client.get("/metrics", headers={"Accept": "text/plain"})
        check(
            "Telemetry Prometheus Plaintext (/metrics)",
            r.status_code == 200 and "app_uptime_seconds" in r.text,
            f"(HTTP {r.status_code}, length={len(r.text)} bytes)",
        )
    except Exception as exc:
        check("Telemetry Prometheus Plaintext (/metrics)", False, f"Exception: {exc}")

    # 5. Correlation ID & Latency Headers
    try:
        custom_trace = "smoke-test-trace-99"
        r = client.get("/health/live", headers={"X-Request-ID": custom_trace})
        has_req_id = r.headers.get("X-Request-ID") == custom_trace
        has_latency = "X-Response-Time-Ms" in r.headers
        check(
            "Telemetry Headers (X-Request-ID & X-Response-Time-Ms)",
            has_req_id and has_latency,
            f"(latency={r.headers.get('X-Response-Time-Ms')}ms)",
        )
    except Exception as exc:
        check("Telemetry Headers", False, f"Exception: {exc}")

    # 6. CORS Configuration Verification
    try:
        r = client.options(
            "/api/v1/calls/upload",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        check(
            "CORS Preflight (OPTIONS)",
            r.status_code in [200, 204],
            f"(HTTP {r.status_code})",
        )
    except Exception as exc:
        check("CORS Preflight", False, f"Exception: {exc}")

    print("-" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed.")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CallIntel AI Smoke Test")
    parser.add_argument("--target", type=str, default=None, help="Target URL (e.g. http://localhost:8000)")
    args = parser.parse_args()

    success = run_smoke_tests(args.target)
    sys.exit(0 if success else 1)
