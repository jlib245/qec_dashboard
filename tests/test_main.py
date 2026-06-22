# tests/test_main.py
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class TestMainRoutes(unittest.TestCase):
    """FastAPI 라우터 테스트.
    run_* 함수는 mock으로 대체하고 라우팅 및 응답 형태만 검증한다."""

    def setUp(self):
        self.client = TestClient(app)
        self.simulate_payload = {
            "distance": 3, "rounds": 3,
            "p_gate": 0.01, "p_meas": 0.01,
            "shots": 100,
        }
        self.visualize_payload = {
            "distance": 3, "rounds": 3,
            "p_gate": 0.01, "p_meas": 0.01,
        }

    def test_home_serves_html(self):
        """GET / 은 200과 HTML을 반환해야 한다"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual("<html" in response.text.lower(), True)

    @patch("app.main.run_simulation")
    def test_simulate_returns_200(self, mock_run):
        """POST /simulate 200 응답 + 결과 통과"""
        mock_run.return_value = {"ler": 0.1}
        response = self.client.post("/simulate", json=self.simulate_payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ler": 0.1})

    @patch("app.main.run_simulation")
    def test_simulate_passes_kwargs(self, mock_run):
        """payload 값이 run_simulation에 그대로 전달되어야 한다"""
        mock_run.return_value = {"ler": 0.0}
        self.client.post("/simulate", json=self.simulate_payload)
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["distance"], 3)
        self.assertEqual(kwargs["shots"], 100)

    @patch("app.main.run_simulation")
    def test_simulate_default_values(self, mock_run):
        """payload에 shots 없으면 기본값으로 호출되어야 한다"""
        mock_run.return_value = {"ler": 0.0}
        minimal = {"distance": 3, "rounds": 3, "p_gate": 0.01, "p_meas": 0.01}
        self.client.post("/simulate", json=minimal)
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["shots"], 1000)

    @patch("app.main.run_simulation")
    def test_simulate_failure_returns_500(self, mock_run):
        """run_*가 예외를 던지면 500 응답을 반환해야 한다"""
        mock_run.side_effect = RuntimeError("boom")
        response = self.client.post("/simulate", json=self.simulate_payload)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"], "Internal Server Error")

    @patch("app.main.create_github_issue")
    @patch("app.main.run_simulation")
    def test_simulate_failure_creates_issue(self, mock_run, mock_issue):
        """run_*가 예외를 던지면 GitHub Issue 생성이 호출되어야 한다"""
        mock_run.side_effect = RuntimeError("boom")
        self.client.post("/simulate", json=self.simulate_payload)
        mock_issue.assert_called_once()

    @patch("app.main.run_visualize")
    def test_visualize_returns_200(self, mock_run):
        """POST /visualize 200 응답"""
        mock_run.return_value = {
            "logical_error": False, "data_qubits": [],
            "ancillas": [], "ancillas_by_round": [],
            "corrected_qubits": [], "edges": [],
        }
        response = self.client.post("/visualize", json=self.visualize_payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual("logical_error" in response.json(), True)

    @patch("app.main.run_stats")
    def test_stats_returns_200(self, mock_run):
        """POST /stats 200 응답"""
        mock_run.return_value = {
            "avg_errors": 1.0, "avg_error_rate": 0.1, "n_data_qubits": 9,
        }
        response = self.client.post("/stats", json=self.simulate_payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["n_data_qubits"], 9)

    @patch("app.main.run_decode")
    def test_decode_returns_200(self, mock_run):
        """POST /decode 200 응답 + mode 통과"""
        mock_run.return_value = {"mode": "mwpm", "distance": 3, "rounds": 3, "ler": 0.1}
        response = self.client.post(
            "/decode", json={"p_gate": 0.01, "p_meas": 0.01, "shots": 100}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "mwpm")

    @patch("app.main.run_compare")
    def test_compare_returns_200(self, mock_run):
        """POST /compare 200 응답 + 두 디코더 LER 통과"""
        mock_run.return_value = {
            "distance": 3, "rounds": 3, "shots": 100,
            "mwpm_ler": 0.1, "neural_ler": 0.15, "neural_error": None,
        }
        response = self.client.post(
            "/compare", json={"p_gate": 0.01, "p_meas": 0.01, "shots": 100}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mwpm_ler"], 0.1)


if __name__ == "__main__":
    unittest.main()
