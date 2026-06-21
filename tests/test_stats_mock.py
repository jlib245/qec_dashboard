# tests/test_stats_mock.py
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.stats import run_stats


class TestRunStatsMock(unittest.TestCase):
    """run_stats() Mock 테스트.
    qec-sim/stim을 Mock으로 대체하여 app/stats.py 내부 로직만 격리해서 검증한다."""

    def setUp(self):
        self.params = dict(
            distance=3, rounds=3,
            p_gate=0.01, p_meas=0.01, shots=10,
        )

        # Mock 회로: 4 data qubit + 1 ancilla
        # detector 위치인 (2,2)는 data qubit에서 제외되어야 함
        self.mock_circuit = Mock()
        self.mock_circuit.get_final_qubit_coordinates.return_value = {
            0: (1, 1), 1: (1, 3), 2: (3, 1), 3: (3, 3),  # data
            4: (2, 2),                                      # ancilla (detector)
        }
        self.mock_circuit.get_detector_coordinates.return_value = {
            0: (2, 2, 0),
        }

        self.mock_builder = Mock()
        self.mock_builder.build.return_value = self.mock_circuit

        # FlipSimulator: shots=10, 모든 shot에서 qid 0(data)만 flipped → 평균 1 error/shot
        fake_flips = np.zeros((10, 5), dtype=np.uint8)
        fake_flips[:, 0] = 1
        self.mock_flip_sim = Mock()
        self.mock_flip_sim.peek_pauli_flips.return_value = fake_flips

    @patch("app.stats.stim")
    @patch("app.stats.build_circuit")
    def test_returns_required_keys(self, mock_build, mock_stim):
        """반환 dict에 avg_errors, avg_error_rate, n_data_qubits 키가 있어야 한다"""
        mock_build.return_value = self.mock_builder
        mock_stim.FlipSimulator.return_value = self.mock_flip_sim

        result = run_stats(**self.params)

        for key in ("avg_errors", "avg_error_rate", "n_data_qubits"):
            self.assertEqual(key in result, True)

    @patch("app.stats.stim")
    @patch("app.stats.build_circuit")
    def test_n_data_qubits_excludes_detectors(self, mock_build, mock_stim):
        """detector 위치의 qubit은 data qubit 카운트에서 제외되어야 한다"""
        mock_build.return_value = self.mock_builder
        mock_stim.FlipSimulator.return_value = self.mock_flip_sim

        result = run_stats(**self.params)
        # 5 qubit 중 1개가 detector 위치 → data 4개
        self.assertEqual(result["n_data_qubits"], 4)

    @patch("app.stats.stim")
    @patch("app.stats.build_circuit")
    def test_avg_errors_calculated(self, mock_build, mock_stim):
        """모든 shot의 qid 0이 flipped → avg_errors == 1.0"""
        mock_build.return_value = self.mock_builder
        mock_stim.FlipSimulator.return_value = self.mock_flip_sim

        result = run_stats(**self.params)
        self.assertEqual(result["avg_errors"], 1.0)

    @patch("app.stats.stim")
    @patch("app.stats.build_circuit")
    def test_avg_error_rate_consistent(self, mock_build, mock_stim):
        """avg_error_rate == round(avg_errors / n_data_qubits, 4)"""
        mock_build.return_value = self.mock_builder
        mock_stim.FlipSimulator.return_value = self.mock_flip_sim

        result = run_stats(**self.params)
        expected = round(result["avg_errors"] / result["n_data_qubits"], 4)
        self.assertEqual(result["avg_error_rate"], expected)


if __name__ == "__main__":
    unittest.main()
