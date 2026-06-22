# tests/test_visualize_mock.py
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.visualize import run_visualize


class TestRunVisualizeMock(unittest.TestCase):
    """run_visualize() Mock 테스트.
    qec-sim/stim을 Mock으로 대체하여 app/visualize.py 내부 로직만 격리해서 검증한다."""

    def setUp(self):
        self.params = dict(
            distance=3, rounds=2,
            p_gate=0.01, p_meas=0.01,
        )

        # Mock 회로: 4 data qubit + 2 ancilla (Z 1, X 1)
        # data qubit: (1,1), (1,3), (3,1), (3,3)  — odd/odd
        # ancilla:   (2,2)  → (x+y)%4 == 0 → Z
        #             (2,4) → (x+y)%4 == 2 → X
        self.mock_circuit = Mock()
        self.mock_circuit.get_final_qubit_coordinates.return_value = {
            0: (1, 1), 1: (1, 3), 2: (3, 1), 3: (3, 3),
            4: (2, 2), 5: (2, 4),
        }
        self.mock_circuit.get_detector_coordinates.return_value = {
            0: (2, 2, 0),  # round 0: Z
            1: (2, 2, 1),  # round 1: Z
            2: (2, 4, 1),  # round 1: X
        }
        self.mock_circuit.detector_error_model.return_value = Mock()

        self.mock_builder = Mock()
        self.mock_builder.build.return_value = self.mock_circuit

        # FlipSimulator
        self.mock_flip_sim = Mock()
        self.mock_flip_sim.get_detector_flips.return_value = np.array([[1, 1, 0]], dtype=np.uint8)
        self.mock_flip_sim.get_observable_flips.return_value = np.array([[0]], dtype=np.uint8)
        # 6 qubit, qid 0만 flipped
        self.mock_flip_sim.peek_pauli_flips.return_value = [np.array([1, 0, 0, 0, 0, 0])]

        # MWPMDecoder
        self.mock_decoder = Mock()
        self.mock_decoder.decode_single_with_correction.return_value = {
            "corrected_fault_ids": [0],
            "logical_error": [0],
        }
        self.mock_decoder.get_corrected_qubits.return_value = [{"x": 1, "y": 1}]

    def _patch_all(self):
        """3중 patch를 매 테스트마다 반복하지 않기 위한 헬퍼"""
        return patch.multiple(
            "app.visualize",
            stim=Mock(FlipSimulator=Mock(return_value=self.mock_flip_sim)),
            MWPMDecoder=Mock(return_value=self.mock_decoder),
            build_circuit=Mock(return_value=self.mock_builder),
        )

    def test_returns_required_keys(self):
        """반환 dict에 필수 키들이 모두 있어야 한다"""
        with self._patch_all():
            result = run_visualize(**self.params)
        for key in ("logical_error", "data_qubits", "ancillas",
                    "ancillas_by_round", "corrected_qubits", "edges"):
            self.assertEqual(key in result, True)

    def test_data_qubits_classified_by_parity(self):
        """data qubit은 odd/odd 좌표만 포함되어야 한다"""
        with self._patch_all():
            result = run_visualize(**self.params)
        self.assertEqual(len(result["data_qubits"]), 4)
        for d in result["data_qubits"]:
            self.assertEqual(int(d["x"]) % 2, 1)
            self.assertEqual(int(d["y"]) % 2, 1)

    def test_x_z_stabilizer_classification(self):
        """(x+y) % 4 == 2 → X, == 0 → Z"""
        with self._patch_all():
            result = run_visualize(**self.params)

        ancillas_round0 = result["ancillas_by_round"][0]
        types = {(a["x"], a["y"]): a["stabilizer_type"] for a in ancillas_round0}
        self.assertEqual(types[(2, 2)], "Z")
        self.assertEqual(types[(2, 4)], "X")

    def test_rounds_count_matches_ancillas_by_round(self):
        """ancillas_by_round 길이는 rounds 값과 같아야 한다"""
        with self._patch_all():
            result = run_visualize(**self.params)
        self.assertEqual(len(result["ancillas_by_round"]), self.params["rounds"])

    def test_logical_error_xor_false(self):
        """obs_flips=0, mwpm_pred=0 → logical_error=False"""
        with self._patch_all():
            result = run_visualize(**self.params)
        self.assertEqual(result["logical_error"], False)

    def test_logical_error_xor_true(self):
        """obs_flips=1, mwpm_pred=0 → logical_error=True"""
        self.mock_flip_sim.get_observable_flips.return_value = np.array([[1]], dtype=np.uint8)
        with self._patch_all():
            result = run_visualize(**self.params)
        self.assertEqual(result["logical_error"], True)

    def test_edges_connect_ancilla_to_data(self):
        """edges는 ancilla의 대각선 4방향 data qubit 연결.
        ancilla (2,2)는 데이터 4개와 모두 인접 → 4 edges
        ancilla (2,4)는 데이터 2개와만 인접 ((1,3),(3,3)) → 2 edges
        총 6 edges"""
        with self._patch_all():
            result = run_visualize(**self.params)
        self.assertEqual(len(result["edges"]), 6)


if __name__ == "__main__":
    unittest.main()
