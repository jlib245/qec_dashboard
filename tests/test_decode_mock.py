# tests/test_decode_mock.py
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.decode import run_decode


class TestRunDecodeMock(unittest.TestCase):
    """run_decode() Mock 테스트. qec_sim을 mock으로 격리하고 디코더 선택 분기를 검증한다."""

    def setUp(self):
        shots = 10
        syndromes = np.zeros((shots, 8), dtype=np.uint8)
        observables = np.zeros((shots, 1), dtype=np.uint8)
        predictions = np.zeros((shots, 1), dtype=np.uint8)
        predictions[:3, 0] = 1  # 10 shots 중 3개 오류 → LER = 0.3

        self.mock_circuit = Mock()
        self.mock_circuit.detector_error_model.return_value = Mock()

        self.mock_builder = Mock()
        self.mock_builder.build.return_value = self.mock_circuit

        self.mock_simulator = Mock()
        self.mock_simulator.generate_data.return_value = {
            "syndromes": syndromes,
            "observables": observables,
        }

        self.mock_decoder = Mock()
        self.mock_decoder.decode_batch.return_value = predictions

    def _patch(self):
        return patch.multiple(
            "app.decode",
            build_circuit=Mock(return_value=self.mock_builder),
            CircuitNoiseSimulator=Mock(return_value=self.mock_simulator),
            MWPMDecoder=Mock(return_value=self.mock_decoder),
        )

    def test_mwpm_uses_given_geometry(self):
        """mwpm은 입력 distance/rounds를 그대로 사용하고 LER을 계산한다"""
        with self._patch():
            r = run_decode(decoder="mwpm", distance=5, rounds=5, p_gate=0.01, p_meas=0.01, shots=10)
        self.assertEqual(r["decoder"], "mwpm")
        self.assertEqual(r["distance"], 5)
        self.assertEqual(r["rounds"], 5)
        self.assertAlmostEqual(r["ler"], 0.3)

    def test_mwpm_decode_batch_single_arg(self):
        """erasure 경로 제거 — decode_batch는 syndromes 인자 하나만 받아야 한다"""
        with self._patch():
            run_decode(decoder="mwpm", distance=3, rounds=3, p_gate=0.01, p_meas=0.01, shots=10)
        args, _ = self.mock_decoder.decode_batch.call_args
        self.assertEqual(len(args), 1)

    @patch("app.model_loader.load")
    def test_neural_uses_model_geometry(self, mock_load):
        """neural은 입력 distance를 무시하고 모델 자신의 geometry를 사용한다"""
        mock_load.return_value = {
            "decoder": self.mock_decoder, "distance": 3, "rounds": 3, "run_id": "abc12345",
        }
        with self._patch():
            r = run_decode(
                decoder="models:/mlp_d3@champion",
                distance=5, rounds=5,  # 무시되어야 함
                p_gate=0.01, p_meas=0.01, shots=10,
            )
        self.assertEqual(r["decoder"], "models:/mlp_d3@champion")
        self.assertEqual(r["distance"], 3)   # 모델 geometry
        self.assertEqual(r["rounds"], 3)
        self.assertEqual(r["run_id"], "abc12345")
        self.assertAlmostEqual(r["ler"], 0.3)
        mock_load.assert_called_once_with("models:/mlp_d3@champion")


if __name__ == "__main__":
    unittest.main()
