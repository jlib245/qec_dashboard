# tests/test_decode_mock.py
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.decode import run_decode


class TestRunDecodeMock(unittest.TestCase):
    """run_decode() Mock 테스트. qec_sim을 mock으로 격리하고 MODEL_MODE 분기를 검증한다."""

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

    def test_mwpm_returns_expected_keys_and_ler(self):
        """mwpm 모드(기본)는 mode/distance/rounds/ler을 반환하고 LER을 정확히 계산한다"""
        with self._patch():
            result = run_decode(p_gate=0.01, p_meas=0.01, shots=10)
        self.assertEqual(result["mode"], "mwpm")
        self.assertEqual(result["distance"], 3)
        self.assertEqual(result["rounds"], 3)
        self.assertAlmostEqual(result["ler"], 0.3)

    def test_mwpm_decode_batch_called_without_erasures(self):
        """erasure 경로 제거 — decode_batch는 syndromes 인자 하나만 받아야 한다"""
        with self._patch():
            run_decode(p_gate=0.01, p_meas=0.01, shots=10)
        args, _ = self.mock_decoder.decode_batch.call_args
        self.assertEqual(len(args), 1)

    @patch("app.decode.config.MODEL_MODE", "neural")
    def test_neural_raises_not_implemented(self):
        """neural 모드는 아직 미구현 → NotImplementedError"""
        with self._patch():
            with self.assertRaises(NotImplementedError):
                run_decode(p_gate=0.01, p_meas=0.01, shots=10)

    @patch("app.decode.config.MODEL_MODE", "bogus")
    def test_unknown_mode_raises_value_error(self):
        """알 수 없는 MODEL_MODE → ValueError"""
        with self._patch():
            with self.assertRaises(ValueError):
                run_decode(p_gate=0.01, p_meas=0.01, shots=10)


if __name__ == "__main__":
    unittest.main()
