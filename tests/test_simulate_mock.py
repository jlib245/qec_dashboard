# tests/test_simulate_mock.py
import unittest
from unittest.mock import Mock, patch
import numpy as np

from app.simulate import run_simulation


class TestRunSimulationMock(unittest.TestCase):
    """
    run_simulation()의 Mock 테스트.
    qec_sim 라이브러리를 Mock으로 대체하여
    app/simulate.py 내부 로직만 격리해서 검증한다.
    """

    def setUp(self):
        self.base_params = dict(
            distance=3,
            rounds=3,
            p_gate=0.01,
            p_meas=0.01,
            p_leak=0.0,
            shots=10,
        )

        # Mock 객체 공통 설정
        shots = 10
        syndromes   = np.zeros((shots, 10), dtype=np.uint8)
        observables = np.zeros((shots, 1),  dtype=np.uint8)
        erasures    = np.zeros((shots, 10), dtype=np.uint8)
        predictions = np.zeros((shots, 1),  dtype=np.uint8)
        predictions[:2, 0] = 1  # 10 shots 중 2개 오류 → LER = 0.2

        self.mock_circuit = Mock()
        self.mock_circuit.detector_error_model.return_value = Mock()

        self.mock_builder = Mock()
        self.mock_builder.build.return_value = self.mock_circuit

        self.mock_simulator = Mock()
        self.mock_simulator.generate_data.return_value = {
            "syndromes":   syndromes,
            "observables": observables,
            "erasures":    erasures,
        }

        self.mock_decoder = Mock()
        self.mock_decoder.decode_batch.return_value = predictions

    # ── 호출 여부 검증 ───────────────────────────

    @patch("app.simulate.ErasureMWPM")
    @patch("app.simulate.CircuitNoiseSimulator")
    @patch("app.simulate.build_circuit")
    def test_build_circuit_called_with_surface_code(self, mock_build, mock_sim_cls, mock_decoder_cls):
        """build_circuit()이 'surface_code'로 호출되는지 확인"""
        mock_build.return_value = self.mock_builder
        mock_sim_cls.return_value = self.mock_simulator
        mock_decoder_cls.return_value = self.mock_decoder

        run_simulation(**self.base_params)

        self.assertEqual(mock_build.call_args[0][0], "surface_code")

    @patch("app.simulate.ErasureMWPM")
    @patch("app.simulate.CircuitNoiseSimulator")
    @patch("app.simulate.build_circuit")
    def test_generate_data_called_with_shots(self, mock_build, mock_sim_cls, mock_decoder_cls):
        """generate_data()가 shots 값과 함께 호출되는지 확인"""
        mock_build.return_value = self.mock_builder
        mock_sim_cls.return_value = self.mock_simulator
        mock_decoder_cls.return_value = self.mock_decoder

        run_simulation(**self.base_params)

        self.mock_simulator.generate_data.assert_called_with(shots=10)

    @patch("app.simulate.ErasureMWPM")
    @patch("app.simulate.CircuitNoiseSimulator")
    @patch("app.simulate.build_circuit")
    def test_decode_batch_called(self, mock_build, mock_sim_cls, mock_decoder_cls):
        """decode_batch()가 호출되는지 확인"""
        mock_build.return_value = self.mock_builder
        mock_sim_cls.return_value = self.mock_simulator
        mock_decoder_cls.return_value = self.mock_decoder

        run_simulation(**self.base_params)

        self.assertEqual(self.mock_decoder.decode_batch.called, True)

    # ── 반환값 검증 ────────────────────────────────────────────

    @patch("app.simulate.ErasureMWPM")
    @patch("app.simulate.CircuitNoiseSimulator")
    @patch("app.simulate.build_circuit")
    def test_ler_calculated_correctly(self, mock_build, mock_sim_cls, mock_decoder_cls):
        """predictions와 observables 차이로 LER이 올바르게 계산되는지 확인"""
        mock_build.return_value = self.mock_builder
        mock_sim_cls.return_value = self.mock_simulator
        mock_decoder_cls.return_value = self.mock_decoder

        result = run_simulation(**self.base_params)

        # 10 shots 중 2개 오류 → LER = 0.2
        self.assertEqual(result["ler"], 0.2)


if __name__ == "__main__":
    unittest.main()
