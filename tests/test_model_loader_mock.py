# tests/test_model_loader_mock.py
import unittest
from unittest.mock import patch, MagicMock

import app.model_loader as ml


class TestModelLoaderMock(unittest.TestCase):
    """model_loader Mock 테스트. mlflow/qec_sim을 patch해 로직만 격리 검증한다."""

    def setUp(self):
        ml._decoder = None  # lazy singleton 초기화

    def tearDown(self):
        ml._decoder = None

    def test_get_decoder_builds_neural_decoder_with_injected_core(self):
        """registry core를 wrapper.core_model에 주입하고 NeuralDecoder로 감싼다"""
        core = MagicMock()
        wrapped = MagicMock()
        built = MagicMock()
        with patch.object(ml, "mlflow") as mock_mlflow, \
             patch.object(ml, "ExperimentConfig"), \
             patch.object(ml, "ComponentFactory") as mock_factory, \
             patch.object(ml, "NeuralDecoder", return_value=built) as mock_nd:
            mock_mlflow.pytorch.load_model.return_value = core
            mock_factory.build_system.return_value = (None, wrapped)
            dec = ml.get_decoder()

        self.assertIs(dec, built)
        wrapped.core_model.load_state_dict.assert_called_once()
        mock_nd.assert_called_once_with(model=wrapped)

    def test_get_decoder_is_cached(self):
        """두 번째 호출은 registry를 다시 로드하지 않고 캐시를 반환한다"""
        with patch.object(ml, "mlflow") as mock_mlflow, \
             patch.object(ml, "ExperimentConfig"), \
             patch.object(ml, "ComponentFactory") as mock_factory, \
             patch.object(ml, "NeuralDecoder", return_value=MagicMock()):
            mock_factory.build_system.return_value = (None, MagicMock())
            d1 = ml.get_decoder()
            d2 = ml.get_decoder()

        self.assertIs(d1, d2)
        mock_mlflow.pytorch.load_model.assert_called_once()

    def test_get_model_info_returns_empty_on_failure(self):
        """조회 실패 시 빈 dict을 반환한다 (서빙이 죽지 않도록)"""
        with patch.object(ml, "mlflow") as mock_mlflow:
            mock_mlflow.set_tracking_uri.side_effect = RuntimeError("no server")
            info = ml.get_model_info()
        self.assertEqual(info, {})


if __name__ == "__main__":
    unittest.main()
