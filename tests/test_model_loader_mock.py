# tests/test_model_loader_mock.py
import unittest
from unittest.mock import patch, MagicMock

import app.model_loader as ml


class TestModelLoaderMock(unittest.TestCase):
    """model_loader Mock 테스트. mlflow/qec_sim을 patch해 로직만 격리 검증한다."""

    def setUp(self):
        ml._cache.clear()

    def tearDown(self):
        ml._cache.clear()

    def _exp(self, d=3, r=3):
        exp = MagicMock()
        exp.code.distance = d
        exp.code.rounds = r
        return exp

    def test_load_builds_decoder_with_model_geometry(self):
        """모델 run의 config.yaml에서 geometry를 읽고 core를 wrapper에 주입한다"""
        built = MagicMock()
        wrapped = MagicMock()
        with patch.object(ml, "mlflow") as mock_mlflow, \
             patch.object(ml, "ExperimentConfig") as mock_exp_cls, \
             patch.object(ml, "ComponentFactory") as mock_factory, \
             patch.object(ml, "NeuralDecoder", return_value=built):
            mock_mlflow.pytorch.load_model.return_value = MagicMock()
            mock_mlflow.models.get_model_info.return_value = MagicMock(run_id="run123")
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/config.yaml"
            mock_exp_cls.from_yaml.return_value = self._exp(3, 3)
            mock_factory.build_system.return_value = (None, wrapped)

            rec = ml.load("models:/mlp_d3@champion")

        self.assertIs(rec["decoder"], built)
        self.assertEqual(rec["distance"], 3)
        self.assertEqual(rec["rounds"], 3)
        self.assertEqual(rec["run_id"], "run123")
        wrapped.core_model.load_state_dict.assert_called_once()

    def test_load_is_cached_per_uri(self):
        """같은 uri 두 번째 호출은 registry를 다시 로드하지 않는다"""
        with patch.object(ml, "mlflow") as mock_mlflow, \
             patch.object(ml, "ExperimentConfig") as mock_exp_cls, \
             patch.object(ml, "ComponentFactory") as mock_factory, \
             patch.object(ml, "NeuralDecoder", return_value=MagicMock()):
            mock_mlflow.pytorch.load_model.return_value = MagicMock()
            mock_mlflow.models.get_model_info.return_value = MagicMock(run_id="r")
            mock_mlflow.artifacts.download_artifacts.return_value = "/tmp/config.yaml"
            mock_exp_cls.from_yaml.return_value = self._exp()
            mock_factory.build_system.return_value = (None, MagicMock())

            ml.load("models:/mlp_d3@champion")
            ml.load("models:/mlp_d3@champion")

        self.assertEqual(mock_mlflow.pytorch.load_model.call_count, 1)

    def test_list_models_returns_empty_on_failure(self):
        """registry 조회 실패 시 빈 리스트 (UI는 MWPM만 보여줌)"""
        with patch.object(ml, "mlflow") as mock_mlflow:
            mock_mlflow.set_tracking_uri.side_effect = RuntimeError("no server")
            self.assertEqual(ml.list_models(), [])

    def test_list_models_includes_geometry(self):
        """등록 모델을 alias별로 나열하고 run params에서 geometry를 읽는다"""
        m = MagicMock()
        m.name = "mlp_d3"
        m.aliases = {"champion": 1}
        with patch.object(ml, "mlflow"), \
             patch.object(ml, "MlflowClient") as mock_client_cls:
            client = mock_client_cls.return_value
            client.search_registered_models.return_value = [m]
            client.get_model_version.return_value = MagicMock(run_id="r1")
            run = MagicMock()
            run.data.params = {"code.distance": "3", "code.rounds": "3"}
            client.get_run.return_value = run
            out = ml.list_models()

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["uri"], "models:/mlp_d3@champion")
        self.assertEqual(out[0]["distance"], 3)
        self.assertEqual(out[0]["rounds"], 3)


if __name__ == "__main__":
    unittest.main()
