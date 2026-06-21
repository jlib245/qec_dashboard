# app/model_loader.py
from app import config

try:
    import mlflow
    from qec_sim.config.schema import ExperimentConfig
    from qec_sim.trainer.factory import ComponentFactory
    from qec_sim.decoders.neural import NeuralDecoder
except ImportError:
    # mlflow/qec-sim 미설치 환경(mock CI 등): 테스트에서 patch되거나 호출 시 에러로 드러남
    mlflow = None
    ExperimentConfig = ComponentFactory = NeuralDecoder = None

_decoder = None  # lazy singleton (프로세스당 1회 로드)


def get_decoder():
    """MLflow registry에서 학습된 모델을 로드해 NeuralDecoder로 반환 (캐싱).

    registry에는 core_model만 등록되므로, BUNDLED_CONFIG로 wrapper를 재조립한 뒤
    로드한 가중치를 core_model에 주입한다 (eval_pipeline와 동일한 추론 경로).
    """
    global _decoder
    if _decoder is None:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        core = mlflow.pytorch.load_model(config.MODEL_URI)

        exp = ExperimentConfig.from_yaml(config.BUNDLED_CONFIG)
        _, wrapped = ComponentFactory.build_system(exp)
        wrapped.core_model.load_state_dict(core.state_dict())
        wrapped.eval()
        _decoder = NeuralDecoder(model=wrapped)
    return _decoder


def get_model_info() -> dict:
    """현재 서빙 중인 모델의 메타데이터 반환 (실패 시 빈 dict)."""
    try:
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        # MODEL_URI 형식: "models:/<name>@<alias>"
        name = config.MODEL_URI.split("models:/")[1].split("@")[0]
        alias = config.MODEL_URI.split("@")[1]
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        return {"name": name, "alias": alias, "version": mv.version, "run_id": mv.run_id}
    except Exception:
        return {}
