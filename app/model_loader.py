# app/model_loader.py
from app import config

try:
    import mlflow
    import mlflow.artifacts
    from mlflow import MlflowClient
    from qec_sim.config.schema import ExperimentConfig
    from qec_sim.trainer.factory import ComponentFactory
    from qec_sim.decoders.neural import NeuralDecoder
except ImportError:
    # mlflow/qec-sim 미설치 환경(mock CI 등): 테스트에서 patch되거나 호출 시 에러로 드러남
    mlflow = None
    MlflowClient = None
    ExperimentConfig = ComponentFactory = NeuralDecoder = None

_cache = {}  # model_uri -> {"decoder", "distance", "rounds", "run_id"}


def list_models() -> list:
    """MLflow registry의 등록 모델 + alias 목록을 반환 (UI 드롭다운용).

    mlflow 미설정/연결 실패 시 빈 리스트 (UI는 MWPM만 보여줌).
    """
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client = MlflowClient()
        out = []
        for m in client.search_registered_models():
            for alias, version in (m.aliases or {}).items():
                # geometry는 run params(code.distance/code.rounds)에서 싸게 읽는다.
                distance = rounds = None
                try:
                    mv = client.get_model_version(m.name, str(version))
                    params = client.get_run(mv.run_id).data.params
                    distance = int(params["code.distance"])
                    rounds = int(params["code.rounds"])
                except Exception:
                    pass
                out.append({
                    "uri": f"models:/{m.name}@{alias}",
                    "name": m.name,
                    "alias": alias,
                    "version": str(version),
                    "distance": distance,
                    "rounds": rounds,
                })
        return out
    except Exception:
        return []


def load(model_uri: str) -> dict:
    """model_uri의 모델을 로드해 NeuralDecoder + geometry로 캐싱(uri별).

    registry에는 core_model만 있으므로, 모델 자신의 run에 기록된 config.yaml로
    wrapper를 재조립한다 (self-describing — geometry가 모델마다 자동으로 맞춰짐).
    """
    if model_uri not in _cache:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        core = mlflow.pytorch.load_model(model_uri)
        run_id = mlflow.models.get_model_info(model_uri).run_id

        # 실제로 서빙되는 버전 resolve (alias는 움직이므로 로드 시점 버전을 기록)
        version = None
        try:
            ref = model_uri.replace("models:/", "")
            if "@" in ref:
                mname, alias = ref.split("@")
                version = MlflowClient().get_model_version_by_alias(mname, alias).version
            elif "/" in ref:  # models:/name/3
                version = ref.split("/")[1]
        except Exception:
            pass

        try:
            cfg_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="config.yaml"
            )
            exp = ExperimentConfig.from_yaml(cfg_path)
        except Exception:
            # run에 config.yaml이 없으면 로컬 기본 config로 폴백
            exp = ExperimentConfig.from_yaml(config.BUNDLED_CONFIG)

        _, wrapped = ComponentFactory.build_system(exp)
        wrapped.core_model.load_state_dict(core.state_dict())
        wrapped.eval()

        _cache[model_uri] = {
            "decoder": NeuralDecoder(model=wrapped),
            "distance": exp.code.distance,
            "rounds": exp.code.rounds,
            "run_id": run_id,
            "version": version,
        }
    return _cache[model_uri]
