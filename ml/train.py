# ml/train.py
import os
import argparse

import torch
import mlflow
from mlflow import MlflowClient

from qec_sim.trainer.pipeline import TrainingPipeline
from qec_sim.trainer.factory import ComponentFactory


def _register_best_model(pipeline) -> None:
    """학습된 best 모델을 MLflow Model Registry에 등록한다 (대시보드 소유 단계).

    qec_sim v0.1.0의 native 등록부는 mlflow 3.x와 비호환(pt2 기본값)이라,
    여기서 serialization_format="pickle"로 직접 log_model → register → alias 한다.
    registry에는 wrapper가 아닌 core_model만 올린다(서빙 시 wrapper 재조립).
    """
    cfg = pipeline.config.mlflow
    name = cfg.registered_model_name or (
        f"{pipeline.config.model.name}_d{pipeline.config.code.distance}"
    )

    # best_model.pth(= wrapper state_dict)를 다시 조립해 core_model만 추출.
    best_path = pipeline.workspace["best_model"]
    _, wrapped = ComponentFactory.build_system(pipeline.config)
    wrapped.load_state_dict(torch.load(best_path, map_location="cpu"))
    core = getattr(wrapped, "core_model", wrapped)
    core.eval()

    # qec_sim이 방금 닫은 run에 이어붙여(model + metric을 한 run에) 등록한다.
    last = mlflow.last_active_run()
    with mlflow.start_run(run_id=last.info.run_id):
        info = mlflow.pytorch.log_model(
            core, name="model", serialization_format="pickle"
        )
        mv = mlflow.register_model(info.model_uri, name)
        if cfg.register_alias:
            MlflowClient().set_registered_model_alias(name, cfg.register_alias, mv.version)

    alias = f" @{cfg.register_alias}" if cfg.register_alias else ""
    print(f"[train] registered {name} v{mv.version}{alias}")


def main():
    parser = argparse.ArgumentParser(
        description="qec_sim + MLflow로 neural QEC 디코더를 학습/등록한다."
    )
    parser.add_argument("--config", required=True, help="experiment YAML 경로")
    args = parser.parse_args()

    pipeline = TrainingPipeline(args.config)

    # ngrok 무료 도메인은 매번 바뀌므로 yaml에 박지 않고 env로 주입.
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        pipeline.config.mlflow.tracking_uri = uri
        print(f"[train] MLFLOW_TRACKING_URI override -> {uri}")

    # 등록 의도(register_model)는 우리가 처리한다. qec_sim의 native 등록은
    # mlflow 3.x 비호환이라 꺼두고(파라미터/메트릭 로깅만 사용), 등록은 학습 후 직접.
    want_register = pipeline.config.mlflow.register_model
    pipeline.config.mlflow.register_model = False

    pipeline.run()

    if pipeline.config.mlflow.enable and want_register:
        _register_best_model(pipeline)


if __name__ == "__main__":
    main()
