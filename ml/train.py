# ml/train.py
import os
import argparse

import numpy as np
import torch
import mlflow
from mlflow import MlflowClient

from qec_sim.trainer.pipeline import TrainingPipeline
from qec_sim.trainer.factory import ComponentFactory
from qec_sim.circuit.registry import build_circuit
from qec_sim.circuit.simulator import CircuitNoiseSimulator
from qec_sim.decoders.neural import NeuralDecoder


def _test_ler(pipeline, wrapped, shots_per_noise: int = 5000) -> float:
    """학습/모델선택에 안 쓴 fresh held-out shots로 LER 측정 (배포 판정용 unbiased metric).

    val_ler은 best epoch을 고르는 데 쓰여 선택 편향이 있으므로, 승격 기준은 이 test_ler.
    config의 노이즈 조합 전체에 대해 새로 생성해 평균.
    """
    cfg = pipeline.config
    decoder = NeuralDecoder(model=wrapped)
    errs = total = 0
    for noise in cfg.get_expanded_noise_configs():
        circuit = build_circuit(cfg.code.name, cfg.code, noise).build()
        data = CircuitNoiseSimulator(circuit, noise).generate_data(shots=shots_per_noise)
        preds = decoder.decode_batch(data["syndromes"], batch_size=4096)
        errs += int(np.any(preds != data["observables"], axis=1).sum())
        total += len(data["observables"])
    return errs / total


def _promote_if_better(client, name, new_version, new_test_ler):
    """학습 시점 자동 교체: 신규 모델 test_ler이 현재 champion보다 낮으면 champion으로 승격.
    (champion이 없으면 첫 champion으로 지정. test_ler 낮을수록 좋음.)"""
    champ_ler = None
    try:
        champ = client.get_model_version_by_alias(name, "champion")
        champ_ler = client.get_run(champ.run_id).data.metrics.get("test_ler")
    except Exception:
        champ = None

    if champ is None or champ_ler is None or new_test_ler < champ_ler:
        client.set_registered_model_alias(name, "champion", str(new_version))
        base = f"test_ler={new_test_ler:.4f}"
        why = "첫 champion" if champ_ler is None else f"< {champ_ler:.4f}"
        print(f"[promote] champion -> v{new_version} ({base}, {why})")
    else:
        print(f"[promote] champion 유지 (v{new_version} test_ler={new_test_ler:.4f} >= {champ_ler:.4f})")


def _register_best_model(pipeline) -> None:
    """best 모델을 held-out test_ler과 함께 등록하고, 더 좋으면 champion으로 자동 교체."""
    cfg = pipeline.config.mlflow
    name = cfg.registered_model_name or (
        f"{pipeline.config.model.name}_d{pipeline.config.code.distance}"
    )

    best_path = pipeline.workspace["best_model"]
    _, wrapped = ComponentFactory.build_system(pipeline.config)
    wrapped.load_state_dict(torch.load(best_path, map_location="cpu"))
    wrapped.eval()
    core = getattr(wrapped, "core_model", wrapped)

    # 배포 판정용 held-out 평가
    test_ler = _test_ler(pipeline, wrapped)

    client = MlflowClient()
    last = mlflow.last_active_run()
    with mlflow.start_run(run_id=last.info.run_id):
        mlflow.log_metric("test_ler", test_ler)
        info = mlflow.pytorch.log_model(core, name="model", serialization_format="pickle")
        mv = mlflow.register_model(info.model_uri, name)
        if cfg.register_alias:
            client.set_registered_model_alias(name, cfg.register_alias, mv.version)

    alias = f" @{cfg.register_alias}" if cfg.register_alias else ""
    print(f"[train] registered {name} v{mv.version}{alias} | test_ler={test_ler:.4f}")

    # 학습 시점 자동 교체 (test_ler 게이트)
    _promote_if_better(client, name, mv.version, test_ler)


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

    # 등록은 우리가 처리 (qec_sim native 등록은 mlflow 3.x 비호환). 학습 후 직접.
    want_register = pipeline.config.mlflow.register_model
    pipeline.config.mlflow.register_model = False

    pipeline.run()

    if pipeline.config.mlflow.enable and want_register:
        _register_best_model(pipeline)


if __name__ == "__main__":
    main()
